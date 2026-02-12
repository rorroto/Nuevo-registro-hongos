from __future__ import annotations

import base64
import hashlib
import secrets
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import altair as alt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DB_PATH = Path("data") / "registro_hongos.db"
BACKUPS_DIR = Path("backups")
APP_TIMEZONE = timezone(timedelta(hours=-6))
LOGO_PATH = Path("assets") / "logo_bio_funga.png"


def now_gmt6() -> datetime:
    return datetime.now(APP_TIMEZONE)


def inject_watermark_logo() -> None:
    if not LOGO_PATH.exists():
        return

    encoded_logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            background-image: url("data:image/png;base64,{encoded_logo}");
            background-repeat: no-repeat;
            background-position: center;
            background-size: min(55vw, 500px);
            opacity: 0.08;
            pointer-events: none;
            z-index: 0;
        }}

        .stApp > header,
        .stApp > div,
        .main,
        section.main > div {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS greenhouses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                greenhouse_id INTEGER NOT NULL,
                recorded_at TEXT NOT NULL,
                temp_max REAL NOT NULL,
                temp_min REAL NOT NULL,
                humidity_max REAL NOT NULL,
                humidity_min REAL NOT NULL,
                co2 REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (greenhouse_id) REFERENCES greenhouses(id) ON DELETE CASCADE
            )
            """
        )


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000
    ).hex()


def create_user(username: str, password: str) -> None:
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
            (username.strip().lower(), password_hash, salt, now_gmt6().isoformat()),
        )


def authenticate_user(username: str, password: str) -> bool:
    with get_connection() as conn:
        user = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?",
            (username.strip().lower(),),
        ).fetchone()
    if not user:
        return False
    return hash_password(password, user["salt"]) == user["password_hash"]


def get_setting(key: str, default: str) -> str:
    with get_connection() as conn:
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO app_settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )


def run_scheduled_backup(interval_hours: int = 24) -> tuple[bool, str]:
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        return False, "La base de datos aún no existe."

    last_backup_iso = get_setting("last_backup_at", "")
    now = now_gmt6()
    if last_backup_iso:
        last_backup = datetime.fromisoformat(last_backup_iso)
        if now - last_backup < timedelta(hours=interval_hours):
            return False, last_backup.strftime("%Y-%m-%d %H:%M")

    backup_name = f"registro_hongos_{now.strftime('%Y%m%d_%H%M%S')}.db"
    backup_path = BACKUPS_DIR / backup_name
    shutil.copy2(DB_PATH, backup_path)
    set_setting("last_backup_at", now.isoformat())

    backups = sorted(BACKUPS_DIR.glob("registro_hongos_*.db"), reverse=True)
    for old_backup in backups[30:]:
        old_backup.unlink(missing_ok=True)

    return True, now.strftime("%Y-%m-%d %H:%M")


def _pchip_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)

    for k in range(1, n - 1):
        if delta[k - 1] == 0 or delta[k] == 0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            d[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    d[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]) if n > 2 else delta[0]
    d[-1] = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2]) if n > 2 else delta[-1]

    y_new = np.empty_like(x_new)
    idx = np.searchsorted(x, x_new) - 1
    idx = np.clip(idx, 0, n - 2)

    xk = x[idx]
    xk1 = x[idx + 1]
    yk = y[idx]
    yk1 = y[idx + 1]
    dk = d[idx]
    dk1 = d[idx + 1]
    hk = xk1 - xk

    t = (x_new - xk) / hk
    t2 = t * t
    t3 = t2 * t

    y_new[:] = (
        (2 * t3 - 3 * t2 + 1) * yk
        + (t3 - 2 * t2 + t) * hk * dk
        + (-2 * t3 + 3 * t2) * yk1
        + (t3 - t2) * hk * dk1
    )
    return y_new


def smooth_temperature_series(date_series: pd.Series, temp_series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = mdates.date2num(pd.to_datetime(date_series).to_pydatetime())
    y = temp_series.to_numpy(dtype=float)

    if len(x) < 3 or np.any(np.diff(x) <= 0):
        return x, y

    x_dense = np.linspace(x.min(), x.max(), num=max(180, len(x) * 30))
    y_dense = _pchip_interpolate(x, y, x_dense)
    return x_dense, y_dense


def get_greenhouses() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query("SELECT id, name, created_at FROM greenhouses ORDER BY name ASC", conn)


def create_greenhouse(name: str) -> None:
    with get_connection() as conn:
        conn.execute("INSERT INTO greenhouses (name, created_at) VALUES (?, ?)", (name.strip(), now_gmt6().isoformat()))


def update_greenhouse(greenhouse_id: int, new_name: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE greenhouses SET name = ? WHERE id = ?", (new_name.strip(), greenhouse_id))


def delete_greenhouse(greenhouse_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM readings WHERE greenhouse_id = ?", (greenhouse_id,))
        conn.execute("DELETE FROM greenhouses WHERE id = ?", (greenhouse_id,))


def add_reading(greenhouse_id: int, recorded_at: datetime, temp_max: float, temp_min: float, humidity_max: float, humidity_min: float, co2: float) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO readings (greenhouse_id, recorded_at, temp_max, temp_min, humidity_max, humidity_min, co2, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (greenhouse_id, recorded_at.isoformat(), temp_max, temp_min, humidity_max, humidity_min, co2, now_gmt6().isoformat()),
        )


def get_readings(greenhouse_id: int) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT id, greenhouse_id, recorded_at, temp_max, temp_min, humidity_max, humidity_min, co2
            FROM readings WHERE greenhouse_id = ? ORDER BY recorded_at DESC
            """,
            conn,
            params=(greenhouse_id,),
        )
    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    return df


def update_reading(reading_id: int, recorded_at: datetime, temp_max: float, temp_min: float, humidity_max: float, humidity_min: float, co2: float) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE readings
            SET recorded_at = ?, temp_max = ?, temp_min = ?, humidity_max = ?, humidity_min = ?, co2 = ?
            WHERE id = ?
            """,
            (recorded_at.isoformat(), temp_max, temp_min, humidity_max, humidity_min, co2, reading_id),
        )


def delete_reading(reading_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM readings WHERE id = ?", (reading_id,))


def get_daily_summary(greenhouse_id: int) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT DATE(recorded_at) AS date,
                   AVG(temp_max) AS temp_max_avg,
                   AVG(temp_min) AS temp_min_avg,
                   AVG(humidity_max) AS humidity_max_avg,
                   AVG(humidity_min) AS humidity_min_avg,
                   AVG(co2) AS co2_avg,
                   COUNT(*) AS readings_count
            FROM readings
            WHERE greenhouse_id = ?
            GROUP BY DATE(recorded_at)
            ORDER BY DATE(recorded_at) ASC
            """,
            conn,
            params=(greenhouse_id,),
        )

    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["temp_avg"] = (df["temp_max_avg"] + df["temp_min_avg"]) / 2
    df["humidity_avg"] = (df["humidity_max_avg"] + df["humidity_min_avg"]) / 2
    return df


def build_pdf_report(greenhouse_name: str, daily_df: pd.DataFrame, report_title: str = "Reporte climático") -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"{report_title} - {greenhouse_name}", styles["Title"]))
    story.append(Paragraph(f"Generado: {now_gmt6().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    chart_df = daily_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])

    fig1, ax1 = plt.subplots(figsize=(8, 3.2))
    x_smooth, y_smooth = smooth_temperature_series(chart_df["date"], chart_df["temp_avg"])
    ax1.plot(mdates.num2date(x_smooth), y_smooth, color="#F15854", linewidth=2.2)
    ax1.scatter(chart_df["date"], chart_df["temp_avg"], color="#F15854", s=14, zorder=3)
    ax1.set_ylabel("Temperatura promedio (°C)")
    ax1.set_xlabel("Fecha")

    ax2 = ax1.twinx()
    ax2.bar(chart_df["date"], chart_df["humidity_avg"], color="#5DA5DA", alpha=0.65)
    ax2.set_ylabel("Humedad relativa promedio (%)")
    fig1.autofmt_xdate()
    fig1.tight_layout()

    chart_1_buffer = BytesIO()
    fig1.savefig(chart_1_buffer, format="png", dpi=180)
    chart_1_buffer.seek(0)
    plt.close(fig1)

    story.append(Paragraph("Climograma diario", styles["Heading2"]))
    story.append(Image(chart_1_buffer, width=17 * cm, height=6.5 * cm))
    story.append(Spacer(1, 0.4 * cm))

    fig2, ax3 = plt.subplots(figsize=(8, 2.8))
    ax3.bar(chart_df["date"], chart_df["co2_avg"], color="#60BD68")
    ax3.set_ylabel("CO₂ promedio (ppm)")
    ax3.set_xlabel("Fecha")
    fig2.autofmt_xdate()
    fig2.tight_layout()

    chart_2_buffer = BytesIO()
    fig2.savefig(chart_2_buffer, format="png", dpi=180)
    chart_2_buffer.seek(0)
    plt.close(fig2)

    story.append(Paragraph("CO₂ promedio diario", styles["Heading2"]))
    story.append(Image(chart_2_buffer, width=17 * cm, height=5.8 * cm))
    story.append(Spacer(1, 0.5 * cm))

    table_df = chart_df.copy()
    table_df["date"] = table_df["date"].dt.strftime("%Y-%m-%d")
    table_df = table_df[["date", "temp_avg", "humidity_avg", "co2_avg", "readings_count"]].round(2)
    table_df.columns = ["Fecha", "Temp. prom (°C)", "HR prom (%)", "CO₂ prom (ppm)", "Lecturas"]

    table = Table([table_df.columns.tolist()] + table_df.values.tolist(), repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F4B7C")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )

    story.append(Paragraph("Resumen diario", styles["Heading2"]))
    story.append(table)

    doc.build(story)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


def validate_extremes(min_value: float, max_value: float, metric_name: str) -> None:
    if min_value > max_value:
        st.error(f"⚠️ {metric_name}: el valor mínimo no puede ser mayor al máximo.")
        st.stop()


def render_auth() -> bool:
    st.sidebar.markdown("## 🔐 Acceso")

    if st.session_state.get("authenticated"):
        st.sidebar.success(f"Sesión activa: {st.session_state.get('username')}")
        if st.sidebar.button("Cerrar sesión"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        return True

    tab_login, tab_register = st.sidebar.tabs(["Iniciar sesión", "Crear usuario"])

    with tab_login:
        username = st.text_input("Usuario", key="login_user")
        password = st.text_input("Contraseña", type="password", key="login_pass")
        if st.button("Entrar"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username.strip().lower()
                st.rerun()
            st.error("Credenciales inválidas")

    with tab_register:
        new_user = st.text_input("Nuevo usuario", key="new_user")
        pass_1 = st.text_input("Nueva contraseña", type="password", key="new_pass_1")
        pass_2 = st.text_input("Repite contraseña", type="password", key="new_pass_2")
        if st.button("Crear cuenta"):
            if not new_user.strip() or len(pass_1) < 6:
                st.warning("Usuario válido y contraseña mínima de 6 caracteres.")
            elif pass_1 != pass_2:
                st.warning("Las contraseñas no coinciden.")
            else:
                try:
                    create_user(new_user, pass_1)
                    st.success("Usuario creado. Ahora inicia sesión.")
                except sqlite3.IntegrityError:
                    st.warning("Ese usuario ya existe.")

    return False


def apply_date_filter(df: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    return df[(df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)].copy()


def main() -> None:
    st.set_page_config(page_title="Registro ambiental de hongos", page_icon="🍄", layout="wide")
    inject_watermark_logo()
    init_db()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    st.title("🍄 Registro ambiental para producción de hongos")
    st.caption("Registra lecturas por invernadero, calcula promedios diarios automáticamente y genera reportes.")
    st.caption(f"Zona horaria operativa: GMT-6 | Hora actual: {now_gmt6().strftime('%Y-%m-%d %H:%M:%S')}")

    if not render_auth():
        st.info("Inicia sesión o crea un usuario para usar la aplicación.")
        return

    backup_hours = int(get_setting("backup_interval_hours", "24"))
    with st.sidebar.expander("💾 Copias de seguridad programadas"):
        new_interval = st.number_input("Intervalo (horas)", min_value=1, max_value=168, value=backup_hours)
        if st.button("Guardar configuración de respaldo"):
            set_setting("backup_interval_hours", str(int(new_interval)))
            st.success("Intervalo actualizado")
            backup_hours = int(new_interval)

        if st.button("Ejecutar respaldo ahora"):
            created, stamp = run_scheduled_backup(interval_hours=0)
            if created:
                st.success(f"Respaldo creado: {stamp}")
            else:
                st.info(stamp)

        last_backup = get_setting("last_backup_at", "No disponible")
        st.caption(f"Último respaldo: {last_backup}")

    run_scheduled_backup(interval_hours=backup_hours)

    greenhouses = get_greenhouses()
    st.sidebar.header("Invernaderos")

    with st.sidebar.expander("➕ Añadir invernadero", expanded=greenhouses.empty):
        with st.form("add_greenhouse_form", clear_on_submit=True):
            new_name = st.text_input("Nombre del invernadero")
            if st.form_submit_button("Guardar"):
                if not new_name.strip():
                    st.warning("Escribe un nombre válido.")
                else:
                    try:
                        create_greenhouse(new_name)
                        st.success("Invernadero agregado.")
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.warning("Ya existe un invernadero con ese nombre.")

    if greenhouses.empty:
        st.info("Primero añade al menos un invernadero para comenzar a registrar datos.")
        return

    selected_name = st.sidebar.selectbox("Selecciona un invernadero", greenhouses["name"].tolist())
    selected_row = greenhouses[greenhouses["name"] == selected_name].iloc[0]
    greenhouse_id = int(selected_row["id"])

    with st.sidebar.expander("✏️ Editar / eliminar invernadero"):
        rename_value = st.text_input("Nuevo nombre", value=selected_name)
        c1, c2 = st.columns(2)
        if c1.button("Renombrar"):
            try:
                update_greenhouse(greenhouse_id, rename_value)
                st.success("Nombre actualizado")
                st.rerun()
            except sqlite3.IntegrityError:
                st.warning("Ese nombre ya está en uso")
        if c2.button("Eliminar"):
            delete_greenhouse(greenhouse_id)
            st.success("Invernadero eliminado")
            st.rerun()

    tab_registro, tab_historial, tab_graficas = st.tabs(["📝 Registrar lectura", "🧾 Historial y edición", "📊 Gráficas y reportes"])

    with tab_registro:
        st.subheader(f"Nueva lectura en: {selected_name}")
        with st.form("new_reading_form", clear_on_submit=True):
            col_date, col_time = st.columns(2)
            current_local_dt = now_gmt6()
            record_date = col_date.date_input("Fecha", value=current_local_dt.date())
            record_time = col_time.time_input("Hora (GMT-6)", value=current_local_dt.time().replace(microsecond=0))

            c1, c2, c3 = st.columns(3)
            temp_max = c1.number_input("Temperatura máxima (°C)", value=25.0, step=0.1, format="%.1f")
            temp_min = c2.number_input("Temperatura mínima (°C)", value=18.0, step=0.1, format="%.1f")
            co2 = c3.number_input("CO₂ (ppm)", value=800.0, min_value=0.0, step=0.1, format="%.1f")

            c4, c5 = st.columns(2)
            humidity_max = c4.number_input("Humedad relativa máxima (%)", value=90.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
            humidity_min = c5.number_input("Humedad relativa mínima (%)", value=75.0, min_value=0.0, max_value=100.0, step=0.1, format="%.1f")

            if st.form_submit_button("Guardar lectura"):
                validate_extremes(temp_min, temp_max, "Temperatura")
                validate_extremes(humidity_min, humidity_max, "Humedad relativa")
                add_reading(greenhouse_id, datetime.combine(record_date, record_time), float(temp_max), float(temp_min), float(humidity_max), float(humidity_min), float(co2))
                st.success("Lectura guardada correctamente.")
                st.rerun()

    readings_df = get_readings(greenhouse_id)

    with tab_historial:
        st.subheader("Historial de lecturas")
        if readings_df.empty:
            st.info("Aún no hay lecturas para este invernadero.")
        else:
            if "editing_reading_id" not in st.session_state:
                st.session_state.editing_reading_id = None

            headers = st.columns([1.8, 1, 1, 1, 1, 1, 1.5])
            for i, label in enumerate(["Fecha y hora", "T máx", "T mín", "HR máx", "HR mín", "CO₂", "Acciones"]):
                headers[i].markdown(f"**{label}**")

            for row in readings_df.itertuples(index=False):
                cols = st.columns([1.8, 1, 1, 1, 1, 1, 1.5])
                cols[0].write(row.recorded_at.strftime("%Y-%m-%d %H:%M"))
                cols[1].write(f"{row.temp_max:.1f}")
                cols[2].write(f"{row.temp_min:.1f}")
                cols[3].write(f"{row.humidity_max:.1f}")
                cols[4].write(f"{row.humidity_min:.1f}")
                cols[5].write(f"{row.co2:.1f}")
                ac = cols[6].columns(2)
                if ac[0].button("✏️", key=f"edit_{row.id}"):
                    st.session_state.editing_reading_id = int(row.id)
                if ac[1].button("🗑️", key=f"del_{row.id}"):
                    delete_reading(int(row.id))
                    st.success("Lectura eliminada.")
                    st.rerun()

            selected_reading_id = st.session_state.editing_reading_id
            if selected_reading_id and selected_reading_id in readings_df["id"].tolist():
                selected = readings_df[readings_df["id"] == selected_reading_id].iloc[0]
                st.markdown("### Editar lectura seleccionada")
                with st.form("edit_reading_form"):
                    e1, e2 = st.columns(2)
                    edit_date = e1.date_input("Fecha", value=selected["recorded_at"].date())
                    edit_time = e2.time_input("Hora", value=selected["recorded_at"].time().replace(microsecond=0))

                    e3, e4, e5 = st.columns(3)
                    edit_temp_max = e3.number_input("Temp. máx (°C)", value=float(selected["temp_max"]), step=0.1, format="%.1f")
                    edit_temp_min = e4.number_input("Temp. mín (°C)", value=float(selected["temp_min"]), step=0.1, format="%.1f")
                    edit_co2 = e5.number_input("CO₂ (ppm)", value=float(selected["co2"]), min_value=0.0, step=0.1, format="%.1f")

                    e6, e7 = st.columns(2)
                    edit_hmax = e6.number_input("HR máx (%)", value=float(selected["humidity_max"]), min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
                    edit_hmin = e7.number_input("HR mín (%)", value=float(selected["humidity_min"]), min_value=0.0, max_value=100.0, step=0.1, format="%.1f")

                    b1, b2 = st.columns(2)
                    if b1.form_submit_button("Guardar cambios"):
                        validate_extremes(edit_temp_min, edit_temp_max, "Temperatura")
                        validate_extremes(edit_hmin, edit_hmax, "Humedad relativa")
                        update_reading(int(selected_reading_id), datetime.combine(edit_date, edit_time), float(edit_temp_max), float(edit_temp_min), float(edit_hmax), float(edit_hmin), float(edit_co2))
                        st.session_state.editing_reading_id = None
                        st.success("Lectura actualizada")
                        st.rerun()
                    if b2.form_submit_button("Cancelar"):
                        st.session_state.editing_reading_id = None
                        st.rerun()

    with tab_graficas:
        st.subheader("Climograma y CO₂ promedio diario")
        daily_df = get_daily_summary(greenhouse_id)

        if daily_df.empty:
            st.info("Agrega lecturas para visualizar las gráficas y generar reportes.")
            return

        min_date = daily_df["date"].min().date()
        max_date = daily_df["date"].max().date()
        fr1, fr2 = st.columns(2)
        start_date = fr1.date_input("Desde", value=min_date, min_value=min_date, max_value=max_date)
        end_date = fr2.date_input("Hasta", value=max_date, min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.warning("El rango de fechas no es válido")
            return

        filtered_df = apply_date_filter(daily_df, start_date, end_date)
        if filtered_df.empty:
            st.warning("No hay datos en el rango seleccionado")
            return

        chart_df = filtered_df.copy()
        chart_df["date_label"] = chart_df["date"].dt.strftime("%Y-%m-%d")
        sort_order = chart_df["date_label"].tolist()

        temp_chart = (
            alt.Chart(chart_df)
            .mark_line(color="#FF8A65", point=True, strokeWidth=3, interpolate="monotone")
            .encode(
                x=alt.X("date_label:N", title="Fecha", sort=sort_order),
                y=alt.Y("temp_avg:Q", title="Temperatura promedio (°C)"),
                tooltip=["date_label:N", "humidity_avg:Q", "temp_avg:Q", "readings_count:Q"],
            )
        )

        hum_chart = (
            alt.Chart(chart_df)
            .mark_bar(color="#4DD0E1", opacity=0.7)
            .encode(
                x=alt.X("date_label:N", title="Fecha", sort=sort_order),
                y=alt.Y("humidity_avg:Q", title="Humedad relativa promedio (%)", axis=alt.Axis(orient="right")),
                tooltip=["date_label:N", "humidity_avg:Q", "temp_avg:Q", "readings_count:Q"],
            )
        )

        st.altair_chart(alt.layer(hum_chart, temp_chart).resolve_scale(y="independent").properties(height=360), use_container_width=True)

        co2_chart = (
            alt.Chart(chart_df)
            .mark_bar(color="#81C784")
            .encode(
                x=alt.X("date_label:N", title="Fecha", sort=sort_order),
                y=alt.Y("co2_avg:Q", title="CO₂ promedio diario (ppm)"),
                tooltip=["date_label:N", "co2_avg:Q", "readings_count:Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(co2_chart, use_container_width=True)

        export_df = chart_df.copy()
        export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")

        st.markdown("### Descargas")
        st.download_button(
            label="⬇️ Descargar promedios diarios (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"promedios_{selected_name.replace(' ', '_')}_{start_date}_{end_date}.csv",
            mime="text/csv",
        )

        pdf_bytes = build_pdf_report(selected_name, chart_df, report_title=f"Reporte por rango {start_date} a {end_date}")
        st.download_button(
            label="⬇️ Descargar reporte de rango (PDF)",
            data=pdf_bytes,
            file_name=f"reporte_rango_{selected_name.replace(' ', '_')}_{start_date}_{end_date}.pdf",
            mime="application/pdf",
        )

        st.markdown("### Reporte anual automático")
        year_options = sorted(daily_df["date"].dt.year.unique().tolist())
        selected_year = st.selectbox("Selecciona año", year_options, index=len(year_options) - 1)
        annual_df = daily_df[daily_df["date"].dt.year == selected_year].copy()
        if not annual_df.empty:
            annual_pdf = build_pdf_report(selected_name, annual_df, report_title=f"Reporte anual {selected_year}")
            st.download_button(
                label=f"⬇️ Descargar reporte anual {selected_year} (PDF)",
                data=annual_pdf,
                file_name=f"reporte_anual_{selected_name.replace(' ', '_')}_{selected_year}.pdf",
                mime="application/pdf",
            )

        st.dataframe(
            export_df[["date", "temp_max_avg", "temp_min_avg", "temp_avg", "humidity_max_avg", "humidity_min_avg", "humidity_avg", "co2_avg", "readings_count"]].round(2),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
