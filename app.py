from __future__ import annotations

import sqlite3
from datetime import datetime
from io import BytesIO
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

DB_PATH = Path("data") / "registro_hongos.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
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


def get_greenhouses() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            "SELECT id, name, created_at FROM greenhouses ORDER BY name ASC", conn
        )


def create_greenhouse(name: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO greenhouses (name, created_at) VALUES (?, ?)",
            (name.strip(), datetime.now().isoformat()),
        )


def update_greenhouse(greenhouse_id: int, new_name: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE greenhouses SET name = ? WHERE id = ?",
            (new_name.strip(), greenhouse_id),
        )


def delete_greenhouse(greenhouse_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM readings WHERE greenhouse_id = ?", (greenhouse_id,))
        conn.execute("DELETE FROM greenhouses WHERE id = ?", (greenhouse_id,))


def add_reading(
    greenhouse_id: int,
    recorded_at: datetime,
    temp_max: float,
    temp_min: float,
    humidity_max: float,
    humidity_min: float,
    co2: float,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO readings (
                greenhouse_id, recorded_at, temp_max, temp_min,
                humidity_max, humidity_min, co2, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                greenhouse_id,
                recorded_at.isoformat(),
                temp_max,
                temp_min,
                humidity_max,
                humidity_min,
                co2,
                datetime.now().isoformat(),
            ),
        )


def get_readings(greenhouse_id: int) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                id,
                greenhouse_id,
                recorded_at,
                temp_max,
                temp_min,
                humidity_max,
                humidity_min,
                co2
            FROM readings
            WHERE greenhouse_id = ?
            ORDER BY recorded_at DESC
            """,
            conn,
            params=(greenhouse_id,),
        )

    if not df.empty:
        df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    return df


def update_reading(
    reading_id: int,
    recorded_at: datetime,
    temp_max: float,
    temp_min: float,
    humidity_max: float,
    humidity_min: float,
    co2: float,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE readings
            SET recorded_at = ?, temp_max = ?, temp_min = ?,
                humidity_max = ?, humidity_min = ?, co2 = ?
            WHERE id = ?
            """,
            (
                recorded_at.isoformat(),
                temp_max,
                temp_min,
                humidity_max,
                humidity_min,
                co2,
                reading_id,
            ),
        )


def delete_reading(reading_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM readings WHERE id = ?", (reading_id,))


def get_daily_summary(greenhouse_id: int) -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query(
            """
            SELECT
                DATE(recorded_at) AS date,
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


def build_pdf_report(greenhouse_name: str, daily_df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Reporte climático - {greenhouse_name}", styles["Title"]))
    story.append(Paragraph(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    chart_df = daily_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"])

    fig1, ax1 = plt.subplots(figsize=(8, 3.2))
    ax1.bar(chart_df["date"], chart_df["humidity_avg"], color="#5DA5DA", alpha=0.65, label="Humedad (%)")
    ax1.set_ylabel("Humedad relativa promedio (%)")
    ax1.set_xlabel("Fecha")

    ax2 = ax1.twinx()
    ax2.plot(chart_df["date"], chart_df["temp_avg"], color="#F15854", marker="o", linewidth=2, label="Temp. (°C)")
    ax2.set_ylabel("Temperatura promedio (°C)")
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
    table_df = table_df[["date", "temp_avg", "humidity_avg", "co2_avg", "readings_count"]]
    table_df.columns = ["Fecha", "Temp. prom (°C)", "HR prom (%)", "CO₂ prom (ppm)", "Lecturas"]
    table_df = table_df.round(2)

    table_data = [table_df.columns.tolist()] + table_df.values.tolist()
    table = Table(table_data, repeatRows=1)
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


def main() -> None:
    st.set_page_config(page_title="Registro ambiental de hongos", page_icon="🍄", layout="wide")
    st.title("🍄 Registro ambiental para producción de hongos")
    st.caption(
        "Registra lecturas por invernadero, calcula promedios diarios automáticamente y genera reportes."
    )

    init_db()
    greenhouses = get_greenhouses()

    st.sidebar.header("Invernaderos")

    with st.sidebar.expander("➕ Añadir invernadero", expanded=greenhouses.empty):
        with st.form("add_greenhouse_form", clear_on_submit=True):
            new_name = st.text_input("Nombre del invernadero")
            create_submit = st.form_submit_button("Guardar")
            if create_submit:
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
        col_rename, col_delete = st.columns(2)
        if col_rename.button("Renombrar"):
            if not rename_value.strip():
                st.warning("El nuevo nombre no puede estar vacío.")
            else:
                try:
                    update_greenhouse(greenhouse_id, rename_value)
                    st.success("Nombre actualizado.")
                    st.rerun()
                except sqlite3.IntegrityError:
                    st.warning("Ese nombre ya está en uso.")
        if col_delete.button("Eliminar", type="secondary"):
            delete_greenhouse(greenhouse_id)
            st.success("Invernadero eliminado.")
            st.rerun()

    tab_registro, tab_historial, tab_graficas = st.tabs(
        ["📝 Registrar lectura", "🧾 Historial y edición", "📊 Gráficas y reportes"]
    )

    with tab_registro:
        st.subheader(f"Nueva lectura en: {selected_name}")
        with st.form("new_reading_form", clear_on_submit=True):
            col_date, col_time = st.columns(2)
            record_date = col_date.date_input("Fecha", value=datetime.today())
            record_time = col_time.time_input("Hora", value=datetime.now().time())

            c1, c2, c3 = st.columns(3)
            temp_max = c1.number_input("Temperatura máxima (°C)", value=25.0)
            temp_min = c2.number_input("Temperatura mínima (°C)", value=18.0)
            co2 = c3.number_input("CO₂ (ppm)", value=800.0, min_value=0.0)

            c4, c5 = st.columns(2)
            humidity_max = c4.number_input("Humedad relativa máxima (%)", value=90.0, min_value=0.0, max_value=100.0)
            humidity_min = c5.number_input("Humedad relativa mínima (%)", value=75.0, min_value=0.0, max_value=100.0)

            submitted = st.form_submit_button("Guardar lectura")
            if submitted:
                validate_extremes(temp_min, temp_max, "Temperatura")
                validate_extremes(humidity_min, humidity_max, "Humedad relativa")
                record_dt = datetime.combine(record_date, record_time)
                add_reading(
                    greenhouse_id,
                    record_dt,
                    float(temp_max),
                    float(temp_min),
                    float(humidity_max),
                    float(humidity_min),
                    float(co2),
                )
                st.success("Lectura guardada correctamente.")
                st.rerun()

    readings_df = get_readings(greenhouse_id)

    with tab_historial:
        st.subheader("Historial de lecturas")
        if readings_df.empty:
            st.info("Aún no hay lecturas para este invernadero.")
        else:
            show_df = readings_df.copy()
            show_df["recorded_at"] = show_df["recorded_at"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(show_df, use_container_width=True, hide_index=True)

            ids = readings_df["id"].tolist()
            selected_reading_id = st.selectbox("Selecciona una lectura para editar o borrar", ids)
            selected_reading = readings_df[readings_df["id"] == selected_reading_id].iloc[0]

            with st.form("edit_reading_form"):
                rd_at = selected_reading["recorded_at"]
                e1, e2 = st.columns(2)
                edit_date = e1.date_input("Fecha", value=rd_at.date(), key="edit_date")
                edit_time = e2.time_input("Hora", value=rd_at.time(), key="edit_time")

                e3, e4, e5 = st.columns(3)
                edit_temp_max = e3.number_input("Temp. máx (°C)", value=float(selected_reading["temp_max"]), key="edit_tmax")
                edit_temp_min = e4.number_input("Temp. mín (°C)", value=float(selected_reading["temp_min"]), key="edit_tmin")
                edit_co2 = e5.number_input("CO₂ (ppm)", value=float(selected_reading["co2"]), min_value=0.0, key="edit_co2")

                e6, e7 = st.columns(2)
                edit_hmax = e6.number_input(
                    "HR máx (%)", value=float(selected_reading["humidity_max"]), min_value=0.0, max_value=100.0, key="edit_hmax"
                )
                edit_hmin = e7.number_input(
                    "HR mín (%)", value=float(selected_reading["humidity_min"]), min_value=0.0, max_value=100.0, key="edit_hmin"
                )

                col_update, col_del = st.columns(2)
                update_submit = col_update.form_submit_button("Guardar cambios")
                delete_submit = col_del.form_submit_button("Borrar lectura")

                if update_submit:
                    validate_extremes(edit_temp_min, edit_temp_max, "Temperatura")
                    validate_extremes(edit_hmin, edit_hmax, "Humedad relativa")
                    update_reading(
                        int(selected_reading_id),
                        datetime.combine(edit_date, edit_time),
                        float(edit_temp_max),
                        float(edit_temp_min),
                        float(edit_hmax),
                        float(edit_hmin),
                        float(edit_co2),
                    )
                    st.success("Lectura actualizada.")
                    st.rerun()

                if delete_submit:
                    delete_reading(int(selected_reading_id))
                    st.success("Lectura eliminada.")
                    st.rerun()

    with tab_graficas:
        st.subheader("Climograma y CO₂ promedio diario")
        daily_df = get_daily_summary(greenhouse_id)

        if daily_df.empty:
            st.info("Agrega lecturas para visualizar las gráficas y generar reportes.")
        else:
            chart_df = daily_df.copy()

            hum_chart = (
                alt.Chart(chart_df)
                .mark_bar(color="#72B7B2", opacity=0.65)
                .encode(
                    x=alt.X("date:T", title="Fecha"),
                    y=alt.Y("humidity_avg:Q", title="Humedad relativa promedio (%)"),
                    tooltip=["date:T", "humidity_avg:Q", "temp_avg:Q", "readings_count:Q"],
                )
            )

            temp_chart = (
                alt.Chart(chart_df)
                .mark_line(color="#E45756", point=True, strokeWidth=3)
                .encode(
                    x="date:T",
                    y=alt.Y("temp_avg:Q", title="Temperatura promedio (°C)"),
                    tooltip=["date:T", "humidity_avg:Q", "temp_avg:Q", "readings_count:Q"],
                )
            )

            st.altair_chart(
                alt.layer(hum_chart, temp_chart).resolve_scale(y="independent").properties(height=360),
                use_container_width=True,
            )

            co2_chart = (
                alt.Chart(chart_df)
                .mark_bar(color="#54A24B")
                .encode(
                    x=alt.X("date:T", title="Fecha"),
                    y=alt.Y("co2_avg:Q", title="CO₂ promedio diario (ppm)"),
                    tooltip=["date:T", "co2_avg:Q", "readings_count:Q"],
                )
                .properties(height=320)
            )
            st.altair_chart(co2_chart, use_container_width=True)

            export_df = chart_df.copy()
            export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")

            st.markdown("### Descargas")
            csv_bytes = export_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar promedios diarios (CSV)",
                data=csv_bytes,
                file_name=f"promedios_diarios_{selected_name.replace(' ', '_')}.csv",
                mime="text/csv",
            )

            pdf_bytes = build_pdf_report(selected_name, chart_df)
            st.download_button(
                label="⬇️ Descargar reporte (PDF)",
                data=pdf_bytes,
                file_name=f"reporte_climatico_{selected_name.replace(' ', '_')}.pdf",
                mime="application/pdf",
            )

            st.dataframe(
                export_df[
                    [
                        "date",
                        "temp_max_avg",
                        "temp_min_avg",
                        "temp_avg",
                        "humidity_max_avg",
                        "humidity_min_avg",
                        "humidity_avg",
                        "co2_avg",
                        "readings_count",
                    ]
                ].round(2),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()
