# Nuevo-registro-hongos

App en Streamlit para registro ambiental de invernaderos de hongos comestibles, con persistencia histórica, gráficos y reportes.

## Funciones principales

- Autenticación de usuarios (registro + inicio de sesión).
- Gestión de invernaderos (crear, editar nombre, eliminar).
- Registro de lecturas múltiples por día:
  - Temperatura máxima y mínima.
  - Humedad relativa máxima y mínima.
  - CO₂.
- Historial editable con acciones por fila (editar / eliminar).
- Gráficas automáticas:
  - Climograma (temperatura y humedad promedio diaria).
  - Barras de CO₂ promedio diario.
- Filtros por rango de fechas para análisis y exportación.
- Reporte anual automático (PDF por año seleccionado).
- Exportación de datos:
  - CSV de promedios diarios filtrados.
  - PDF de rango de fechas.
  - PDF anual.
- Copias de seguridad programadas de la base de datos cada 30 días (con opción de ejecución manual).
- Tema oscuro y logo atenuado opcional como marca de agua.

## Instalación local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue en Streamlit Cloud

1. Sube este repositorio a GitHub.
2. Entra a <https://share.streamlit.io>.
3. Crea la app seleccionando `app.py`.

## Estructura

- `app.py`: app principal.
- `requirements.txt`: dependencias.
- `.streamlit/config.toml`: tema global.
- `assets/logo_bio_funga.png`: logo opcional para fondo atenuado.
- `data/registro_hongos.db`: base SQLite (se crea automáticamente).
- `backups/`: respaldos automáticos de la base de datos.
