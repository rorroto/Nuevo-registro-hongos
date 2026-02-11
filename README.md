# Nuevo-registro-hongos

Aplicación en **Streamlit** para registrar parámetros ambientales en producción de hongos comestibles, con persistencia de datos, edición/borrado de registros y generación automática de gráficas y reportes.

## ¿Qué hace esta app?

- Gestión de múltiples invernaderos:
  - Crear invernaderos
  - Renombrarlos
  - Eliminarlos
- Registro de lecturas ambientales (una o varias por día):
  - Temperatura máxima y mínima
  - Humedad relativa máxima y mínima
  - CO₂
  - Fecha y hora exacta de cada lectura
- Historial editable:
  - Editar lecturas existentes
  - Borrar lecturas
- Análisis automático diario:
  - Cálculo de promedios diarios aunque haya 0, 1 o varias lecturas por día
  - Climograma combinado (temperatura promedio + humedad relativa promedio)
  - Gráfica de barras de CO₂ promedio diario
- Exportables:
  - CSV con promedios diarios
  - PDF con gráficas y resumen tabular
- Persistencia en el tiempo:
  - Base de datos SQLite local (`data/registro_hongos.db`)
- Experiencia visual:
  - Tema oscuro por defecto para mejor uso en campo y baja iluminación

## Instalación local

1. Clona el repositorio.
2. Instala dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecuta la app:

```bash
streamlit run app.py
```

## Despliegue recomendado en Streamlit Community Cloud

1. Sube este repositorio a GitHub.
2. Entra a [https://share.streamlit.io](https://share.streamlit.io).
3. Crea una nueva app seleccionando este repo y `app.py` como archivo principal.
4. Al abrir la app desde tu teléfono, podrás registrar lecturas en campo.

> Nota: SQLite funciona bien para uso personal o equipos pequeños. Si en el futuro crece el número de usuarios simultáneos, conviene migrar a una base de datos remota (por ejemplo PostgreSQL/Supabase).

## Estructura

- `app.py`: app principal Streamlit + lógica de base de datos + reportes PDF.
- `requirements.txt`: dependencias Python.
- `.streamlit/config.toml`: configuración global de Streamlit (tema oscuro).
- `data/registro_hongos.db`: se crea automáticamente al ejecutar la app.

## Siguiente mejora sugerida

- Añadir autenticación de usuarios y roles.
- Filtros por rango de fechas y reporte anual automático.
- Copias de seguridad programadas de la base de datos.
