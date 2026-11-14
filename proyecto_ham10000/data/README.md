# Datos y particiones

Los originales se conservan intactos fuera del proyecto, en `../../descargaDataSet/HAM10000`.

`processed/` contiene la partición oficial verificada (~70% / 15% / 15%, agrupada por `lesion_id` y SHA-256, 0 fugas):

- `processed/train/<clase>/` — 7,009 imágenes (`224x224` RGB sin deformar)
- `processed/val/<clase>/` — 1,503 imágenes (`224x224` RGB sin deformar)
- `processed/test/<clase>/` — 1,503 imágenes (`224x224` RGB sin deformar)
- `processed/split.csv` — manifiesto maestro con rutas relativas al dataset original
- `processed/dataset_procesado.csv` — manifiesto con `ruta_procesada` y `aumentada=False`
- `processed/estadisticas.json` — media/desviación RGB de train, normalización ImageNet y pesos de clase
- `processed/auditoria.json` — resumen y huellas SHA-256 de auditoría

El entrenamiento lee directamente las imágenes originales mediante `split.csv` para evitar doble compresión JPEG, mientras `train/`, `val/` y `test/` proveen el desglose físico listo por clase para inspección o `ImageFolder`.

No editar manualmente los CSV ni volver a sortear splits después de mirar resultados. `auditoria.json` registra las huellas del dataset y del manifiesto.
