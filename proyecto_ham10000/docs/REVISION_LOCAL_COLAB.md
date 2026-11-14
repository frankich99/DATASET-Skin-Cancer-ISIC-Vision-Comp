# Revisión de integración local / Colab

**Estudio:** Evaluación Comparativa de Tres Arquitecturas CNN para Clasificación de Lesiones Cutáneas Pigmentadas mediante Dermatoscopia (HAM10000).

## Problemas corregidos

| Hallazgo | Corrección |
|---|---|
| `from google.colab import drive` fallaba en VS Code/Jupyter local | Importación y montaje exclusivos de la rama de Colab alojado; descubrimiento de proyecto/datos local y diagnóstico del intérprete. |
| Título viejo y “resultados pendientes” pese al experimento local terminado | Título formal y estado real comunes al cuaderno, README, informe/artículo, metadatos PDF, presentación y documentación. Se conservan nombres originales del dataset y referencias. |
| Cambio automático de workers 4→2 impedía reanudar checkpoints locales | Se conserva el JSON completo; no se crea una configuración Colab divergente. Para otro protocolo se exige un JSON y raíz explícitos. |
| Volver a abrir un run terminado reescribía su metadata | Reanudación terminada sin modificar artefactos; carpetas no vacías no se reemplazan como si fueran runs nuevos. |
| Cualquier carpeta existente se omitía como evaluación “hecha” | Se verifican archivos, checkpoint/split, muestras, etiquetas y probabilidades. Una carpeta incompleta/incompatible detiene el flujo. |
| Una interrupción podía dejar solo parte de una evaluación final | CSV/JSON se construyen en una carpeta temporal propia y se publica el destino al completar ambos. No se borra ni sobrescribe un resultado existente. |
| Un checkpoint CUDA no podía evaluarse explícitamente en CPU | `--device cpu --num-workers 0`, sin modificar la configuración guardada; dispositivo de inferencia registrado por separado. |
| Mediciones alternativas podían duplicarse o mezclar CPU/GPU al comparar | Selección `--subcarpeta-evaluacion`, rechazo de duplicados/entornos mezclados y etiqueta CPU correcta aunque el equipo tenga GPU disponible. |
| Requisitos CUDA fuera del paquete portable | Requisitos CUDA, comunes y opcionales de notebook dentro de `python/`. Colab conserva su PyTorch compatible. |
| Cuaderno limitado a una CNN y entrenamiento/evaluación poco explícitos | Controles para las tres arquitecturas, validación/test protegidos, 7 escenarios test por modelo, métricas y eficiencia, ROC/figuras, discusión y exportación de evidencias. |
| Referencias documentales a ZIP de 28 MiB y validación solo estática | Contenido portable descrito con precisión; tamaño informado al empaquetar. Documentación distingue ejecución local, simulación y nube real pendiente. |

## Validación efectuada

- **22 pruebas del proyecto**: 10 de pipeline, 2 de integración y 10 de runtime/artefactos; más **4 pruebas del descargador**.
- Integración con imágenes **sintéticas**, incluida interrupción tras la primera época y continuación real de la segunda; checkpoint CUDA evaluado en CPU sin editarlo; fallo de escritura sin publicar una evaluación incompleta.
- Extracción segura, rutas maliciosas/enlaces/colisiones, recuperación de una extracción completa, rechazo de destinos no acreditados y limpieza exclusiva del directorio temporal creado por la prueba.
- Detección y ejecución de la rama Colab **simuladas**, sin credenciales ni autenticación real.
- Esquema oficial `nbformat`, sintaxis de todas las celdas y controles seguros del notebook distribuido, sin salidas incrustadas.
- Ejecución local completa del cuaderno en un kernel del mismo intérprete `.venv`: **11 celdas de código**, verificación de **10 015 hashes de imágenes** y comprobación de **61 artefactos experimentales sin cambios**. Entrenamiento, inferencia nueva, instalación, figuras y exportación desactivados.
- `pip check` sin conflictos. Informe/artículo y presentación compilados; logs revisados para advertencias y desbordes. ZIP portable generado después de compilar y comprobado con CRC.

Los entrenamientos HAM10000, checkpoints, split y métricas publicados se conservaron. Las pruebas sintéticas no agregan resultados al estudio.

## Límites que siguen vigentes

- **No se acredita una ejecución real en Google Colab.** La simulación no prueba Drive, permisos, cuota, GPU asignada ni supervivencia de una sesión externa.
- El ZIP no contiene originales ni checkpoints. Incluye el paquete común completo y el CSV local publicado como referencia; entrenar/evaluar de nuevo exige los datos y, para inferencia, los checkpoints correspondientes.
- La continuidad entre GPU/versions no implica identidad bit a bit. Comparar tiempos/VRAM de entrenamiento exige revisar también las sesiones, no solo el dispositivo de evaluación.
- El modo de inspección muestra resultados existentes, incluidos los de test, pero no ejecuta evaluaciones nuevas. Quien diseñe un estudio nuevo no debe usar esa referencia test para ajustar su protocolo.
- La exportación opcional acredita procedencia de archivos, no que se hayan producido en la sesión actual. Un experimento nuevo no actualiza automáticamente cifras ni textos científicos.
- El guion de diez minutos sigue requiriendo ensayo cronometrado; su duración no se presenta como medida. No se incluye despliegue ni validación clínica.

## Repetir la comprobación

Desde la raíz del proyecto, después de instalar los requisitos opcionales del notebook:

```powershell
..\.venv\Scripts\python.exe colab\generar_notebook.py
..\.venv\Scripts\python.exe colab\verificar_notebook.py --ejecutar-local --salida <carpeta_externa_de_verificacion>
```

La segunda orden guarda el cuaderno ejecutado y un JSON de verificación fuera del proyecto, sin sustituir el notebook limpio. Ver instrucciones completas en `colab/README_COLAB.md`.
