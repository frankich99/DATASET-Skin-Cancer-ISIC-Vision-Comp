# Evaluación Comparativa de Tres Arquitecturas CNN para Clasificación de Lesiones Cutáneas Pigmentadas mediante Dermatoscopia (HAM10000)

## Cuaderno integrado local y Google Colab

`HAM10000_Grupo09.ipynb` utiliza **todo el paquete `python/ham10000`**, su CLI, configuración y split fijo. No mantiene una segunda implementación de entrenamiento. Integra controles, utilidades de inicialización, diagnóstico, inventario del código, verificación de datos, entrenamiento/reanudación, evaluación, comparación, figuras, discusión y exportación de evidencias.

Los tres entrenamientos y 24 evaluaciones publicados se completaron en la GTX 1650 local. El cuaderno se verifica con pruebas automáticas y ejecución local en modo de inspección; **esto no acredita una ejecución en una cuenta real de Colab**, todavía pendiente. No incluye despliegue ni constituye diagnóstico clínico.

## Ejecutar localmente en VS Code/Jupyter

1. Abra el cuaderno dentro de `proyecto_ham10000/colab/` y seleccione el kernel de `..\.venv\Scripts\python.exe` (ruta relativa a la raíz del proyecto). La primera celda de inicialización imprime el intérprete exacto.
2. Si necesita preparar otro entorno, instale por separado `python/requirements-cuda.txt`, `python/requirements-common.txt` y, para Jupyter local, `python/requirements-notebook.txt`. El primero reproduce las versiones CUDA del experimento; no sirve para cualquier GPU/controlador. No copie la `.venv` de Windows a Linux.
3. El descubrimiento automático busca la raíz del proyecto y `../descargaDataSet/HAM10000`. Si usa otra ubicación, indique `RUTA_PROYECTO` y `RUTA_DATOS` en los controles. La raíz de datos debe contener `HAM10000_metadata.csv` y las rutas originales del split.
4. Ejecute las celdas en orden. **No se importa `google.colab`, no se monta Drive ni se exige GPU para inspeccionar.** Instalar `google.colab` no es la solución al error de la captura.

## Ejecutar en Google Colab alojado

Desde la raíz del proyecto, después de actualizar/compilar los entregables:

```powershell
..\.venv\Scripts\python.exe colab\empaquetar.py
```

1. Cree `Mi unidad/Grupo09HAM/` en Google Drive y suba manualmente `proyecto_ham10000_colab.zip` y el ZIP limpio `HAM10000.zip` (aproximadamente 2.9 GB decimales).
2. Abra https://colab.research.google.com/ y suba `HAM10000_Grupo09.ipynb`.
3. Active **Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU** si va a entrenar. La disponibilidad, GPU y duración gratuitas no están garantizadas.
4. Ejecute en orden y autorice el montaje de Drive. Se detecta el backend alojado mediante variables de entorno, no por la mera existencia del paquete `google.colab`. Un runtime local conectado a la interfaz Colab se trata como local.

Los ZIP se copian y extraen de forma segura en `/content/ham10000_sesion/`; los originales se leen desde ese disco, no mediante miles de lecturas desde Drive. Se comprueban nombres, rutas, enlaces, espacio, SHA de la copia y CRC durante la extracción. Un destino no acreditado se rechaza sin sobrescribir. Una extracción completa puede reutilizarse si conserva la huella del mismo ZIP; los hashes del dataset se verifican aparte.

## Contenido portable y resultados de referencia

- Incluye paquete Python completo, dependencias, configuración, manifiestos, notebook, fuentes LaTeX, tablas/figuras/PDF compilados, presentación y `resultados/comparacion.csv` **del experimento local publicado**.
- Excluye imágenes originales y exportadas, `.venv`, checkpoints, `runs/`, resultados pesados y salidas nuevas `resultados_notebook/`.
- La tabla de referencia permite inspeccionar métricas sin entrenar; **no prueba una ejecución nueva**. No permite reconstruir predicciones/ROC nuevas sin los runs/checkpoints originales.
- No se solicitan claves de Kaggle ni se suben imágenes automáticamente. Use archivos de confianza y respete su licencia.

## Controles y seguridad del protocolo

| Control inicial | Efecto |
|---|---|
| `INSTALAR_COMUNES=False` | No instala ni modifica paquetes; al activarlo solo instala dependencias comunes, nunca PyTorch. |
| `VERIFICAR_HASHES=True` | Comprueba SHA-256 de los 10 015 originales y conserva el split canónico. |
| `ENTRENAR=False` | No inicia ni reanuda entrenamientos. |
| `EVALUAR=False` | No ejecuta inferencia ni mediciones nuevas. |
| `CONFIRMAR_TEST=False` | Si se activa evaluación, usa solo `val_clean`. |
| `GENERAR_FIGURAS=False` | No regenera ni sustituye gráficos. |
| `EXPORTAR_EVIDENCIAS=False` | No crea archivos de exportación. |

Para entrenar, active `ENTRENAR` y seleccione `MODELOS_ENTRENAR`: `resnet18`, `efficientnet_b0`, `mobilenet_v3_small`. La configuración base mantiene 224 px, batch 8, FP32, semilla 42 y **num_workers=4 también en Colab**. No se genera un JSON diferente con workers 2. Para otro protocolo, indique otro JSON y otra `RUTA_RUNS` explícitamente; no mezcle resultados.

La reanudación exige coincidencia exacta de configuración, arquitectura y SHA del split; usa `last.pt`. Las épocas incompletas se repiten. Una corrida terminada no se reescribe. La misma semilla no garantiza igualdad bit a bit entre GPU/versions. En Colab los checkpoints, historiales y métricas se escriben en Drive (`Grupo09HAM/runs`) y la caché de pesos en `torch_cache`. `/content` desaparece al finalizar la sesión; reconstruirlo y reanudar requiere mismo split/JSON. No se garantiza persistencia ante fallos de almacenamiento o montaje.

## Evaluación, métricas y figuras

Active `EVALUAR=True`: por defecto evalúa clean en validación de los modelos seleccionados. Solo tras congelar decisiones, active además `CONFIRMAR_TEST=True`: ejecuta 5 condiciones base (clean, noise_002, noise_005, blur_1, blur_2) y 2 mitigaciones (median_3, gaussian_1 sobre noise_005) por modelo, total **21 evaluaciones test**. No reajuste con test.

Antes de reutilizar una carpeta existente se comprueban `metrics.json`, `predictions.csv`, procedencia del checkpoint/split, todas las muestras y probabilidades. Una carpeta incompleta detiene el flujo, no se omite ni sobrescribe. Las evaluaciones nuevas se publican desde un directorio temporal al terminar ambos archivos.

Puede evaluar un checkpoint entrenado en CUDA usando `DISPOSITIVO_EVAL="cpu"`, `WORKERS_EVAL=0` y `SUBCARPETA_EVAL="evaluacion_cpu"`, sin editarlo. Para CLI: `evaluar --device cpu --num-workers 0 --salida <otra_salida>`. Las latencias CPU no sustituyen las publicadas de GPU. `comparar --subcarpeta-evaluacion evaluacion` selecciona solo esa carpeta; se rechazan duplicados y entornos de evaluación mezclados.

Se muestran Macro Precision, F1, Balanced Accuracy, sensibilidad de melanoma, AUC, robustez/mitigación, tiempos/VRAM/parámetros y, si hay runs, reportes por clase/matrices. `GENERAR_FIGURAS=True` invoca la implementación común para producir ROC, matrices, curvas y comparación, con color y 300 DPI, en **`resultados_notebook/`**, sin sustituir `resultados/` ni figuras LaTeX. La comparación principal requiere un mismo entorno de evaluación. Verificar también las sesiones de entrenamiento al interpretar tiempo/VRAM: una reanudación en otra GPU rompe esa comparabilidad.

## Exportación, documentos y mantenimiento

`EXPORTAR_EVIDENCIAS=True` crea un ZIP con CSV/JSON, configuración/split y figuras nuevas más un manifiesto de fuentes, SHA-256 y entorno. Excluye imágenes y checkpoints; conserva los runs por separado. Exportar archivos existentes no acredita una ejecución en la sesión actual.

Informe: `latex/main.tex`; artículo: `latex/paper_ieee.tex`; exposición: `presentacion/presentacion_grupo09.tex` y `guion_exposicion_10min.md`. Las cifras publicadas siguen siendo las locales. Un estudio nuevo requiere revisar las 3 validaciones + 21 test y actualizar deliberadamente tablas/figuras/textos, sin afirmar resultados de nube inexistentes. `latex/compilar.py` requiere MiKTeX/TeX Live; la presentación se compila con dos pasadas de `pdflatex`. Empaquete **después** de compilar.

El notebook se mantiene con `colab/generar_notebook.py` (integra las utilidades de `runtime.py`). Tras cambiarlo, regenere y verifique:

```powershell
..\.venv\Scripts\python.exe colab\generar_notebook.py
..\.venv\Scripts\python.exe colab\verificar_notebook.py
..\.venv\Scripts\python.exe colab\verificar_notebook.py --ejecutar-local
```

La última orden requiere `requirements-notebook.txt`, valida el esquema nbformat y ejecuta un kernel del mismo intérprete, con entrenamiento/evaluación/figuras/exportación desactivados. Su salida ejecutada y reporte se guardan fuera del proyecto si se indica `--salida`; nunca reemplazan el cuaderno distribuido limpio.

Referencias: [FAQ Colab](https://research.google.com/colaboratory/faq.html), [checkpoints PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [HAM10000](https://doi.org/10.1038/sdata.2018.161).
