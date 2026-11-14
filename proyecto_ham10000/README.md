# Grupo 9 — Evaluación Comparativa de Tres Arquitecturas CNN para Clasificación de Lesiones Cutáneas Pigmentadas mediante Dermatoscopia (HAM10000)

**Universidad Nacional de San Antonio Abad del Cusco — Visión por Computador (2026-II)**  
**Docente:** Mg. Hans Harley Ccacyahuillca Bejar

| Integrante | Código |
|---|---|
| Choquenaira Quispe, Noe Franklin | 133962 |
| Porroa Sivana, Yeni Ruth | 120893 |
| Colque Galindo, Jean Franco | 193109 |

## Objetivo y estado real

Evaluación comparativa de **ResNet-18, EfficientNet-B0 y MobileNetV3-Small** con transfer learning (`IMAGENET1K_V1`) para la clasificación de siete categorías de lesiones cutáneas pigmentadas mediante dermatoscopia (HAM10000). Se evalúan Macro F1, exactitud balanceada, sensibilidad de melanoma (`mel`), AUC OVR, robustez ante degradaciones y mitigación espacial, así como latencia y VRAM en hardware común.

**Estado actual: Completado (Fases 1–9).** Se ejecutaron íntegramente la auditoría real (2 pares SHA-256 en `nv`, verificación de sesión con 0 pares casi-idénticos $\mathrm{RMSE}<4.5$ entre lesiones sin artefacto independiente persistido y 0 fugas en el split `StratifiedGroupKFold(20)`), los tres entrenamientos en la GPU local (**Acer Nitro AN515-45, NVIDIA GTX 1650 4 GiB, driver 621.01, PyTorch 2.11.0+cu128, FP32, batch 8, num_workers 4, seed 42**), las 24 evaluaciones sobre `val` y `test` limpio/perturbado (incluyendo mitigación `median_3` y `gaussian_1` sobre `noise_005`), la consolidación en `resultados/`, la presentación Beamer (`presentacion/presentacion_grupo09.pdf`) y el guion de exposición (`presentacion/guion_exposicion_10min.md`).

El despliegue interactivo (web o móvil) quedó explícitamente excluido del alcance del proyecto por indicación docente.

## Resumen de resultados reales (GTX 1650, seed 42)

### Entrenamiento y eficiencia

| Arquitectura | Parámetros | Épocas (Mejor) | s / época | VRAM pico | Mejor Val Macro F1 |
|---|---:|:---:|---:|---:|---:|
| **ResNet-18** | 11 180 103 (11.18M) | 16 (ép. 11) | **127.5** | 434.2 MiB | 0.6927 |
| **EfficientNet-B0** | 4 016 515 (4.02M) | 20 (ép. 19) | 188.6 | 805.7 MiB | **0.7059** |
| **MobileNetV3-Small** | 1 525 031 (1.53M) | 19 (ép. 14) | 156.3 | **211.8 MiB** | 0.6213 |

### Desempeño en `test` limpio ($n = 1\,503$)

| Arquitectura | Accuracy | Macro Prec. | Macro F1 | Bal. Acc. | Recall `mel` | AUC OVR | Lat. `fwd` (`batch=1`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ResNet-18** | **0.8031** | 0.6591 | **0.6786** | **0.7174** | 0.6012 | **0.9596** | **7.22 ms** *(6.86–9.46)* |
| **EfficientNet-B0** | 0.8004 | **0.6741** | 0.6778 | 0.7025 | **0.6607** | 0.9587 | 24.60 ms *(22.49–34.82)* |
| **MobileNetV3-Small** | 0.7625 | 0.6474 | 0.5700 | 0.5604 | 0.4167 | 0.9407 | 22.40 ms *(15.26–22.40)* |

> *Nota sobre latencia:* mide exclusivamente el paso `forward()` en GPU FP32 con `batch=1` (10 warmup + 50 pasadas sincronizadas), excluyendo lectura E/S de disco, redimensionamiento y filtrado espacial en CPU; entre paréntesis se indica el rango observado en las 7 evaluaciones sobre `test`.

### Robustez y mitigación espacial en `test` (Macro F1)

| Arquitectura | `clean` | `noise_002` | `noise_005` | `blur_1` | `blur_2` | `+ median_3` *(s/ n005)* | `+ gaussian_1` *(s/ n005)* |
|---|---:|---:|---:|---:|---:|---:|---:|
| **ResNet-18** | **0.6786** | **0.4561** | 0.2710 | **0.5769** | **0.4895** | **0.4624** *(Rec. mel 0.6190)* | **0.5104** *(Rec. mel **0.6905**)* |
| **EfficientNet-B0** | 0.6778 | 0.2745 | 0.1201 *(Rec. mel 0.0000)* | 0.5167 | 0.2815 | 0.4383 *(Rec. mel 0.3095)* | 0.4518 *(Rec. mel 0.1250)* |
| **MobileNetV3-Small** | 0.5700 | 0.3580 | **0.3079** | 0.5398 | 0.3949 | 0.3804 *(Rec. mel 0.1964)* | 0.4505 *(Rec. mel 0.2381)* |

> **Intercambio Precisión / Falsos Positivos en melanoma (`mel`, $n=168$):** el prefiltrado `gaussian_1` sobre `noise_005` constituye una **mitigación parcial** (no una restauración plena). En **ResNet-18**, aunque el Recall `mel` sube de `0.6012` en limpio (`0.1250` en `noise_005`) a `0.6905`, la **Precision `mel` desciende de `0.5233` a `0.3648`**, elevando los falsos positivos de **92 a 202** (F1 `mel` `0.4774` frente a `0.5596` en limpio; con `median_3`: Prec. `0.3675`, Rec. `0.6190`, `179` FP).

**Veredicto final:** Sin superioridad ni equivalencia estadística demostrada en Macro F1 frente a EfficientNet-B0 ($\Delta=0.00084$; IC95% bootstrap sobre 1 118 grupos en `test`: `[-0.0429, +0.0464]`), **ResNet-18** ofrece el **mejor balance observado** en este protocolo y GTX 1650 (Macro F1 **0.6786**, Bal. Acc. **0.7174**, AUC OVR **0.9596** y **3.4×** menor latencia `forward()` en limpio con **7.22 ms** frente a **24.60 ms** de EfficientNet-B0 y **22.40 ms** de MobileNetV3-Small). **EfficientNet-B0** lidera en Macro Prec. (**0.6741**) y sensibilidad limpia de melanoma (**0.6607**), pero colapsa ante ruido aditivo (`noise_005`: Macro F1 **0.1201**, Recall `mel` **0.0000**). El filtrado espacial previo `gaussian_1` aporta **+0.14 a +0.33** puntos de Macro F1 sobre `noise_005` (**+0.1426** en MobileNetV3-Small, **+0.2394** en ResNet-18 y **+0.3317** en EfficientNet-B0).

## Estructura

```text
proyecto_ham10000/
├── README.md
├── PLAN_TRABAJO.md
├── configs/base.json
├── python/
│   ├── ejecutar.py                  # CLI común local y Colab
│   ├── requirements-common.txt
│   ├── ham10000/                    # datos, modelos, entrenamiento, métricas y gráficos
│   └── tests/                       # pruebas automáticas
├── colab/
│   ├── HAM10000_Grupo09.ipynb
│   ├── README_COLAB.md
│   └── empaquetar.py
├── data/
│   └── processed/
│       ├── train/                  # 7009 imágenes 224x224 por clase
│       ├── val/                    # 1503 imágenes 224x224 por clase
│       ├── test/                   # 1503 imágenes 224x224 por clase
│       ├── split.csv               # rutas relativas al dataset original
│       ├── dataset_procesado.csv   # manifiesto compatible con carpetas procesadas
│       ├── estadisticas.json       # momentos RGB de train y pesos de clase
│       └── auditoria.json
├── runs/                           # checkpoints, history.csv, run.json y evaluaciones val/test
├── resultados/                     # comparacion.csv, curvas, matrices ROC/confusión y robustez
├── latex/
│   ├── main.tex                    # informe técnico ampliado (>= 25 páginas)
│   ├── paper_ieee.tex              # artículo científico en plantilla IEEEtran
│   ├── referencias.bib
│   ├── secciones/
│   ├── img/                        # escudo y figuras reales
│   ├── tablas/                     # tablas de auditoría generadas
│   └── compilar.py
└── presentacion/
    ├── presentacion_grupo09.tex    # diapositivas Beamer (11 láminas 16:9 UNSAAC)
    ├── presentacion_grupo09.pdf    # presentación compilada con resultados reales
    └── guion_exposicion_10min.md   # guion cronometrado (3:20 por integrante) + 5 preguntas
```

Los originales **no se movieron ni modificaron**. Siguen en `../descargaDataSet/HAM10000`. `part_1` y `part_2` se usan juntas; no son clases ni splits.

`data/processed` contiene la partición fija en `split.csv`, las carpetas exportadas `train/val/test` (224×224 RGB sin aumento en disco), `dataset_procesado.csv` y `estadisticas.json`. El entrenamiento lee los originales vía `split.csv` para evitar doble compresión JPEG. `colab/empaquetar.py` genera `../proyecto_ham10000_colab.zip` (tamaño exacto informado al empaquetar), excluyendo checkpoints, `runs/`, resultados pesados, salidas del notebook, `data/processed/{train,val,test}` e imágenes originales. Incluye como referencia **solo `resultados/comparacion.csv` del experimento local**, junto con código completo, configuración, manifiestos, fuentes y entregables. Para reproducir el entorno CUDA registrado, el propio paquete contiene `python/requirements-cuda.txt`; instale además `python/requirements-common.txt` y, para Jupyter local/verificación del notebook, `python/requirements-notebook.txt`. Colab conserva su PyTorch compatible; no se instala allí la wheel local de Windows.

## Plan resumido (completado)

1. [x] Revisar el alcance y excluir despliegue web/móvil por indicación docente.
2. [x] Mantener un split único y validado (`StratifiedGroupKFold(20)`, 0 fugas; verificación previa de sesión con 0 pares $\mathrm{RMSE}<4.5$ entre lesiones).
3. [x] Verificar entorno local GPU (GTX 1650 4 GiB, driver 621.01, PyTorch 2.11.0+cu128) y configuración común (`batch=8`, FP32).
4. [x] Entrenar las tres CNN bajo receta homogénea, seleccionando `best.pt` por Macro F1 en `val`.
5. [x] Congelar decisiones y evaluar `test` limpio, perturbaciones (`noise_002`, `noise_005`, `blur_1`, `blur_2`) y mitigación espacial parcial (`median_3`, `gaussian_1`).
6. [x] Consolidar métricas, curvas ROC, matrices de confusión y gráficos comparativos en `resultados/`.
7. [x] Elaborar y compilar la presentación Beamer (`presentacion/presentacion_grupo09.pdf`) y el guion de exposición de 10 minutos (`presentacion/guion_exposicion_10min.md`).

Ver `PLAN_TRABAJO.md` para el detalle de fases completadas y evidencias verificadas.

## Entorno local

Se usa el Python de `../.venv`, ya comprobado con GTX 1650 y controlador 621.01. El Python global de Laragon puede seguir mostrando CPU-only. Para reproducir ese entorno local se incluyen PyTorch CUDA (`python/requirements-cuda.txt`), dependencias comunes (`python/requirements-common.txt`) y requisitos opcionales de Jupyter (`python/requirements-notebook.txt`) dentro del propio proyecto portable.

Desde `F:\unsaac\vision\Article -DATASET`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements-cuda.txt
.\.venv\Scripts\python.exe -m pip install -r .\proyecto_ham10000\python\requirements-common.txt
.\.venv\Scripts\python.exe .\verificar_gpu.py
```

PyTorch local: `2.11.0+cu128`; torchvision: `0.26.0+cu128`. CUDA del paquete no tiene que coincidir numéricamente con CUDA del controlador. No es necesario instalar CUDA Toolkit completo.

En los comandos siguientes, el directorio de trabajo es `proyecto_ham10000`:

```powershell
..\.venv\Scripts\python.exe python\ejecutar.py validar --datos ..\descargaDataSet\HAM10000 --split data\processed\split.csv
```

Si se preparan los datos por primera vez en otro proyecto:

```powershell
..\.venv\Scripts\python.exe python\ejecutar.py preparar --datos ..\descargaDataSet\HAM10000 --salida data\processed
```

No se sobrescribe un split diferente. La semilla del split es 42; cambiar la semilla de entrenamiento no debe cambiar las particiones.

## Entrenamiento: ejecutar después de revisar el protocolo

```powershell
..\.venv\Scripts\python.exe python\ejecutar.py entrenar --modelo resnet18 --datos ..\descargaDataSet\HAM10000 --split data\processed\split.csv --config configs\base.json --salida runs
```

Repetir sustituyendo el modelo por `efficientnet_b0` y `mobilenet_v3_small`. Se descargan los pesos ImageNet la primera vez; son adicionales al dataset y al paquete PyTorch. No se descargaron automáticamente al generar este proyecto.

Para reanudar el mismo experimento:

```powershell
..\.venv\Scripts\python.exe python\ejecutar.py entrenar --modelo resnet18 --datos ..\descargaDataSet\HAM10000 --split data\processed\split.csv --config configs\base.json --salida runs --resume
```

Cada época completada guarda `last.pt`, `best.pt`, historial y estado del optimizador, AMP y generadores aleatorios. Tras una interrupción se repite la época incompleta. No se garantiza reanudación exacta a mitad de lote ni identidad bit a bit entre GPUs distintas. Los checkpoints deben proceder de este proyecto; no cargue archivos de terceros sin confianza.

La configuración base usa entrada 224, batch 8, dos épocas de cabeza y hasta dieciocho de ajuste fino, parada temprana con paciencia 5 y AMP desactivado inicialmente. Train descarta el último lote incompleto para evitar BatchNorm con batch 1. Validación y test incluyen todas sus imágenes. Primero se debe comprobar el tiempo y la memoria reales del piloto.

## Evaluación

Validación:

```powershell
..\.venv\Scripts\python.exe python\ejecutar.py evaluar --checkpoint runs\resnet18_seed42\best.pt --datos ..\descargaDataSet\HAM10000 --split data\processed\split.csv --salida runs\resnet18_seed42\evaluacion
```

**Solo al congelar el protocolo**, añadir `--particion test --confirmar-test`. Repetir para `--condicion clean`, `noise_002`, `noise_005`, `blur_1` y `blur_2`. El ruido utiliza una semilla estable por imagen, idéntica entre modelos. Ninguna condición se usa para reajustar después de mirar test.

Para evaluar mitigación de ruido aditivo mediante filtrado espacial determinista previo a la normalización (`--mitigacion {none, median_3, gaussian_1}`, usando `ImageFilter.MedianFilter(3)` o `ImageFilter.GaussianBlur(1.0)`), añadir `--mitigacion` sobre muestras ruidosas:

```powershell
..\.venv\Scripts\python.exe python\ejecutar.py evaluar --checkpoint runs\resnet18_seed42\best.pt --datos ..\descargaDataSet\HAM10000 --split data\processed\split.csv --salida runs\resnet18_seed42\evaluacion --particion test --confirmar-test --condicion noise_005 --mitigacion median_3
```

Los resultados base (`--mitigacion none`) se guardan en `{split}_{condicion}` y las corridas mitigadas (`median_3` o `gaussian_1`) en `{split}_{condicion}_{mitigacion}` (por ejemplo, `test_noise_005_median_3` y `test_noise_005_gaussian_1`), sin colisionar con las cinco condiciones base.

La evaluación admite `--device cpu --num-workers 0` para inspeccionar un checkpoint CUDA sin editarlo; use una salida independiente, por ejemplo `evaluacion_cpu`. Las métricas registran dispositivo/workers de inferencia por separado del entrenamiento. Las salidas nuevas se publican juntas al finalizar `metrics.json` y `predictions.csv`; una carpeta existente nunca se sobrescribe. El cuaderno comprueba integridad y procedencia antes de reutilizarla; una evaluación incompleta detiene el flujo.

Consolidar resultados existentes, sin inventarlos:

```powershell
..\.venv\Scripts\python.exe python\ejecutar.py comparar --runs runs --salida resultados --subcarpeta-evaluacion evaluacion
```

Además de `comparacion.csv`, `curvas_*.png` y `confusion_*.png`, `comparar` genera curvas ROC OVR por modelo y comparativas (`roc_*.png`, `roc_comparativa_*.png`), barras comparativas de desempeño y eficiencia (`comparacion_metricas.png`) y la curva comparativa de sensibilidad y mitigación (`comparacion_robustez.png`).

No compare directamente latencias de laptop contra Colab como si fueran diferencias entre arquitecturas. Haga la comparación de velocidad de las tres en el mismo equipo y registre GPU, software y protocolo.

`comparar` rechaza evaluaciones duplicadas y entornos de inferencia mezclados. Seleccione una subcarpeta o raíces separadas para mediciones alternativas. Al interpretar tiempo/VRAM de entrenamiento, revise también las sesiones de `run.json`: reanudar en otra GPU impide considerarlas mediciones homogéneas.

## Cuaderno integrado local / Colab

Abrir `colab/HAM10000_Grupo09.ipynb` y seguir `colab/README_COLAB.md`. Usa el título formal del estudio y **el mismo paquete `python/ham10000`**, no otra implementación. Integra diagnóstico del intérprete/dependencias/GPU, inventario del código, validación del split y originales, inspección de runs, tres CNN, reanudación protegida, evaluación multimétrica, robustez/mitigación, ROC/figuras, discusión y exportación con procedencia.

- **VS Code/Jupyter local:** seleccionar el kernel `.venv`; descubre proyecto y originales o admite rutas explícitas. **No importa `google.colab` ni monta Drive.** No es necesario instalar ese paquete para corregir `ModuleNotFoundError`.
- **Colab alojado:** subir manualmente ambos ZIP a `Mi unidad/Grupo09HAM/`, autorizar Drive y extraer a `/content`. Runs y caché persisten en Drive. GPU gratuita y duración no garantizadas; no copiar `.venv` ni instalar controladores de Windows.
- **Controles seguros:** `ENTRENAR`, `EVALUAR`, `CONFIRMAR_TEST`, instalación, regeneración de figuras y exportación desactivados; hashes de originales activados. No cambia `num_workers=4` ni escribe otro protocolo al pasar a Colab.
- **Resultados:** muestra las evidencias verificadas disponibles o el CSV local publicado, claramente identificado. Las salidas nuevas van a `resultados_notebook/`, sin sustituir los resultados/entregables existentes.
- **Validación:** esquema nbformat, sintaxis, regresiones, rama Colab simulada y ejecución local completa en modo de inspección; **sin ejecución acreditada en una cuenta real de Colab**. No se han relanzado los entrenamientos reales.

Mantenimiento: `colab/generar_notebook.py` genera el cuaderno limpio desde código revisable; `colab/verificar_notebook.py --ejecutar-local` lo ejecuta con el mismo intérprete y verifica que los artefactos experimentales no cambien. Consulte la guía para exportación y actualización deliberada de LaTeX/presentación.

## Documentación y pruebas

```powershell
..\.venv\Scripts\python.exe latex\compilar.py
```

El informe ampliado (`latex/main.tex`) explica dataset, CNN, split, preprocesamiento, instalación, protocolo local/Colab, métricas, robustez, ética y reproducibilidad; el artículo IEEE (`latex/paper_ieee.tex`) es una versión separada en plantilla `IEEEtran`. La presentación final con los resultados experimentales completos y el guion cronometrado de 10 minutos se encuentran en `presentacion/presentacion_grupo09.tex` (`presentacion_grupo09.pdf`) y `presentacion/guion_exposicion_10min.md`.

Para las pruebas, desde la raíz superior:

```powershell
$env:PYTHONPATH = "$PWD\proyecto_ham10000\python"
.\.venv\Scripts\python.exe -m unittest discover -s .\proyecto_ham10000\python\tests -v
```

## Límites y uso responsable

- Agrupamiento por `lesion_id` y SHA-256 exacto, no por paciente: no existe `patient_id` en este CSV.
- Dos registros adicionales tienen bytes idénticos a otras imágenes; se conservan y se mantienen en el mismo grupo. La similitud perceptual entre lesiones distintas aún requiere revisión especializada.
- Test es interno; no es una validación externa ni garantiza generalización clínica.
- Desbalance severo y pocos ejemplos de clases minoritarias: reportar métricas por clase y no solo accuracy.
- No es un producto médico ni una herramienta de diagnóstico.
- Revisar licencia de la distribución utilizada antes de redistribuir imágenes. El ZIP del proyecto no incluye el dataset.
