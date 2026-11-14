"""Fuente mantenible del cuaderno integrado; no ejecuta entrenamiento ni evaluación."""

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent
TITLE = "Evaluación Comparativa de Tres Arquitecturas CNN para Clasificación de Lesiones Cutáneas Pigmentadas mediante Dermatoscopia (HAM10000)"


def generar():
    cells = []

    def cell(kind, identifier, source):
        value = dict(cell_type=kind, id=identifier, metadata={}, source=dedent(source).strip().splitlines(keepends=True))
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    def md(identifier, source):
        cell("markdown", identifier, source)

    def code(identifier, source):
        cell("code", identifier, source)

    md("portada", f"""
    # {TITLE}
    **Grupo N. 9 · UNSAAC · Visión por Computador (2026-II)**  
    Choquenaira Quispe, Noe Franklin (133962) · Porroa Sivana, Yeni Ruth (120893) · Colque Galindo, Jean Franco (193109)

    Cuaderno **local (VS Code/Jupyter) y Google Colab**. HAM10000 es el conjunto de imágenes, no un lugar de estudio.
    Se clasifican siete categorías de lesiones pigmentadas en imágenes dermatoscópicas; no es diagnóstico clínico ni segmentación.
    **Despliegue excluido.** Los tres entrenamientos y 24 evaluaciones publicados se ejecutaron en la GTX 1650 local.
    La ejecución en una cuenta real de Colab sigue **sin acreditarse**; abrir este cuaderno no produce resultados nuevos.

    ## Recorrido y requisitos
    1. Revisar controles; detectar entorno y localizar/descomprimir proyecto y originales.
    2. Diagnosticar dependencias y GPU; comprobar código común, split fijo y SHA-256 de imágenes.
    3. Inspeccionar protocolo y corridas; entrenar/reanudar **solo mediante activación explícita**.
    4. Validar; evaluar test protegido; revisar Precision/F1, ROC, robustez, mitigación y eficiencia.
    5. Generar figuras o exportar evidencias si se solicita; integrar con informe, artículo y exposición.

    **Local:** seleccione el kernel de `..\\.venv\\Scripts\\python.exe`. No instale `google.colab` ni monte Drive.
    Abra el cuaderno dentro del proyecto o indique las rutas en la siguiente celda.
    **Colab alojado:** suba manualmente `proyecto_ham10000_colab.zip` y `HAM10000.zip` a `Mi unidad/Grupo09HAM/`;
    active GPU si va a entrenar. Drive requiere su autorización; `/content` es temporal y la GPU gratuita no se garantiza.
    El ZIP de proyecto incluye **todo `python/ham10000`**, CLI, configuración, split, documentación y entregables;
    no contiene imágenes ni checkpoints. No se requieren claves de Kaggle.
    """)
    code("controles", '''
    # Rutas opcionales; None permite descubrimiento automático local.
    RUTA_PROYECTO = None  # Ej.: r"F:\\unsaac\\vision\\Article -DATASET\\proyecto_ham10000"
    RUTA_DATOS = None     # Raíz que contiene HAM10000_metadata.csv e imágenes originales.
    RUTA_DRIVE = "/content/drive/MyDrive/Grupo09HAM"
    RUTA_RUNS = None      # None: proyecto/runs local; Grupo09HAM/runs en Colab.
    RUTA_CONFIG = None    # None: configs/base.json. Nunca se cambian sus workers automáticamente.
    INSTALAR_COMUNES = False  # Instala solo requirements-common, nunca reemplaza torch/torchvision.
    VERIFICAR_HASHES = True   # Lee los 10 015 originales; tarda más que comprobar solo rutas.
    ENTRENAR = False
    MODELOS_ENTRENAR = ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]
    EVALUAR = False
    MODELOS_EVALUAR = ["resnet18", "efficientnet_b0", "mobilenet_v3_small"]
    CONFIRMAR_TEST = False    # Solo después de congelar todas las decisiones; no ajustar mirando test.
    DISPOSITIVO_EVAL = None   # None: CUDA disponible o CPU. Para nueva medición CPU, use otra subcarpeta.
    WORKERS_EVAL = 0          # Solo inferencia; no cambia los checkpoints/protocolo de entrenamiento.
    SUBCARPETA_EVAL = "evaluacion"  # Ej.: "evaluacion_cpu" para una medición independiente.
    GENERAR_FIGURAS = False
    EXPORTAR_EVIDENCIAS = False
    ''')
    md("inicio-texto", """
    ## 1. Inicialización compatible con ambos entornos
    La siguiente celda integra utilidades de biblioteca estándar desde `colab/runtime.py`.
    La importación de `google.colab` está **dentro de la rama Colab alojado**: nunca se ejecuta localmente.
    La extracción comprueba rutas, colisiones, enlaces, espacio y CRC al leer; solo publica un destino completo.
    Un destino no acreditado se rechaza en lugar de sobrescribirse. Use únicamente ZIP y checkpoints de confianza.
    """)
    code("utilidades-runtime", (ROOT / "runtime.py").read_text(encoding="utf-8"))
    code("inicializacion", '''
    import sys
    import subprocess
    import csv
    from collections import Counter
    from datetime import datetime, timezone

    EN_COLAB = es_colab_alojado()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if EN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
        BASE = Path(RUTA_DRIVE)
        WORK = Path("/content/ham10000_sesion")
        WORK.mkdir(parents=True, exist_ok=True)
        def copiar_zip(nombre):
            source = BASE / nombre
            if not source.is_file():
                raise FileNotFoundError(f"Suba manualmente {source} antes de continuar.")
            digest = hash_archivo(source)
            local = WORK / f"{digest[:12]}_{nombre}"
            if not local.exists():
                temporary = local.with_suffix(".copiando")
                shutil.copy2(source, temporary)
                if hash_archivo(temporary) != digest:
                    raise ValueError("Copia incompleta del ZIP; no se extrae.")
                temporary.replace(local)
            elif hash_archivo(local) != digest:
                raise ValueError(f"Copia temporal alterada: {local}. Elija otra carpeta de sesión.")
            return local, digest
        if RUTA_PROYECTO:
            PROYECTO = localizar_proyecto(RUTA_PROYECTO)
        else:
            zip_code, code_sha = copiar_zip("proyecto_ham10000_colab.zip")
            code_root = extraer_zip(zip_code, WORK / f"codigo_{code_sha[:12]}")
            PROYECTO = localizar_proyecto(code_root / "proyecto_ham10000")
        if RUTA_DATOS:
            DATOS = localizar_datos(PROYECTO, RUTA_DATOS)
        else:
            zip_data, data_sha = copiar_zip("HAM10000.zip")
            data_root = extraer_zip(zip_data, WORK / f"datos_{data_sha[:12]}")
            DATOS = localizar_datos(PROYECTO, data_root)
        os.environ.setdefault("TORCH_HOME", str(BASE / "torch_cache"))
    else:
        PROYECTO = localizar_proyecto(RUTA_PROYECTO)
        DATOS = localizar_datos(PROYECTO, RUTA_DATOS)
        BASE = PROYECTO
    RUNS = Path(RUTA_RUNS).expanduser().resolve() if RUTA_RUNS else BASE / "runs"
    # Las salidas nuevas del notebook no sustituyen resultados/ ni tablas publicadas.
    RESULTADOS_NUEVOS = BASE / "resultados_notebook"
    CONFIG = Path(RUTA_CONFIG).expanduser().resolve() if RUTA_CONFIG else PROYECTO / "configs/base.json"
    CLI = PROYECTO / "python/ejecutar.py"
    SPLIT = PROYECTO / "data/processed/split.csv"
    def ejecutar(*args):
        command = [sys.executable, str(CLI), *map(str, args)]
        print(subprocess.list2cmdline(command))
        subprocess.run(command, cwd=PROYECTO, check=True)
    print("Entorno:", "Google Colab alojado" if EN_COLAB else "Local / Jupyter / VS Code")
    print("Python:", sys.executable, "\\nProyecto:", PROYECTO, "\\nDatos:", DATOS, "\\nRuns:", RUNS)
    ''')
    md("dependencias-texto", """
    ## 2. Dependencias, dispositivo e inventario de código
    Se conserva PyTorch/TorchVision del entorno. En Colab no instale la wheel CUDA de la laptop.
    En Windows, para reproducir el entorno registrado, instale en un entorno separado
    `python/requirements-cuda.txt` y `python/requirements-common.txt`, y seleccione ese kernel.
    Si instala paquetes después de importarlos, reinicie el kernel y ejecute las celdas en orden.
    La inspección y evaluación explícita pueden usar CPU; **entrenar con el protocolo base exige CUDA**.
    """)
    code("dependencias", '''
    if INSTALAR_COMUNES:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(PROYECTO / "python/requirements-common.txt")], check=True)
    try:
        import torch
        import torchvision
        import numpy as np
        import PIL
        import sklearn
        import matplotlib
        from IPython.display import display, Markdown, Image as DisplayImage
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(f"Kernel incompatible o dependencias ausentes en {sys.executable}. "
                           "Revise README_COLAB.md; no instale google.colab para ejecutar localmente. "
                           f"Detalle original: {error}") from error
    sys.path.insert(0, str(PROYECTO / "python"))
    import ham10000
    if not Path(ham10000.__file__).resolve().is_relative_to(PROYECTO):
        raise RuntimeError("Otro paquete ham10000 está cargado. Reinicie el kernel con este proyecto.")
    from ham10000 import CLASES, MODELOS
    from ham10000.io import cargar_split, leer_csv, sha256
    from ham10000.artifacts import verificar_evaluacion
    from ham10000.engine import entorno
    AMBIENTE = entorno()
    print(json.dumps(AMBIENTE, indent=2, ensure_ascii=False))
    print("NumPy", np.__version__, "Pillow", PIL.__version__, "scikit-learn", sklearn.__version__,
          "matplotlib", matplotlib.__version__)
    if not torch.cuda.is_available():
        print("Sin CUDA: la revisión funciona; el entrenamiento base queda bloqueado. Evaluación CPU opcional.")
    print("Modelos:", ", ".join(MODELOS), "| Clases:", ", ".join(CLASES))
    for name in ("__init__", "io", "prepare", "data", "models", "engine", "metrics", "plots", "artifacts"):
        file = PROYECTO / f"python/ham10000/{name}.py"
        if not file.is_file():
            raise FileNotFoundError(f"Paquete incompleto: {file}")
        print(f"{file.relative_to(PROYECTO)}  SHA-256 {sha256(file)[:16]}…")
    ''')
    md("split-texto", """
    ## 3. Originales y partición fija: no volver a sortear
    El entrenamiento lee **JPEG originales + `split.csv`**; la exportación `train/val/test` es opcional para inspección.
    `StratifiedGroupKFold(20, shuffle=True, random_state=42)` agrupa por lesión y duplicados SHA-256;
    folds 0–2 test, 3–5 val, resto train. Se preservan 7 009/1 503/1 503 imágenes y 7 470 lesiones.
    Esta celda revisa rutas, identidades, categorías, grupos/lesiones/hashes entre particiones y el digest canónico.
    `VERIFICAR_HASHES=True` comprueba además el contenido de **todos** los originales. No se altera el dataset.
    """)
    code("validacion-datos", '''
    SPLIT_ESPERADO = "583779fb75f28758f49b555898d1a7b50fc466574e63dc207c61e644f944ba39"
    SPLIT_SHA = sha256(SPLIT)
    if SPLIT_SHA != SPLIT_ESPERADO:
        raise ValueError("split.csv no corresponde al protocolo publicado; no se regenera automáticamente.")
    FILAS = cargar_split(SPLIT, DATOS)
    CONTEOS = Counter(row["split"] for row in FILAS)
    if CONTEOS != {"train": 7009, "val": 1503, "test": 1503}:
        raise ValueError(f"Conteos inesperados: {CONTEOS}")
    for field in ("lesion_id", "group_id", "sha256"):
        seen = {}
        for row in FILAS:
            if row[field] in seen and seen[row[field]] != row["split"]:
                raise ValueError(f"Fuga entre particiones: {field}")
            seen[row[field]] = row["split"]
    for partition in ("train", "val", "test"):
        if {r["dx"] for r in FILAS if r["split"] == partition} != set(CLASES):
            raise ValueError(f"No están las siete categorías en {partition}")
    if VERIFICAR_HASHES:
        ejecutar("validar", "--datos", DATOS, "--split", SPLIT)
    else:
        print("Solo estructura/split/rutas comprobados; contenido de imágenes NO rehasheado en esta sesión.")
    print("Split SHA-256:", SPLIT_SHA, "\\nImágenes:", len(FILAS), "| Lesiones:", len({r["lesion_id"] for r in FILAS}))
    print("Clase       train    val   test")
    for c in CLASES:
        values = [sum(r["dx"] == c and r["split"] == s for r in FILAS) for s in ("train", "val", "test")]
        print(f"{c:<10} {values[0]:>6} {values[1]:>6} {values[2]:>6}")
    ''')
    md("protocolo-texto", """
    ## 4. Preprocesamiento, protocolo y corridas disponibles
    `data.py`: RGB, letterbox isotrópico 224×224, normalización ImageNet; aumentos deterministas solo en train.
    `models.py`: ResNet-18, EfficientNet-B0 y MobileNetV3-Small, pesos `IMAGENET1K_V1`, cabeza de siete salidas.
    `engine.py`: dos épocas de cabeza y ajuste fino hasta 20, AdamW, pérdida ponderada con frecuencias de train,
    parada temprana y selección por Macro F1 de validación. FP32, batch 8, semilla 42; no cambiar estos valores
    silenciosamente. El `num_workers=4` del protocolo se conserva también en Colab.

    La reanudación exige **configuración completa y split idénticos**. Si se necesita otro protocolo, use otro JSON y
    otra raíz de runs; no lo mezcle con el experimento publicado. Misma semilla no garantiza igualdad bit a bit entre GPU/versions.
    """)
    code("protocolo", '''
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    print("Configuración:", CONFIG)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("\\nModelo                   Estado          Épocas   best.pt   last.pt")
    for modelo in MODELOS:
        run = RUNS / f"{modelo}_seed{config['seed']}"
        record = json.loads((run / "run.json").read_text(encoding="utf-8")) if (run / "run.json").is_file() else {}
        print(f"{modelo:<25} {record.get('status', 'sin corrida'):<15} {record.get('completed_epochs', '-'):>5}"
              f" {str((run / 'best.pt').is_file()):>9} {str((run / 'last.pt').is_file()):>9}")
        if record and (record.get("config") != config or record.get("split_sha256") != SPLIT_SHA):
            print("  ADVERTENCIA: protocolo diferente; no se reanudará ni se mezclará automáticamente.")
    ''')
    md("entrenar-texto", """
    ## 5. Entrenamiento/reanudación explícitos de las tres CNN
    Por defecto **no se entrena**. Para nuevas corridas, cambie `ENTRENAR=True` y elija modelos en los controles.
    Se descargan los pesos ImageNet la primera vez. Un `last.pt` permite continuar desde la última época completa;
    un run ya terminado se conserva sin reescribir. Carpetas sin procedencia/checkpoint se rechazan.
    Colab escribe checkpoints/métricas en Drive para persistencia; no garantiza sobrevivir a todo fallo del montaje.
    """)
    code("entrenamiento", '''
    if ENTRENAR:
        if not MODELOS_ENTRENAR or not set(MODELOS_ENTRENAR).issubset(MODELOS):
            raise ValueError("Seleccione arquitecturas válidas.")
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Este cuaderno no inicia el entrenamiento base sin CUDA. Revise GPU/kernel.")
        for modelo in MODELOS_ENTRENAR:
            run = RUNS / f"{modelo}_seed{config['seed']}"
            command = ["entrenar", "--modelo", modelo, "--datos", DATOS, "--split", SPLIT,
                       "--config", CONFIG, "--salida", RUNS]
            if run.exists() and any(run.iterdir()):
                if not (run / "last.pt").is_file() or not (run / "run.json").is_file():
                    raise RuntimeError(f"Corrida incompleta sin checkpoint/procedencia: {run}. Elija otra raíz.")
                record = json.loads((run / "run.json").read_text(encoding="utf-8"))
                if record["config"] != config or record["split_sha256"] != SPLIT_SHA:
                    raise ValueError(f"Protocolo incompatible en {run}; no se cambia ni sobrescribe.")
                command.append("--resume")
            ejecutar(*command)
    else:
        print("Entrenamiento desactivado. No se modificaron checkpoints, historias ni runs.")
    ''')
    md("evaluacion-texto", """
    ## 6. Validación y test protegido; detección de resultados incompletos
    `EVALUAR=False`: solo inspección posterior. `EVALUAR=True` evalúa clean en val para las tres arquitecturas.
    Además, **solo** con `CONFIRMAR_TEST=True`, evalúa 21 escenarios finales:
    clean, ruido σ=0.02/0.05 y blur σ=1/2; mediana 3×3 y gaussiano σ=1 sobre noise_005 por modelo.
    Test no se usa para seleccionar hiperparámetros. Los filtros son mitigación experimental parcial, no restauración clínica.

    Una carpeta solo se reutiliza tras comprobar ambos archivos, procedencia, todas las muestras y probabilidades;
    una evaluación incompleta detiene la celda, no se omite silenciosamente ni sobrescribe.
    Las evaluaciones nuevas se publican juntas al finalizar. Para medir en CPU o repetir en otro hardware,
    indique `DISPOSITIVO_EVAL="cpu"` y otra `SUBCARPETA_EVAL`/raíz; el checkpoint CUDA no se modifica.
    """)
    code("evaluacion", '''
    if not SUBCARPETA_EVAL or Path(SUBCARPETA_EVAL).name != SUBCARPETA_EVAL or SUBCARPETA_EVAL in (".", ".."):
        raise ValueError("SUBCARPETA_EVAL debe ser un nombre simple.")
    if EVALUAR:
        if not MODELOS_EVALUAR or not set(MODELOS_EVALUAR).issubset(MODELOS):
            raise ValueError("Seleccione arquitecturas válidas.")
        partition = "test" if CONFIRMAR_TEST else "val"
        scenarios = [("clean", "none")]
        if CONFIRMAR_TEST:
            scenarios += [(c, "none") for c in ("noise_002", "noise_005", "blur_1", "blur_2")]
            scenarios += [("noise_005", m) for m in ("median_3", "gaussian_1")]
        device_eval = DISPOSITIVO_EVAL or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoints = [RUNS / f"{m}_seed{config['seed']}" / "best.pt" for m in MODELOS_EVALUAR]
        missing = [str(p) for p in checkpoints if not p.is_file()]
        if missing:
            raise FileNotFoundError("Faltan checkpoints; el ZIP portable no los incluye: " + ", ".join(missing))
        for checkpoint in checkpoints:
            checkpoint_sha = sha256(checkpoint)
            for condition, mitigation in scenarios:
                suffix = "" if mitigation == "none" else f"_{mitigation}"
                output = checkpoint.parent / SUBCARPETA_EVAL
                target = output / f"{partition}_{condition}{suffix}"
                if target.exists():
                    old = verificar_evaluacion(target, checkpoint_sha, SPLIT_SHA, partition, condition, mitigation, FILAS)
                    previous_device = old.get("evaluation_device", "cuda" if old["environment"].get("gpu") else "cpu")
                    if DISPOSITIVO_EVAL and previous_device != DISPOSITIVO_EVAL:
                        raise ValueError("Evaluación existente de otro dispositivo. Use otra SUBCARPETA_EVAL.")
                    print("Evaluación íntegra conservada:", target, "| hardware registrado:", old["environment"].get("gpu"))
                    continue
                command = ["evaluar", "--checkpoint", checkpoint, "--datos", DATOS, "--split", SPLIT,
                           "--salida", output, "--particion", partition, "--condicion", condition,
                           "--mitigacion", mitigation, "--device", device_eval, "--num-workers", WORKERS_EVAL]
                if CONFIRMAR_TEST:
                    command.append("--confirmar-test")
                ejecutar(*command)
    else:
        print("Evaluación desactivada. Se pueden inspeccionar los resultados locales publicados sin ejecutar test.")
    ''')
    md("resultados-texto", """
    ## 7. Inspección de evidencias reales y comparación
    Se muestran métricas registradas, **no estimaciones**. La tabla CSV incluida es referencia del experimento local GTX 1650,
    no prueba de ejecución en esta sesión ni en Colab. Si hay checkpoints y evaluaciones en `RUNS`, se verifica su procedencia
    antes de leerlos. Macro Precision, Macro F1, Balanced Accuracy, Recall de melanoma y Macro AUC OVR son complementarias.
    La matriz y `classification_report` conservan Precision/Recall/F1 por clase; ROC se calcula desde probabilidades de `predictions.csv`.
    Latencia es forward FP32 batch 1 (10 warmup + 50 mediciones), sin lectura/preprocesamiento; no es latencia clínica extremo a extremo.
    No promediar ni comparar velocidad de CPU/GPU diferentes como si fueran equivalentes.
    """)
    code("lectura-resultados", '''
    METRICAS = []
    for modelo in MODELOS:
        run = RUNS / f"{modelo}_seed{config['seed']}"
        checkpoint = run / "best.pt"
        folders = sorted((run / SUBCARPETA_EVAL).iterdir()) if (run / SUBCARPETA_EVAL).is_dir() else []
        checkpoint_sha = sha256(checkpoint) if checkpoint.is_file() else None
        for target in folders:
            if not target.is_dir() or target.name.startswith("."):
                continue
            if not checkpoint_sha:
                raise FileNotFoundError(f"No se puede acreditar {target} sin best.pt.")
            if not (target / "metrics.json").is_file():
                raise ValueError(f"Evaluación incompleta: {target}; no se presenta como resultado válido.")
            record = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
            METRICAS.append(verificar_evaluacion(target, checkpoint_sha, SPLIT_SHA, record["split"],
                            record["condition"], record.get("mitigation", "none"), FILAS))
    referencia = PROYECTO / "resultados/comparacion.csv"
    TABLA = METRICAS if METRICAS else (leer_csv(referencia) if referencia.is_file() else [])
    ORIGEN = "Evaluaciones verificadas de RUNS (consulte ambiente registrado)" if METRICAS else "Referencia publicada: GTX 1650 local; no ejecución nueva"
    print(ORIGEN, "|", len(TABLA), "registros")
    if TABLA and any(r["split_sha256"] != SPLIT_SHA for r in TABLA):
        raise ValueError("La tabla contiene otra partición; no se combina.")
    if TABLA:
        print("Modelo                 split  Precision   F1 macro   Bal.Acc  Recall mel  AUC OVR")
        for r in TABLA:
            if r["condition"] == "clean" and r.get("mitigation", "none") == "none":
                print(f"{r['model']:<23} {r['split']:<5} " + " ".join(f"{float(r[k]):>10.4f}" for k in
                      ("macro_precision", "macro_f1", "balanced_accuracy", "melanoma_recall", "macro_auc_ovr")))
        print("\\nRobustez/mitigación (test): modelo | condición | filtro | Macro F1 | Recall mel")
        for r in TABLA:
            if r["split"] == "test":
                print(r["model"], r["condition"], r.get("mitigation", "none"),
                      f"{float(r['macro_f1']):.4f}", f"{float(r['melanoma_recall']):.4f}")
        print("\\nEficiencia registrada en test clean: modelo | hardware | mediana/P95 ms | VRAM MiB | parámetros | entrenamiento s")
        for r in TABLA:
            if r["split"] == "test" and r["condition"] == "clean" and r.get("mitigation", "none") == "none":
                hardware = "CPU" if r.get("evaluation_device") == "cpu" else (r.get("environment", {}).get("gpu") or r.get("gpu") or "CPU")
                print(r["model"], hardware, *[r.get(k) for k in ("latency_median_ms", "latency_p95_ms",
                      "peak_training_allocated_mib", "parameters", "training_seconds")])
        for r in METRICAS:
            if r["split"] == "test" and r["condition"] == "clean":
                print("\\nPrecision/Recall/F1 por clase y matriz:", r["model"])
                display({c: r["classification_report"][c] for c in CLASES})
                display(r["confusion_matrix"])
    else:
        print("No hay métricas disponibles; no se inventan. Entrene/evalúe explícitamente o incorpore evidencias acreditadas.")
    ''')
    md("figuras-texto", """
    ## 8. Figuras y consolidación sin sustituir los entregables publicados
    `GENERAR_FIGURAS=True` crea gráficos descriptivos y, si existen evaluaciones verificadas en la carpeta estándar,
    curvas de entrenamiento, matrices, ROC, comparación y robustez en `resultados_notebook/`.
    Para una evaluación CPU independiente, conserve su carpeta separada y no la mezcle con `evaluacion/`.
    Las figuras se muestran **individualmente**, con color y alta resolución; no se regeneran por defecto.
    """)
    code("figuras", '''
    if GENERAR_FIGURAS:
        if SUBCARPETA_EVAL != "evaluacion":
            raise ValueError("La consolidación estándar usa evaluacion/. Mantenga las mediciones alternativas separadas.")
        if METRICAS:
            environments = {(r.get("evaluation_device", "cuda" if r["environment"].get("gpu") else "cpu"),
                             r["environment"].get("gpu"), r["environment"].get("torch")) for r in METRICAS}
            if len(environments) > 1:
                raise ValueError("Hardware/entornos de evaluación mezclados; separe las raíces antes de comparar eficiencia.")
        ejecutar("figuras", "--datos", DATOS, "--split", SPLIT, "--salida", RESULTADOS_NUEVOS / "dataset")
        if METRICAS:
            ejecutar("comparar", "--runs", RUNS, "--salida", RESULTADOS_NUEVOS,
                     "--subcarpeta-evaluacion", SUBCARPETA_EVAL)
    for name in ("comparacion_metricas.png", "roc_comparativa_test_clean.png", "comparacion_robustez.png"):
        path = RESULTADOS_NUEVOS / name
        if not path.is_file():
            path = PROYECTO / "latex/img" / name
        if path.is_file():
            print("Figura almacenada:", path, "(no implica regeneración en esta sesión)")
            display(DisplayImage(filename=str(path), width=950))
    ''')
    md("discusion", """
    ## 9. Discusión científica y entrega
    - Bajo el protocolo/GPU locales, ResNet-18 mostró el mejor balance observado rendimiento–latencia.
      Su diferencia de Macro F1 con EfficientNet-B0 es pequeña (~0.00084); no demuestra superioridad estadística ni universal.
    - EfficientNet-B0 tuvo mayor Recall de melanoma en clean; examine también Precision y falsos positivos.
    - Ruido y desenfoque degradan los modelos; filtros pueden recuperar sensibilidad a costa de más falsos positivos.
      En ResNet, clean→noise_005+Gaussian: Precision mel 0.5233→0.3648, Recall 0.6012→0.6905, FP 92→202.
    - Una sola semilla, un dataset, desbalance, etiquetas heterogéneas y ausencia de validación externa limitan la conclusión clínica.
      No extrapolar a fotografías convencionales de piel ni a pacientes/poblaciones no representados.

    **Integración documental:** `latex/main.tex` es el informe ampliado; `latex/paper_ieee.tex`, el artículo;
    `presentacion/presentacion_grupo09.tex` y `guion_exposicion_10min.md`, la exposición.
    Todos usan el título de esta portada. Las cifras publicadas pertenecen al experimento local terminado.
    Para actualizarlo con un experimento nuevo: audite **3 val + 21 test**, mismo split/protocolo y hardware comparable;
    revise CSV/JSON/predicciones y copie deliberadamente las figuras/tablas que correspondan. El cuaderno no cambia textos
    ni afirma que un estudio nuevo esté completo. Compile con `python latex/compilar.py` (requiere TeX), y la presentación
    con dos pasadas de `pdflatex`. El ZIP de proyecto conserva tablas/figuras/PDF ya compilados; por sí solo no reproduce el entrenamiento.
    Conclusiones, Recomendaciones y Referencias son secciones distintas; no se incluye despliegue.
    """)
    code("exportacion", '''
    if EXPORTAR_EVIDENCIAS:
        destination = BASE / f"evidencias_notebook_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.zip"
        files = {SPLIT, CONFIG}
        for modelo in MODELOS:
            run = RUNS / f"{modelo}_seed{config['seed']}"
            files.update(p for p in (run / "run.json", run / "history.csv") if p.is_file())
            if (run / SUBCARPETA_EVAL).is_dir():
                files.update(p for p in (run / SUBCARPETA_EVAL).rglob("*") if p.is_file() and p.suffix in (".json", ".csv"))
        if RESULTADOS_NUEVOS.is_dir():
            files.update(p for p in RESULTADOS_NUEVOS.rglob("*") if p.is_file() and p.suffix in (".csv", ".png", ".json"))
        BASE.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
            manifest = {"title": ''' + repr(TITLE) + ''', "runtime": AMBIENTE,
                        "colab_hosted": EN_COLAB, "split_sha256": SPLIT_SHA,
                        "note": "Exporta archivos existentes; no acredita nuevas ejecuciones. Checkpoints e imágenes excluidos.",
                        "files": []}
            for i, file in enumerate(sorted(files)):
                name = f"archivos/{i:04d}_{file.name}"
                archive.write(file, name)
                manifest["files"].append({"archive": name, "source": str(file), "sha256": sha256(file)})
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise ValueError("Exportación con error CRC")
        print("Evidencias exportadas:", destination)
    else:
        print("Exportación desactivada. No se copiaron datos ni checkpoints a servicios externos.")
    ''')
    md("errores-fuentes", """
    ## 10. Errores frecuentes y fuentes
    | Síntoma | Acción correcta |
    |---|---|
    | `No module named google.colab` local | Usar este cuaderno actualizado; nunca ejecutar una celda antigua que monte Drive localmente. |
    | No se encuentra proyecto/dataset | Revisar `RUTA_PROYECTO`/`RUTA_DATOS`; originales, no exportados. |
    | CUDA no disponible | Comprobar `sys.executable`/kernel; en Colab activar GPU. No relanzar entrenamiento accidental en CPU. |
    | Configuración/split no coinciden | Recuperar el JSON original o elegir una raíz nueva; no editar el checkpoint. |
    | Evaluación incompleta/incompatible | Conservar e inspeccionar esa carpeta; elegir otra salida. No omitirla como si fuera válida. |
    | Sin `best.pt` en el ZIP | Es deliberado: copiar manualmente runs propios de confianza o entrenar explícitamente. |
    | Reinicio de Colab | `/content` se pierde; montar Drive, reconstruir originales y continuar con mismo JSON/split. |

    Fuentes: [HAM10000, Tschandl et al. (2018)](https://doi.org/10.1038/sdata.2018.161),
    [FAQ oficial Colab](https://research.google.com/colaboratory/faq.html),
    [carga de checkpoints PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
    Documentación completa: `README.md`, `colab/README_COLAB.md`, `docs/REQUISITOS_Y_FUENTES.md` y bibliografía LaTeX.
    """)
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={
        "colab": {"name": "HAM10000_Grupo09.ipynb"},
        "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
        "language_info": {"name": "python"}, "title": TITLE,
    }, cells=cells)
    destination = ROOT / "HAM10000_Grupo09.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Cuaderno generado: {len(cells)} celdas; operaciones costosas desactivadas salvo validación de hashes.")


if __name__ == "__main__":
    generar()
