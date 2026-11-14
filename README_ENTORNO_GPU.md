# Entorno GPU del proyecto HAM10000

El entorno `.venv` es independiente de `C:\laragon\bin\python\python-3.12`. No sustituye ni desinstala el PyTorch global.

## Requisitos

- Windows de 64 bits, Python 3.12 y GTX 1650 de 4 GB.
- Controlador NVIDIA compatible instalado. En este equipo se verificó el 621.01.
- Aproximadamente 2.79 GB de descarga para las versiones seleccionadas y sus dependencias, más el pequeño paquete de actualización de pip si hace falta. Es recomendable disponer de al menos 10 GB libres para instalación, archivos temporales y caché.
- No hace falta instalar por separado CUDA Toolkit, cuDNN ni torchaudio para estas pruebas. El paquete binario incluye las bibliotecas de ejecución que necesita.

Se seleccionaron PyTorch 2.11.0 con CUDA 12.8 y torchvision 0.26.0. No se busca igualar el número de versión CUDA del controlador: un controlador más reciente puede ejecutar un paquete compilado para una versión anterior compatible. La comprobación final debe hacerse ejecutando operaciones reales en la GPU.

## Instalación reproducible

Desde `F:\unsaac\vision\Article -DATASET`, si todavía no existe `.venv`:

```powershell
python -m venv .venv
```

Después:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --timeout 60 --retries 20 --resume-retries 30 -r .\requirements-cuda.txt
```

La instalación requiere Internet. La reanudación de pip depende del servidor y sus límites de reintentos; no garantiza continuar después de cerrar el proceso. Si falla, repetir el comando reutiliza los paquetes que hayan quedado correctamente en caché.

## Verificación

```powershell
.\.venv\Scripts\python.exe .\verificar_gpu.py
.\.venv\Scripts\python.exe -m pip check
```

`verificar_gpu.py` comprueba detección CUDA, multiplicación matricial y forward/backward de ResNet-18, EfficientNet-B0 y MobileNetV3-Small con batch 2 a 224 x 224. Solo usa datos sintéticos: no modifica HAM10000, no descarga pesos preentrenados y no inicia el experimento comparativo.

Una prueba correcta valida ese tamaño de entrada y lote; no garantiza que otros tamaños o configuraciones quepan en 4 GB. La memoria mostrada corresponde a tensores asignados por PyTorch, no al consumo total de toda la GPU.

## Uso posterior

Usar siempre `.venv\Scripts\python.exe` o seleccionar ese intérprete en el editor. Ejecutar solamente `python` sin activar el entorno puede seguir usando el PyTorch CPU de Laragon.

No se han instalado todavía todas las bibliotecas de análisis, notebooks o entrenamiento del proyecto: este entorno prepara únicamente PyTorch con GPU y los modelos de visión.
