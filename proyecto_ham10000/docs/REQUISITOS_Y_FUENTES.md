# Requerimientos, fuentes y alcance

**Título del estudio:** Evaluación Comparativa de Tres Arquitecturas CNN para Clasificación de Lesiones Cutáneas Pigmentadas mediante Dermatoscopia (HAM10000). Se aplica a informe, artículo, presentación, guion, documentación y notebook; los nombres originales del dataset y de las referencias bibliográficas se conservan.

## Material del docente

Se extrajeron y verificaron los requerimientos del PDF plantilla general proporcionado (`EspecificacionesProyectoDeepLearning.pdf`). Su núcleo exige tres modelos, preprocesamiento/partición, evaluación multimétrica, robustez, eficiencia, código y artículo LaTeX. Aunque dicho PDF plantilla general mencionaba el despliegue como entregable, el despliegue interactivo no aplica y queda explícitamente excluido del alcance del proyecto por indicación docente. Nota: el archivo `EspecificacionesProyectoDeepLearning.pdf` ya no se encuentra en el árbol local de archivos, pero sus requerimientos completos quedaron extraídos y verificados en este documento y en los entregables del proyecto.

La transcripción oral pide explicar los pasos seguidos, modelo y arquitectura; comparar al menos tres algoritmos clásicos o profundos, calcular métricas, elaborar informe y exponer diez minutos. Elegir tres CNN satisface la parte común del PDF y del audio. La referencia oral al miércoles 18 o 19 no se interpreta como una fecha confirmada.

Las imágenes sobre SLIC, Watershed, cortes de grafos y otros métodos describen segmentación clásica, una tarea diferente de la clasificación multiclase de HAM10000.

## Investigación y documentación consultada

| Fuente primaria | Uso en el proyecto |
|---|---|
| Tschandl, Rosendahl y Kittler, Scientific Data 2018; DOI 10.1038/sdata.2018.161 | Origen, categorías, verificación de etiquetas y dependencia por lesión |
| He et al., CVPR 2016 | Fundamento de conexiones residuales |
| Tan y Le, ICML 2019, PMLR 97 | EfficientNet y escalado compuesto |
| Howard et al., ICCV 2019 | MobileNetV3 y diseño orientado a hardware |
| Documentación oficial PyTorch/TorchVision | Pesos, modelos, normalización y checkpoints |
| Documentación oficial scikit-learn | Estratificación agrupada y métricas multiclase |
| FAQ oficial de Google Colab | Sesiones temporales, recursos variables y uso de Drive |
| NVIDIA y Acer, páginas oficiales | Compatibilidad del controlador y entorno local |

La bibliografía completa se incluye en `latex/referencias.bib`. Los resultados de otros artículos no se presentan como si fueran resultados del grupo. No se copiaron sus imágenes para ilustrar rendimiento; los mosaicos provienen del dataset local y los diagramas son elaboración propia.

## Estado de las evidencias

- Dataset local auditado (10 015 imágenes JPEG, 7 470 lesiones, 2 pares SHA-256 idénticos en `nv`) y split fijo generado con registros reales (`StratifiedGroupKFold(20)`, 7 009 / 1 503 / 1 503, `split_sha256: 583779fb...`, 0 fugas). La comprobación de pares casi-idénticos ($\mathrm{RMSE}<4.5$) y el piloto `num_workers=0/4` fueron verificaciones de sesión sin artefacto independiente persistido en el repositorio.
- Figuras de distribución, ejemplos, metadata, preprocesamiento, perturbaciones, curvas ROC, matrices de confusión y comparación de métricas/robustez generadas localmente a partir de datos y ejecuciones reales.
- **22 pruebas automáticas del pipeline y notebook** (`python/tests/`: 10 de pipeline, 2 de integración y 10 de runtime/artefactos), incluidas interrupción/reanudación real entre épocas con datos sintéticos, evaluación CPU de checkpoint CUDA, publicación de evaluaciones, extracción segura y rama Colab simulada; más **4 pruebas de descarga/preparación** (`descargaDataSet/test_descargar.py`). Las pruebas sintéticas no producen nuevas métricas HAM10000.
- **3 entrenamientos reales completos** ejecutados en GPU local NVIDIA GTX 1650 (ResNet-18, EfficientNet-B0 y MobileNetV3-Small, `seed=42`, FP32, `batch=8`, `num_workers=4`), con checkpoints y registros persistidos en `runs/`.
- **24 evaluaciones reales** consolidadas en `resultados/comparacion.csv` (3 modelos $\times$ [`val_clean` + 5 condiciones base en `test` + 2 mitigaciones `median_3`/`gaussian_1` sobre `noise_005`]), junto con reportes por clase y matrices de confusión en `runs/*/evaluacion/`.
- Informe técnico (`latex/main.tex`), artículo IEEE (`latex/paper_ieee.tex`), presentación Beamer (`presentacion/presentacion_grupo09.tex`) y guion oral (`presentacion/guion_exposicion_10min.md`) actualizados con los resultados reales y análisis crítico.
- Cuaderno integrado local/Colab (`colab/HAM10000_Grupo09.ipynb`, 22 celdas): esquema nbformat, sintaxis, controles seguros y ejecución local en modo de inspección; detección/extracción y rama Colab probadas con simulación. El mismo paquete Python se distribuye completo; no hay entrenamiento duplicado ni importación de Drive local. Configuración y split se conservan; no se reentrenan los experimentos terminados. **Sin ejecución acreditada en cuenta real de Google Colab**.
- Paquete portable con fuentes/entregables y CSV de referencia local, sin originales/checkpoints/runs. Resultados nuevos del notebook separados en `resultados_notebook/`; exportación opcional de evidencias con manifiesto SHA-256. Empaquetado posterior a la compilación y comprobación CRC.
