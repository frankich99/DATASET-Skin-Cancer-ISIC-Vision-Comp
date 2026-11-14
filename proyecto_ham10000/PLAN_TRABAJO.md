# Plan de ejecución y aceptación — Grupo 9

**Estudio:** Evaluación Comparativa de Tres Arquitecturas CNN para Clasificación de Lesiones Cutáneas Pigmentadas mediante Dermatoscopia (HAM10000).

## Alcance

Se implementó y completó la evaluación comparativa reproducible de tres arquitecturas CNN (**ResNet-18**, **EfficientNet-B0** y **MobileNetV3-Small**) para la clasificación de lesiones cutáneas pigmentadas mediante dermatoscopia (HAM10000) usando un flujo común auditable. El despliegue interactivo (web o móvil) quedó explícitamente excluido del alcance del proyecto por indicación docente. Todas las fases planificadas (Fases 1–9), incluyendo entrenamientos reales en GPU local, evaluación de robustez con mitigación espacial y preparación de la exposición oral de 10 minutos, han sido **completadas**.

## Fases (Fases 1–9 completadas)

| Fase | Actividad | Estado | Evidencia real de terminación |
|---|---|---|---|
| 1. Preparación | Auditar etiquetas, archivos y duplicados; fijar split sin fuga | **Completada** | `auditoria.json`, `split.csv` (`583779fb...`), `estadisticas.json`; 10 015 JPEG, 7 470 lesiones, 2 pares SHA-256 en `nv` y 0 fugas (`7 009 / 1 503 / 1 503`). El barrido $\mathrm{RMSE}<4.5$ en miniaturas $16\times 12$ (0 pares entre lesiones distintas) fue una verificación de auditoría de sesión sin artefacto independiente persistido. |
| 2. Entorno | Verificar GPU local y preparar paquete/cuaderno local-Colab | **Completada** | Acer Nitro AN515-45, NVIDIA GTX 1650 (4 GiB, driver 621.01), PyTorch 2.11.0+cu128 (`python/requirements-cuda.txt`), FP32; 22 pruebas del pipeline/notebook + 4 de descarga. Cuaderno integrado de 22 celdas: esquema, ejecución local de inspección, hashes de 10 015 originales y 61 artefactos experimentales sin cambios; rama Colab simulada, sin ejecución acreditada en nube. ZIP con código/entregables/CSV de referencia local, sin imágenes/checkpoints/runs; tamaño y CRC informados por el empaquetador. |
| 3. Piloto | Medir viabilidad de lote y memoria por modelo | **Completada** | `configs/base.json` congelado con `image_size=224`, `batch_size=8`, `num_workers=4`, `seed=42`, FP32 (`amp=false`). La comprobación piloto comparativa `num_workers=0/4` fue una verificación de auditoría de sesión sin artefacto independiente persistido. |
| 4. Comparación | Entrenar las tres CNN con el mismo protocolo | **Completada** | `runs/{resnet18,efficientnet_b0,mobilenet_v3_small}_seed42/` con `run.json`, `history.csv`, `last.pt` y `best.pt`: ResNet-18 (16 ép., 127.5 s/ép, 434.2 MiB), EfficientNet-B0 (20 ép., 188.6 s/ép, 805.7 MiB), MobileNetV3-Small (19 ép., 156.3 s/ép, 211.8 MiB). |
| 5. Validación | Elegir checkpoints sin mirar `test` | **Completada** | Selección por Macro F1 en `val`: ResNet-18 (`0.6927`, ép. 11), EfficientNet-B0 (`0.7059`, ép. 19), MobileNetV3-Small (`0.6213`, ép. 14). |
| 6. Evaluación final | Test limpio, perturbaciones y mitigación espacial | **Completada** | 24 evaluaciones (`runs/*/evaluacion/`): en `clean`, Macro F1 (Macro Prec.): ResNet-18 **0.6786** (0.6591), EffNet **0.6778** (**0.6741**), MobileNet **0.5700** (0.6474), $\Delta\mathrm{F1}=0.00084$ sin superioridad estadística; perturbaciones (`noise_002`, `noise_005` EffNet F1 0.1201, Rec. `mel` 0.0000; `blur_1`, `blur_2`); mitigación parcial `gaussian_1` sobre `noise_005` (**+0.14 a +0.33** Macro F1: `+0.1426`, `+0.2394`, `+0.3317`; en ResNet-18 Rec. `mel` `0.6012 -> 0.6905`, pero Prec. `mel` `0.5233 -> 0.3648`, `92 -> 202` FP). |
| 7. Eficiencia | Medir `forward()` `batch=1` y memoria en el mismo hardware | **Completada** | Mediana `forward()` `batch=1` FP32 en GTX 1650 (excluye E/S y filtrado en CPU) en limpio y rango entre corridas: ResNet-18 **7.22 ms** (6.86–9.46 ms; $3.4\times$ más rápida en limpio), MobileNetV3-Small **22.40 ms** (15.26–22.40 ms), EfficientNet-B0 **24.60 ms** (22.49–34.82 ms); consolidado en `resultados/comparacion.csv`. |
| 8. Artículo | Documentar metodología, auditoría y figuras reproducibles | **Completada** | Base documental LaTeX (`latex/main.pdf`, `latex/paper_ieee.pdf`, `latex/img/`) y síntesis completa de métricas reales en `resultados/` y `README.md`. |
| 9. Exposición | Preparar presentación y guion de 10 minutos | **Completada** | `presentacion/presentacion_grupo09.tex` (`presentacion_grupo09.pdf`, 11 diapositivas) y `presentacion/guion_exposicion_10min.md` (1 224 palabras habladas, reparto ~408 palabras por integrante + 5 preguntas frecuentes). |

## Reparto de responsabilidades del Grupo N. 9

- **Choquenaira Quispe, Noe Franklin (133962):** integración del código, entorno CUDA local, entrenamiento en GTX 1650 y Bloque 1 de exposición (`0:00 – 3:20`, Diapositivas 1–4).
- **Porroa Sivana, Yeni Ruth (120893):** auditoría SHA-256 de datos, protocolo experimental, revisión de tablas/figuras y Bloque 2 de exposición (`3:20 – 6:40`, Diapositivas 5–8).
- **Colque Galindo, Jean Franco (193109):** consolidación experimental, evaluación de robustez/mitigación, análisis de eficiencia GPU y Bloque 3 de exposición (`6:40 – 10:00`, Diapositivas 9–11).
- Los tres integrantes dominan split, arquitecturas, métricas, análisis de latencia, mitigación espacial y limitaciones del estudio.

## Controles metodológicos cumplidos

1. Se mantuvo exactamente el mismo split (`split_sha256: 583779fb75f28758f49b555898d1a7b50fc466574e63dc207c61e644f944ba39`) para las tres CNN.
2. Se mantuvo idéntica política de preprocesamiento (`letterbox` $224\times 224$) y aumento determinista por `(seed, época, image_id)` solo en `train`.
3. Se registró el preentrenamiento `IMAGENET1K_V1`, las 2 épocas de cabeza congelando `BatchNorm` y el ajuste fino hasta 18 épocas.
4. No se seleccionó modelo ni intensidad de ruido mirando `test`; `--confirmar-test` se ejecutó tras congelar `best.pt`.
5. Los pesos de clase se calcularon exclusivamente sobre `train` (`7 009` muestras) sin combinarlos con sobremuestreo en disco.
6. La receta común (`configs/base.json`) garantizó equidad comparativa bajo presupuesto fijo en 4 GiB VRAM.
7. Todas las latencias de cómputo (`forward()` `batch=1`, excluyendo lectura E/S y filtrado en CPU) se midieron en FP32 (10 warmup + 50 pasadas CUDA sincronizadas) en la misma GTX 1650.
8. El alcance inicial prioritario (semilla 42, test limpio, cinco condiciones de robustez y mitigación determinista `median_3` / `gaussian_1` sobre `noise_005`) quedó cerrado al 100%; repeticiones multisemilla (`3 semillas`) o ablaciones de filtrado quedan como trabajo futuro opcional sin modificar el entregable principal.
9. Los resultados y conclusiones se ciñen estrictamente a métricas verificadas en `resultados/comparacion.csv`, declarando el carácter estrictamente académico y no clínico del estudio.
