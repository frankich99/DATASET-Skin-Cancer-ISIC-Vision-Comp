# Guion de Exposición Oral (10:00 minutos) — Grupo N. 9

**Título del estudio:** Evaluación Comparativa de Tres Arquitecturas CNN para Clasificación de Lesiones Cutáneas Pigmentadas mediante Dermatoscopia (HAM10000).

**Institución:** Universidad Nacional de San Antonio Abad del Cusco (UNSAAC) · FIEEIM · E. P. de Ingeniería Informática y de Sistemas  
**Asignatura:** Visión por Computador (2026-II)  
**Docente:** Mg. Hans Harley Ccacyahuillca Bejar  
**Material de apoyo:** `presentacion_grupo09.pdf` (11 diapositivas)

**Cuaderno de apoyo (no texto hablado):** `colab/HAM10000_Grupo09.ipynb`, integrado con el mismo paquete, protocolo y título. Funciona en modo de inspección local sin montar Drive; la rama Colab fue simulada, no ejecutada en una cuenta real. Las cifras expuestas son las de la GTX 1650 local.

---

## Distribución equitativa del tiempo (3 min 20 s por integrante)

| Bloque | Integrante | Código | Diapositivas | Tramo horario | Habla + transiciones |
|---|---|---|---|---|---|
| **1** | Choquenaira Quispe, Noe Franklin | 133962 | Diapositivas 1 – 4 | `0:00 – 3:20` | 409 palabras (~3 min 02 s + 18 s pausas) |
| **2** | Porroa Sivana, Yeni Ruth | 120893 | Diapositivas 5 – 8 | `3:20 – 6:40` | 407 palabras (~3 min 01 s + 19 s pausas) |
| **3** | Colque Galindo, Jean Franco | 193109 | Diapositivas 9 – 11 | `6:40 – 10:00` | 408 palabras (~3 min 01 s + 19 s pausas) |

> **Recomendación de ritmo:** Mantener una cadencia pausada (~135 palabras por minuto; aproximadamente 1 224 palabras habladas, ~9 min 04 s de habla + ~56 s de pausas y relevos). Es una estimación de 10:00, no una duración medida: confirmar con ensayo cronometrado.

---

## Bloque 1: Choquenaira Quispe, Noe Franklin (`0:00 – 3:20`)

### Diapositiva 1 — Portada (`0:00 – 0:25`, 25 s)
> *«Buenos días, Mg. Hans Harley Ccacyahuillca Bejar y compañeros. Somos el Grupo N. 9 de Visión por Computador, integrado por Yeni Ruth Porroa Sivana, Jean Franco Colque Galindo y quien les habla, Noe Franklin Choquenaira Quispe. Hoy presentaremos nuestra evaluación comparativa de tres arquitecturas CNN para la clasificación de lesiones cutáneas pigmentadas mediante dermatoscopia sobre el conjunto HAM10000 en GPU NVIDIA GTX 1650.»*

### Diapositiva 2 — Tarea dermatoscópica, 7 clases reales y desbalance (`0:25 – 1:25`, 60 s)
> *«HAM10000 reúne 10 015 imágenes dermatoscópicas de 7 470 lesiones únicas. A la izquierda observamos ejemplos reales de las siete categorías diagnósticas: lesiones malignas críticas como melanoma (`mel`), carcinomas basocelular (`bcc`) e intraepitelial (`akiec`), frente a lesiones benignas frecuentes (`bkl`, `df`, `nv`, `vasc`). Observen cómo `mel` y `nv` comparten bordes y pigmentos muy similares, lo que dificulta la separación visual.*
>
> *A la derecha, el gráfico evidencia un desbalance extremo de 58.3 a 1: el nevus (`nv`) concentra el 66.95% (6 705 imágenes). Por ello, una exactitud global trivial superaría el 66% sin valor clínico alguno. Asimismo, al existir múltiples tomas por lesión, la partición agrupada por `lesion_id` es obligatoria para evitar fuga de datos. Priorizamos **Macro F1**, **Macro Precisión**, **Recall de melanoma** y **AUC OVR».*

### Diapositiva 3 — Auditoría SHA-256 y partición fija sin fuga (`1:25 – 2:25`, 60 s)
> *«Aplicamos el pipeline de cinco etapas mostrado en la parte superior. Verificamos los 10 015 archivos frente al archivo de metadatos y depuramos carpetas duplicadas redundantes, bajando el tamaño en disco de 5.20 a 2.71 GiB sin alterar los archivos originales.*
>
> *Con huellas SHA-256 hallamos dos pares binariamente idénticos en `nv` (`HAM_0002300` y `HAM_0001895`) y los mantuvimos agrupados juntos. Como verificación previa de sesión sobre miniaturas de 16 por 12 píxeles, observamos cero pares entre lesiones distintas con RMSE menor a 4.5, sin sustituir una revisión perceptual dermatoscópica.*
>
> *Con `StratifiedGroupKFold` de 20 pliegues y semilla 42 obtuvimos 7 009 imágenes para entrenamiento, 1 503 para validación y 1 503 para prueba, con cero fugas y huella `split_sha256` fijada antes de entrenar.»*

### Diapositiva 4 — Preprocesamiento isotrópico y aumento determinista (`2:25 – 3:20`, 55 s)
> *«Descartamos las miniaturas de 8 por 8 y 28 por 28 píxeles y leímos directamente los JPEG originales en RGB respetando orientación EXIF. Adaptamos cada imagen a 224 por 224 mediante **letterbox** bilineal isotrópico relleno con la media ImageNet, evitando recortar la periferia de la lesión.*
>
> *Calculamos las estadísticas empíricas solo sobre `train` y aplicamos la normalización estándar requerida por los pesos preentrenados. Finalmente, el aumento en línea —volteos horizontal y vertical, rotación de $\pm 30^\circ$ y variación de brillo y contraste entre 0.9 y 1.1— se restringe a `train` con semilla SHA-256 por época e imagen, garantizando aumentos idénticos para las tres redes. Cedo la palabra a mi compañera Yeni Porroa.»*

---

## Bloque 2: Porroa Sivana, Yeni Ruth (`3:20 – 6:40`)

### Diapositiva 5 — Tres arquitecturas CNN comparadas (`3:20 – 4:10`, 50 s)
> *«Gracias, Noe. Para contrastar familias estructurales distintas en igualdad de condiciones, adaptamos tres redes preentrenadas con pesos `IMAGENET1K_V1`, reemplazando únicamente su capa lineal final hacia las siete clases de HAM10000.*
>
> *Primero, **ResNet-18**, referencia residual clásica con convoluciones densas $3\times 3$ y ReLU, con 11.18 millones de parámetros. Segundo, **EfficientNet-B0**, con escalado compuesto, bloques MBConv, atención Squeeze-and-Excitation y SiLU, con 4.02 millones. Y tercero, **MobileNetV3-Small**, orientada a bajo consumo de memoria mediante convoluciones separables en profundidad, módulos SE y `hard-swish`, con solo 1.53 millones de parámetros, 7.3 veces menos que ResNet-18.»*

### Diapositiva 6 — Protocolo homogéneo y entrenamiento en GTX 1650 (`4:10 – 5:05`, 55 s)
> *«Entrenamos las tres redes en la misma laptop con GPU NVIDIA GTX 1650 de 4 GiB, PyTorch 2.11.0+cu128, FP32, lote 8, 4 workers y semilla 42; el despliegue interactivo web o móvil quedó excluido del alcance por indicación docente.*
>
> *Aplicamos dos etapas: 2 épocas de cabeza con el extractor congelado y `BatchNorm` en modo evaluación, más hasta 18 épocas de ajuste fino con AdamW y decaimiento coseno, usando entropía cruzada ponderada solo con frecuencias de `train` y selección por Macro F1 en validación.*
>
> *En la tabla derecha, ResNet-18 activó parada temprana en época 16 (mejor en 11; 127.5 segundos por época y 434.2 MiB); EfficientNet-B0 completó 20 épocas (mejor Val F1 0.7059; 188.6 segundos por época y 805.7 MiB) y MobileNetV3-Small llegó a 19 épocas con apenas 211.8 MiB de memoria pico.»*

### Diapositiva 7 — Desempeño global en test limpio (`5:05 – 5:55`, 50 s)
> *«Congelados los checkpoints, evaluamos el conjunto `test` limpio de 1 503 imágenes. En la columna destacada en naranja evaluamos la **Macro Precisión**, es decir, qué porcentaje de las predicciones positivas fueron correctas, evitando falsas alarmas.*
>
> *Aquí **EfficientNet-B0 lidera con 67.41%** (frente a 65.91% de ResNet-18 y 64.74% de MobileNetV3-Small), destacando especialmente en queratosis actínica (`akiec`) con 85.71% y nevus (`nv`) con 92.97%. Por su parte, **ResNet-18** logra el mejor balance global: lidera en **Exactitud (80.31%)**, **Macro F1 (0.6786)** y **AUC OVR (0.9596)**, siendo además **3.4 veces más rápida en inferencia GPU (7.22 ms frente a 24.60 ms)**. MobileNetV3-Small queda rezagada con 0.5700 en Macro F1 y 0.4167 en Recall de melanoma.»*

### Diapositiva 8 — Curvas ROC y matriz de confusión (`5:55 – 6:40`, 45 s)
> *«Las curvas Macro-ROC OVR a la izquierda muestran una sólida separación probabilística en limpio, todas por encima de 0.94 de AUC. En melanoma (`mel`, 168 casos), EfficientNet-B0 obtiene mayor AUC específico (0.911) con 111 aciertos, frente a 0.895 y 101 aciertos de ResNet-18.*
>
> *En la matriz de confusión de ResNet-18 a la derecha observamos buen acierto en `nv` (886 de 1 005) y en lesiones vasculares `vasc` (19 de 21), concentrándose la confusión en la frontera visual entre melanoma, nevus y queratosis benigna (`bkl`). Ahora Jean Franco presentará robustez y eficiencia GPU.»*

---

## Bloque 3: Colque Galindo, Jean Franco (`6:40 – 10:00`)

### Diapositiva 9 — Sensibilidad al ruido/desenfoque y mitigación espacial (`6:40 – 7:45`, 65 s)
> *«Gracias, Yeni. Evaluamos los modelos congelados en prueba bajo ruido gaussiano aditivo ($\sigma=0.02$ y $0.05$) y desenfoque ($r=1$ y $2$). ResNet-18 resiste mejor el desenfoque (0.5769 y 0.4895 de Macro F1). En cambio, como hipótesis interpretativa por su sensibilidad a texturas de alta frecuencia y bloques SE, EfficientNet-B0 colapsa en `noise_005` a 0.1201 de Macro F1 y 0.0000 de Recall en melanoma, prediciendo casi siempre la clase mayoritaria `nv`.*
>
> *Al prefiltrar `noise_005` sin reentrenar, `gaussian_1` aporta entre **+0.14 y +0.33 puntos** de Macro F1 (+0.1426 en MobileNetV3-Small hasta 0.4505; +0.2394 en ResNet-18 hasta 0.5104; +0.3317 en EfficientNet-B0 hasta 0.4518). Sin embargo, en ResNet-18 es una **mitigación parcial**: aunque el Recall de melanoma sube de 0.6012 a 0.6905, su **Precisión cae de 0.5233 a 0.3648**, duplicando los falsos positivos de 92 a 202 casos.»*

### Diapositiva 10 — Eficiencia GPU: `forward()` `batch=1` y posibles factores (`7:45 – 8:50`, 65 s)
> *«Un resultado llamativo para la ingeniería es que MobileNetV3-Small, con 7.3 veces menos parámetros y solo 211.8 MiB de VRAM, sea 3.1 veces más lenta en limpio que ResNet-18 (22.40 frente a 7.22 ms), y EfficientNet-B0 alcance 24.60 ms. Cabe precisar que estas latencias miden exclusivamente el paso `forward()` en GPU con `batch=1`, excluyendo lectura de disco y filtrado en CPU, y varían entre corridas de 6.86 a 9.46 ms en ResNet-18, 15.26 a 22.40 ms en MobileNetV3-Small y 22.49 a 34.82 ms en EfficientNet-B0.*
>
> *Proponemos tres hipótesis interpretativas sin perfilado por kernel: primero, mayor sobrecarga de lanzamiento por fragmentación de micro-operaciones secuenciales; segundo, menor intensidad aritmética en convoluciones depthwise frente a kernels densos $3\times 3$ de cuDNN; y tercero, mayor tráfico de memoria global por módulos SE y activaciones compuestas.»*

### Diapositiva 11 — Limitaciones, conclusiones y veredicto final (`8:50 – 10:00`, 70 s)
> *«Para cerrar, reconocemos cuatro limitaciones del estudio: falta de identificador de paciente (`patient_id`) en el CSV oficial, sesgo hacia piel clara en un test interno, alta varianza discreta en clases escasas como `df` y `vasc` bajo semilla única 42, y carácter estrictamente académico sin validación clínica ni despliegue interactivo.*
>
> *En síntesis, consolidamos un flujo reproducible y auditable con cero fugas y AUC superior a 0.94 en limpio; identificamos el colapso ante ruido aditivo y cuantificamos que `gaussian_1` actúa solo como mitigación parcial debido al aumento de falsos positivos en melanoma.*
>
> *Como **veredicto final**, sin superioridad estadística en Macro F1 frente a EfficientNet-B0 ($\Delta=0.00084$), **ResNet-18** representa el **mejor balance observado** en este protocolo y GTX 1650: lidera en exactitud balanceada (0.7174), AUC OVR (0.9596), menor latencia `forward()` (7.22 ms) y mejor respuesta parcial con `gaussian_1`. Muchas gracias por su atención; quedamos atentos a sus preguntas.»*

---

## Banco de 5 posibles preguntas del docente y respuestas de precisión

### 1. ¿Por qué no utilizaron `Accuracy` como métrica principal de selección y qué impacto tuvo la pérdida ponderada?
- **Respuesta:** En HAM10000, la clase `nv` representa el 66.95% de los datos (desbalance 58.3:1 frente a `df`). Un modelo que ignore por completo las seis clases minoritarias obtendría ~67% de *Accuracy*. Por eso seleccionamos checkpoints por **Macro F1** en validación, donde cada una de las 7 clases pesa $\frac{1}{7}$. En entrenamiento aplicamos entropía cruzada ponderada calculada estrictamente sobre `train` ($w_{\mathrm{df}}=12.5161$, $w_{\mathrm{vasc}}=10.0129$, $w_{\mathrm{mel}}=1.2853$ vs. $w_{\mathrm{nv}}=0.2133$), normalizada en PyTorch por la suma de pesos de las etiquetas del lote, logrando que ResNet-18 alcance una exactitud balanceada de **0.7174** (70.6% de *Recall* en `df` y 90.5% en `vasc`) sin generar muestras sintéticas duplicadas en disco.

### 2. ¿Cómo explican que ResNet-18 (11.18M parámetros) sea más rápida en `forward()` (`batch=1`) que MobileNetV3-Small (1.53M)?
- **Respuesta:** Como hipótesis interpretativa (sin haber ejecutado perfilado por operador CUDA/Nsight), la reducción de parámetros teóricos de MobileNetV3-Small proviene de sustituir convoluciones densas $3\times 3$ por bloques invertidos con convoluciones *depthwise-separable*, módulos SE y `h-swish`. En GTX 1650 con `batch=1` y FP32, ello fragmentaría la ejecución en múltiples kernels pequeños de baja intensidad aritmética y mayor acceso a memoria global, frente a convoluciones densas $3\times 3$ muy optimizadas en cuDNN. Medimos exclusivamente `forward()` sin E/S ni filtrado en CPU, registrando dispersión entre corridas de **6.86–9.46 ms** (ResNet-18), **15.26–22.40 ms** (MobileNetV3-Small) y **22.49–34.82 ms** (EfficientNet-B0). Donde MobileNetV3-Small sí reduce recursos es en VRAM pico de entrenamiento: **211.8 MiB** frente a **434.2 MiB** de ResNet-18 y **805.7 MiB** de EfficientNet-B0.

### 3. ¿Por qué EfficientNet-B0 colapsa ante ruido `noise_005` (Recall `mel` = 0.0000) y por qué `gaussian_1` es solo mitigación parcial?
- **Respuesta:** Como hipótesis interpretativa, el ruido gaussiano aditivo ($\sigma=0.05$) altera la textura fina de alta frecuencia; en EfficientNet-B0, su dependencia de bloques MBConv expandidos y atención SE sensible a activaciones globales coincidiría con el colapso de separabilidad (AUC cae a **0.4839**, prediciendo `nv` en el 92.7% de los `nv` y en 156/168 melanomas). Al prefiltrar `noise_005` antes de normalizar, `gaussian_1` aporta entre **+0.14 y +0.33** puntos de Macro F1 (+0.1426, +0.2394 y +0.3317), superando a `median_3` (0.3804, 0.4624 y 0.4383). Sin embargo, es una **mitigación parcial** y no una restauración plena: en ResNet-18, el Recall de melanoma sube de **0.6012** en limpio (**0.1250** en `noise_005`) a **0.6905**, pero su Precisión en melanoma cae de **0.5233** a **0.3648**, aumentando los falsos positivos de **92 a 202** y quedando su Macro F1 en **0.5104** (lejos del **0.6786** limpio).

### 4. ¿Cómo garantizan que no hubo fuga de información (*data leakage*) en la evaluación y qué limitación subsiste en los metadatos?
- **Respuesta:** Aplicamos tres barreras verificables: (1) verificamos integridad por SHA-256 agrupando los 2 pares binariamente idénticos (`HAM_0002300` y `HAM_0001895`) y, como verificación previa de sesión sin artefacto independiente persistido, ejecutamos un barrido en miniaturas $16\times 12$ observando **0 pares** con $\mathrm{RMSE}<4.5$ entre `lesion_id` distintos (manteniendo abierta la limitación de similitud perceptual clínica); (2) particionamos con `StratifiedGroupKFold(20)` agrupando por unión de `lesion_id` y SHA-256, verificando 0 cruces entre `train` (7 009), `val` (1 503) y `test` (1 503) mediante `split_sha256` (`583779fb...`); y (3) bloqueamos `test` con `--confirmar-test` hasta congelar `best.pt` en validación. La limitación que subsiste es que `HAM10000_metadata.csv` no incluye `patient_id`.

### 5. Si EfficientNet-B0 tuvo mayor Macro Precisión (0.6741) y Recall de melanoma en limpio (0.6607), ¿por qué señalan a ResNet-18 como mejor balance observado?
- **Respuesta:** La diferencia puntual en Macro F1 entre ResNet-18 (**0.6786**) y EfficientNet-B0 (**0.6778**) es de apenas **0.00084** (IC95% bootstrap pareado sobre 1 118 grupos en test: `[-0.0429, +0.0464]`), por lo que no hay superioridad ni equivalencia estadística demostrada. No obstante, bajo el criterio conjunto en nuestro protocolo y GTX 1650, ResNet-18 logra el mejor balance observado: mayor exactitud global (**0.8031** vs. 0.8004), mayor exactitud balanceada (**0.7174** vs. 0.7025), mayor AUC OVR (**0.9596** vs. 0.9587), casi la mitad de VRAM (**434.2** vs. 805.7 MiB), latencia `forward()` **3.41×** menor en limpio (**7.22** vs. 24.60 ms) y mejor comportamiento ante desenfoque y ante `noise_005` con `gaussian_1` (Macro F1 **0.5104** frente a 0.4518, aunque con intercambio de precisión/FP en melanoma).
