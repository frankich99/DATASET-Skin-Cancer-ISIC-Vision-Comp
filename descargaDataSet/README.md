# Descarga reanudable de HAM10000

Descarga pública de Kaggle, sin cuenta ni token mientras el endpoint permita acceso anónimo. Python 3.10 o posterior; sin instalar paquetes.

## Comando recomendado

Desde `F:\unsaac\vision\Article -DATASET`, copia:

```powershell
python .\descargaDataSet\descargar.py
```

El ZIP queda en `descargaDataSet\HAM10000.zip`. La ubicación no cambia aunque ejecutes el script desde otra carpeta.

Para guardarlo en Descargas sin interferir con la descarga anterior:

```powershell
python .\descargaDataSet\descargar.py --destino "$env:USERPROFILE\Downloads\HAM10000_python.zip"
```

No ejecutes dos descargas simultáneas: compartirían tu conexión. No uses el mismo destino que otro programa esté descargando.

## Qué descarga

- Dataset: **Skin Cancer MNIST: HAM10000**, versión 2, de `kmader`.
- Fuente: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Aproximadamente 2.9 GB; 10,015 imágenes, siete clases y metadatos. Incluye derivados CSV de baja resolución.
- No es el dataset ISIC de nueve clases que existe en el proyecto vecino.
- Reserva al menos 7 GB si también vas a descomprimirlo. El script solo descarga y verifica; no extrae.
- Respeta las condiciones de uso y licencia del dataset.

## Cortes de Internet y pausas

- Guarda progreso en `.zip.part` y metadatos en `.zip.resume.json`.
- Si falla la red o termina antes de tiempo, reintenta indefinidamente con esperas de hasta 30 segundos, salvo una espera mayor solicitada por el servidor.
- Al volver Internet, solicita únicamente los bytes pendientes mediante HTTP Range.
- `Ctrl+C` pausa. Repite el mismo comando y destino para continuar. También funciona después de cerrar y abrir el programa.
- No borres ni edites `.part` o `.resume.json` mientras quieras reanudar.
- Un `.lock` evita dos procesos del script sobre el mismo destino. Si hubo apagado abrupto y no hay otro proceso activo, elimina únicamente ese `.lock` para continuar.
- Si el servidor ignora Range o cambia el archivo, puede ser necesario reiniciar la transferencia; nunca se anexan esos bytes a ciegas.
- Errores permanentes, falta de espacio o metadatos incompatibles detienen el programa con un mensaje.
- Al terminar se verifica el CRC de todos los archivos del ZIP; solo entonces se publica el nombre final. Un ZIP existente no se sobrescribe.

## Velocidad e integridad

Transferencia binaria de 1 MiB por bloque, memoria acotada y progreso una vez por segundo. La velocidad depende de tu conexión y de Kaggle; no se garantiza un tiempo ni se acelera más allá del ancho de banda disponible.

La verificación CRC detecta corrupción accidental, pero no equivale a un SHA-256 oficial ni garantiza autenticidad. Un corte eléctrico puede afectar datos que todavía no llegaron al disco. Si el ZIP falla la verificación, conserva el archivo para revisarlo o elige otro destino para descargar desde cero.

## Pruebas locales

```powershell
python -m unittest discover -s .\descargaDataSet -p "test_*.py" -v
```

Las pruebas usan un servidor HTTP local y un ZIP pequeño; no descargan el dataset.
