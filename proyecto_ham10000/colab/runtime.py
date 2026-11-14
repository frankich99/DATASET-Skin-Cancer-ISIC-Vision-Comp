"""Inicialización local/Colab sin dependencias. Integrada también en el notebook generado."""

from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile


def es_colab_alojado(environ=None):
    env = os.environ if environ is None else environ
    # No basta con que el paquete google.colab esté instalado en un Jupyter local.
    return bool(env.get("COLAB_RELEASE_TAG") or env.get("COLAB_BACKEND_VERSION"))


def localizar_proyecto(explicit=None, cwd=None):
    start = Path(cwd or Path.cwd()).resolve()
    candidates = [Path(explicit).expanduser()] if explicit else [
        candidate for parent in (start, *start.parents)
        for candidate in (parent, parent / "proyecto_ham10000")
    ]
    for candidate in candidates:
        if (candidate / "python/ejecutar.py").is_file() and (candidate / "configs/base.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError("No se encontró proyecto_ham10000. Indique RUTA_PROYECTO con la carpeta "
                            "que contiene python/, configs/, data/ y latex/.")


def localizar_datos(project, explicit=None):
    candidates = [Path(explicit).expanduser()] if explicit else [
        Path(project).parent / "descargaDataSet/HAM10000", Path(project) / "data/raw/HAM10000"
    ]
    for candidate in candidates:
        if (candidate / "HAM10000_metadata.csv").is_file():
            return candidate.resolve()
        matches = list(candidate.glob("*/HAM10000_metadata.csv")) if candidate.is_dir() else []
        if len(matches) == 1:
            return matches[0].parent.resolve()
    raise FileNotFoundError("No se encontró HAM10000_metadata.csv. Indique RUTA_DATOS con la raíz "
                            "de los originales; no use las carpetas exportadas train/val/test.")


def hash_archivo(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extraer_zip(source, destination):
    """Extrae a un destino nuevo y publica al terminar; reusa solo una extracción acreditada.

    Se permite repetir un nombre únicamente con CRC/tamaño idénticos (ZIP limpio original).
    No se aceptan rutas absolutas, escapes, enlaces, nombres Windows ambiguos ni colisiones.
    """
    source, destination = Path(source), Path(destination)
    digest = hash_archivo(source)
    marker = destination / ".extraccion_completa.json"
    if destination.exists():
        if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get("sha256") == digest:
            return destination
        raise FileExistsError(f"Destino existente no acreditado: {destination}. Elija otra carpeta; no se sobrescribe.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}_", dir=destination.parent))
    try:
        with zipfile.ZipFile(source) as archive:
            seen, entries = {}, []
            for item in archive.infolist():
                name = item.filename.replace("\\", "/")
                parts = PurePosixPath(name).parts
                if (not parts or name.startswith("/") or any(p in (".", "..") or ":" in p or
                        p.rstrip(" .") != p or p.split(".")[0].upper() in
                        {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                         *(f"LPT{i}" for i in range(1, 10))} for p in parts)):
                    raise ValueError(f"Ruta insegura en ZIP: {name}")
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError(f"Enlace no permitido: {name}")
                key = "/".join(parts).casefold()
                fingerprint = (item.CRC, item.file_size, item.is_dir())
                if key in seen:
                    if seen[key] != fingerprint:
                        raise ValueError(f"Colisión en ZIP: {name}")
                    continue
                seen[key] = fingerprint
                entries.append((item, parts))
            required = sum(item.file_size for item, _ in entries)
            if shutil.disk_usage(stage).free < required + 256 * 1024**2:
                raise OSError("Espacio insuficiente para extraer el ZIP y conservar un margen de 256 MiB.")
            for item, parts in entries:
                target = stage.joinpath(*parts)
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
        (stage / marker.name).write_text(json.dumps({"sha256": digest}), encoding="utf-8")
        if destination.exists():
            raise FileExistsError(destination)
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return destination
