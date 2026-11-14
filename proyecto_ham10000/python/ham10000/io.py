"""Lectura portable y escritura atómica de artefactos."""

import csv
import hashlib
import json
from pathlib import Path


def leer_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def guardar_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(rows[0])
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def guardar_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cargar_split(path, data_root):
    rows = leer_csv(path)
    required = {"image_id", "lesion_id", "dx", "group_id", "split", "relative_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("El CSV no tiene el esquema de partición requerido.")
    root = Path(data_root).resolve()
    ids = set()
    groups = {}
    for row in rows:
        if row["image_id"] in ids:
            raise ValueError("Identificador de imagen repetido en el split.")
        ids.add(row["image_id"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Partición desconocida.")
        group = row["group_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Fuga detectada: un grupo cruza particiones.")
        groups[group] = row["split"]
        file = (root / row["relative_path"]).resolve()
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError(f"Imagen ausente o ruta no permitida: {row['relative_path']}")
    return rows
