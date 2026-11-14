"""Auditoría de originales y split estratificado por grupos, sin copiar imágenes."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from . import CLASES
from .data import MEAN as IMAGENET_MEAN, STD as IMAGENET_STD, letterbox
from .io import cargar_split, guardar_csv, guardar_json, leer_csv, sha256


def asignar_splits(rows, seed=42):
    """Veinte pliegues disjuntos: 14 train, 3 val y 3 test (~70/15/15)."""
    y = np.array([CLASES.index(row["dx"]) for row in rows])
    groups = np.array([row["group_id"] for row in rows])
    folds = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    assignments = {}
    for fold, (_, indices) in enumerate(folds.split(np.zeros(len(rows)), y, groups)):
        name = "test" if fold < 3 else "val" if fold < 6 else "train"
        for index in indices:
            assignments[int(index)] = name
    result = [dict(row, split=assignments[i]) for i, row in enumerate(rows)]
    for name in ("train", "val", "test"):
        if {r["dx"] for r in result if r["split"] == name} != set(CLASES):
            raise ValueError(f"Faltan clases en {name}; revise el agrupamiento y la semilla.")
    for field in ("lesion_id", "group_id", "sha256"):
        owner = {}
        for row in result:
            if field not in row:
                continue
            key = row[field]
            if key in owner and owner[key] != row["split"]:
                raise AssertionError(f"Fuga detectada por {field}.")
            owner[key] = row["split"]
    return result


def preparar(data_root, output, seed=42):
    root = Path(data_root).resolve()
    output = Path(output)
    rows = sorted(leer_csv(root / "HAM10000_metadata.csv"), key=lambda r: r["image_id"])
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("El metadata contiene image_id repetidos.")
    if {r["dx"] for r in rows} != set(CLASES):
        raise ValueError("Se requieren las siete clases HAM10000, no el dataset ISIC de nueve clases.")
    index = {}
    for folder in ("HAM10000_images_part_1", "HAM10000_images_part_2"):
        for file in sorted((root / folder).glob("*.jpg")):
            if file.stem in index:
                raise ValueError(f"image_id duplicado entre carpetas: {file.stem}")
            index[file.stem] = file
    if set(index) != {r["image_id"] for r in rows}:
        raise ValueError("Las imágenes disponibles no coinciden con el metadata.")

    def inspect(row):
        file = index[row["image_id"]]
        with Image.open(file) as image:
            width, height = image.size
            image.load()
            image.verify()
        return dict(row, relative_path=file.relative_to(root).as_posix(),
                    width=width, height=height, sha256=sha256(file))

    with ThreadPoolExecutor(max_workers=4) as pool:
        audited = list(pool.map(inspect, rows))
    parent = {r["lesion_id"]: r["lesion_id"] for r in audited}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    duplicate_owner = {}
    duplicate_labels = {}
    for row in audited:
        digest, lesion = row["sha256"], row["lesion_id"]
        if digest in duplicate_labels and duplicate_labels[digest] != row["dx"]:
            raise ValueError("Imágenes idénticas con diagnósticos distintos; requiere revisión manual.")
        duplicate_labels[digest] = row["dx"]
        if digest in duplicate_owner:
            a, b = find(lesion), find(duplicate_owner[digest])
            parent[max(a, b)] = min(a, b)
        duplicate_owner[digest] = lesion
    for row in audited:
        row["group_id"] = find(row["lesion_id"])
    split = asignar_splits(audited, seed)
    csv_path = output / "split.csv"
    # No cambiar silenciosamente una partición que ya se utilizó para experimentar.
    if csv_path.exists():
        old = leer_csv(csv_path)
        if old != [{k: str(v) for k, v in r.items()} for r in split]:
            raise FileExistsError("El split existente difiere. Use otra carpeta de salida para un protocolo nuevo.")
    guardar_csv(csv_path, split)
    fingerprint = hashlib.sha256("\n".join(
        f"{r['image_id']}|{r['lesion_id']}|{r['dx']}|{r['sha256']}" for r in audited
    ).encode()).hexdigest()
    summary = {
        "images": len(rows), "lesions": len(parent),
        "groups": len({r["group_id"] for r in split}),
        "classes": dict(Counter(r["dx"] for r in rows)),
        "lesions_by_class": {c: len({r["lesion_id"] for r in rows if r["dx"] == c}) for c in CLASES},
        "split_counts": {s: dict(Counter(r["dx"] for r in split if r["split"] == s)) for s in ("train", "val", "test")},
        "exact_duplicate_excess": len(rows) - len(duplicate_owner),
        "split_sha256": sha256(csv_path), "dataset_sha256": fingerprint, "seed": seed,
        "method": "StratifiedGroupKFold(20); folds 0:3 test, 3:6 val, 6:20 train",
        "limitations": "Agrupamiento por lesión y duplicados binarios, no por paciente. Similitud perceptual no auditada exhaustivamente.",
    }
    guardar_json(output / "auditoria.json", summary)
    print(f"Auditoría completa: {len(rows)} imágenes; split {csv_path}", flush=True)
    return split, summary


def exportar_carpetas(data_root, split_path, output, size=224, quality=95):
    """Exporta train/val/test/<clase>/<image_id>.jpg, dataset_procesado.csv y estadisticas.json."""
    root = Path(data_root).resolve()
    output = Path(output).resolve()
    rows = cargar_split(split_path, root)
    output.mkdir(parents=True, exist_ok=True)
    (output / ".gitkeep").touch(exist_ok=True)
    for split_name in ("train", "val", "test"):
        for clase in CLASES:
            (output / split_name / clase).mkdir(parents=True, exist_ok=True)

    sum_rgb = np.zeros(3, dtype=np.float64)
    sum_sq_rgb = np.zeros(3, dtype=np.float64)
    train_pixels = 0

    def process(row):
        rel_out = f"{row['split']}/{row['dx']}/{row['image_id']}.jpg"
        dst = output / rel_out
        with Image.open(root / row["relative_path"]) as source:
            img = letterbox(source, size=size)
        arr = np.asarray(img, dtype=np.float32) / 255.0 if row["split"] == "train" else None
        tmp = dst.with_name(dst.name + ".tmp.jpg")
        img.save(tmp, format="JPEG", quality=quality)
        tmp.replace(dst)
        return dict(
            image_id=row["image_id"],
            lesion_id=row["lesion_id"],
            dx=row["dx"],
            group_id=row["group_id"],
            split=row["split"],
            aumentada=False,
            ruta_procesada=rel_out,
            ruta_original=row["relative_path"],
        ), arr

    manifest = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item, arr in pool.map(process, rows):
            manifest.append(item)
            if arr is not None:
                flat = arr.reshape(-1, 3).astype(np.float64)
                sum_rgb += flat.sum(axis=0)
                sum_sq_rgb += np.square(flat).sum(axis=0)
                train_pixels += flat.shape[0]

    mean = sum_rgb / train_pixels
    std = np.sqrt(np.maximum(0.0, (sum_sq_rgb / train_pixels) - np.square(mean)))
    train_rows = [r for r in rows if r["split"] == "train"]
    counts_train = Counter(r["dx"] for r in train_rows)
    weights = {
        c: round(len(train_rows) / (len(CLASES) * counts_train[c]), 4)
        for c in CLASES
    }
    manifest.sort(key=lambda r: (r["split"], r["dx"], r["image_id"]))
    guardar_csv(output / "dataset_procesado.csv", manifest)

    split_counts = {
        s: {c: sum(1 for r in rows if r["split"] == s and r["dx"] == c) for c in CLASES}
        for s in ("train", "val", "test")
    }
    stats = {
        "tamano_imagen": size,
        "calidad_jpeg": quality,
        "clases": list(CLASES),
        "media_rgb_train": [round(float(x), 6) for x in mean],
        "desviacion_rgb_train": [round(float(x), 6) for x in std],
        "normalizacion_modelo_imagenet": {
            "media": list(IMAGENET_MEAN),
            "desviacion": list(IMAGENET_STD),
        },
        "pesos_clase_train": weights,
        "conteos_split": {s: sum(split_counts[s].values()) for s in ("train", "val", "test")},
        "conteos_por_clase": split_counts,
        "split_sha256": sha256(split_path),
        "semilla": 42,
    }
    guardar_json(output / "estadisticas.json", stats)
    print(
        f"Exportadas {len(manifest)} imágenes a train/val/test y estadisticas.json en {output}",
        flush=True,
    )
    return manifest, stats
