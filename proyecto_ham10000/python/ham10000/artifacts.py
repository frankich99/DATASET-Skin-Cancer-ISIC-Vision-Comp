"""Verificación de evaluaciones persistidas antes de reutilizarlas; solo biblioteca estándar."""

import math
from pathlib import Path
import json

from . import CLASES
from .io import leer_csv


def verificar_evaluacion(target, checkpoint_sha, split_sha, split, condition, mitigation, expected_rows):
    """Rechaza carpetas incompletas, otra procedencia y predicciones truncadas."""
    target = Path(target)
    try:
        metrics = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
        predictions = leer_csv(target / "predictions.csv")
        expected = {r["image_id"]: r for r in expected_rows if r["split"] == split}
        provenance = dict(checkpoint_sha256=checkpoint_sha, split_sha256=split_sha,
                          split=split, condition=condition, mitigation=mitigation)
        if any(metrics.get(k, "none" if k == "mitigation" else None) != v for k, v in provenance.items()):
            raise ValueError("procedencia incompatible")
        ids = [r["image_id"] for r in predictions]
        if metrics["n"] != len(expected) or len(ids) != len(expected) or set(ids) != set(expected):
            raise ValueError("muestras ausentes, duplicadas o inesperadas")
        for row in predictions:
            original = expected[row["image_id"]]
            if any(row[k] != original[v] for k, v in (("truth", "dx"), ("lesion_id", "lesion_id"), ("group_id", "group_id"))):
                raise ValueError("etiqueta o agrupación incompatible")
            p = [float(row[f"p_{c}"]) for c in CLASES]
            if any(not math.isfinite(x) or not 0 <= x <= 1 for x in p) or abs(sum(p) - 1) > 1e-5:
                raise ValueError("probabilidades inválidas")
            if row["prediction"] != CLASES[max(range(len(p)), key=p.__getitem__)]:
                raise ValueError("predicción incompatible con probabilidades")
        return metrics
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Evaluación incompleta/incompatible: {target}. No se omite ni sobrescribe. "
                         f"Revísela o elija otra salida. Detalle: {error}") from error
