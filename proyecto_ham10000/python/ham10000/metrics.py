"""Métricas multiclase explícitas; valores indefinidos se guardan como null."""

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from . import CLASES


def calcular(y, probabilities):
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities)
    if not len(y) or probabilities.shape != (len(y), len(CLASES)):
        raise ValueError("Dimensiones de etiquetas/probabilidades incompatibles.")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-4):
        raise ValueError("Probabilidades inválidas.")
    pred = probabilities.argmax(axis=1)
    labels = list(range(len(CLASES)))
    report = classification_report(y, pred, labels=labels, target_names=CLASES,
                                   zero_division=0, output_dict=True)
    all_present = set(y) == set(labels)
    auc = None
    if all_present:
        auc = float(roc_auc_score(y, probabilities, labels=labels, multi_class="ovr", average="macro"))
    auc_class = {}
    for i, name in enumerate(CLASES):
        binary = (y == i).astype(int)
        auc_class[name] = float(roc_auc_score(binary, probabilities[:, i])) if len(set(binary)) == 2 else None
    return {
        "n": len(y), "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(np.mean([report[c]["recall"] for c in CLASES])) if all_present else None,
        "melanoma_recall": report["mel"]["recall"] if CLASES.index("mel") in y else None,
        "macro_auc_ovr": auc, "auc_by_class": auc_class,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "missing_classes": [c for i, c in enumerate(CLASES) if i not in y],
    }
