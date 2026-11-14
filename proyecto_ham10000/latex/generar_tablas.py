"""Genera tablas LaTeX limpias desde auditoría, estadísticas y modelos reales."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "python"))

from ham10000 import CLASES, MODELOS
from ham10000.models import construir


def tabla(caption, columns, header, rows, total_row=False, width=r"\textwidth", size=r"\small", sep="6pt"):
    head = " & ".join(f"\\textbf{{{h}}}" for h in header)
    body = []
    for idx, row in enumerate(rows):
        formatted = " & ".join(map(str, row)) + " \\\\"
        if total_row and idx == len(rows) - 1:
            body.append("\\midrule\n" + formatted)
        else:
            body.append(formatted)
    return (
        f"\\begin{{table}}[htbp]\\centering{size}\n"
        f"\\setlength{{\\tabcolsep}}{{{sep}}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\begin{{tabularx}}{{{width}}}{{{columns}}}\\toprule\n"
        + head
        + " \\\\ \\midrule\n"
        + "\n".join(body)
        + "\n\\bottomrule\\end{tabularx}\\end{table}\n"
    )


def main():
    audit = json.loads((ROOT.parent / "data/processed/auditoria.json").read_text(encoding="utf-8"))
    stats_path = ROOT.parent / "data/processed/estadisticas.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else None
    out = ROOT / "tablas"
    out.mkdir(exist_ok=True)

    rows = [
        (
            f"\\texttt{{{c}}}",
            f"{audit['classes'][c]:,}".replace(",", r"\,"),
            f"{audit['lesions_by_class'][c]:,}".replace(",", r"\,"),
            f"{100 * audit['classes'][c] / audit['images']:.2f}\\%",
        )
        for c in CLASES
    ]
    rows.append((
        "\\textbf{Total}",
        f"\\textbf{{{audit['images']:,}}}".replace(",", r"\,"),
        f"\\textbf{{{audit['lesions']:,}}}".replace(",", r"\,"),
        "\\textbf{100.00\\%}",
    ))
    (out / "clases.tex").write_text(
        tabla(
            "Distribución verificada del conjunto HAM10000 por clase y lesión única.",
            r">{\raggedright\arraybackslash}Xrrr",
            ["Clase", "Imágenes", "Lesiones únicas", "Porcentaje"],
            rows,
            total_row=True,
            width=r"0.92\textwidth",
            size=r"\normalsize",
            sep="8pt",
        ),
        encoding="utf-8",
    )

    rows = [
        (
            f"\\texttt{{{c}}}",
            f"{audit['split_counts']['train'].get(c, 0):,}".replace(",", r"\,"),
            f"{audit['split_counts']['val'].get(c, 0):,}".replace(",", r"\,"),
            f"{audit['split_counts']['test'].get(c, 0):,}".replace(",", r"\,"),
            f"{stats['pesos_clase_train'][c]:.4f}" if stats else "---",
        )
        for c in CLASES
    ]
    rows.append((
        "\\textbf{Total}",
        f"\\textbf{{{sum(audit['split_counts']['train'].values()):,}}}".replace(",", r"\,"),
        f"\\textbf{{{sum(audit['split_counts']['val'].values()):,}}}".replace(",", r"\,"),
        f"\\textbf{{{sum(audit['split_counts']['test'].values()):,}}}".replace(",", r"\,"),
        "---",
    ))
    (out / "split.tex").write_text(
        tabla(
            "Conteos reales de la partición fija libre de fuga (\\textasciitilde 70/15/15) y pesos de clase en entrenamiento.",
            r">{\raggedright\arraybackslash}Xrrrr",
            ["Clase", "Train", "Val", "Test", "Peso $w_c$"],
            rows,
            total_row=True,
            width=r"\textwidth",
            size=r"\normalsize",
            sep="8pt",
        ),
        encoding="utf-8",
    )

    descriptions = {
        "resnet18": ("ResNet-18", "Bloques residuales $3\\times 3$", "147 MiB"),
        "efficientnet_b0": ("EfficientNet-B0", "MBConv + Squeeze-and-Excitation", "208 MiB"),
        "mobilenet_v3_small": ("MobileNetV3-Small", "Depthwise separable + SE + h-swish", "58 MiB"),
    }
    rows = []
    for name in MODELOS:
        model, _ = construir(name, pretrained=False)
        count = f"{sum(p.numel() for p in model.parameters()):,}".replace(",", r"\,")
        label, mech, vram = descriptions[name]
        rows.append((f"\\textbf{{{label}}}", mech, "\\texttt{IMAGENET1K\\_V1}", count, vram))
    (out / "modelos.tex").write_text(
        tabla(
            "Especificación técnica y parámetros exactos de las tres CNN adaptadas a siete clases (VRAM sintética con batch 2 a $224\\times 224$).",
            r"l>{\raggedright\arraybackslash}Xlrr",
            ["Arquitectura", "Mecanismo central", "Pesos base", "Parámetros", "VRAM sintética"],
            rows,
            width=r"\textwidth",
            size=r"\small",
            sep="6pt",
        ),
        encoding="utf-8",
    )
    print("Tablas generadas con datos reales.")


if __name__ == "__main__":
    main()
