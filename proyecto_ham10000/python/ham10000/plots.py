"""Figuras reales de auditoría y comparación, sin métricas simuladas."""

from collections import Counter, defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
from sklearn.metrics import auc, roc_curve

from . import CLASES, MODELOS
from .data import CONDITIONS, HAMDataset, MEAN, STD, letterbox, perturbar
from .io import cargar_split, guardar_csv, guardar_json, leer_csv

COLOR = "#1b4965"
COLOR_ALT = "#e07a5f"
COLOR_THIRD = "#2a9d8f"
COLOR_FOURTH = "#457b9d"
COLOR_LIGHT = "#94a3b8"
COLOR_TEXT = "#0f172a"
COLOR_CARD = "#edf4f8"
COLOR_CARD_ALT = "#f8fafc"
COLOR_CARD_WARM = "#fde8e1"
COLOR_BORDER = "#cbd5e1"
PALETTE_SPLIT = ("#1b4965", "#457b9d", "#2a9d8f")
HATCHES_SPLIT = (None, None, None)

_DX_NOMBRES = {
    "akiec": "Queratosis actínica /\ncarcinoma intraepitelial",
    "bcc": "Carcinoma basocelular",
    "bkl": "Lesión benigna tipo\nqueratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Nevo melanocítico",
    "vasc": "Lesión vascular",
}


def _estilizar(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")
    ax.tick_params(labelsize=10, colors=COLOR_TEXT)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.45, color=COLOR_LIGHT)
        ax.set_axisbelow(True)


def save(fig, root, name):
    fig.savefig(Path(root) / name, dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def figuras_dataset(data_root, split_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = cargar_split(split_path, data_root)
    total = len(rows)
    counts = Counter(r["dx"] for r in rows)
    train_counts = Counter(r["dx"] for r in rows if r["split"] == "train")
    lesions = {c: len({r["lesion_id"] for r in rows if r["dx"] == c}) for c in CLASES}

    # 01_clases.png
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    vals = [counts[c] for c in CLASES]
    bars = ax.bar(CLASES, vals, color=COLOR, width=0.62, edgecolor="#1e293b", linewidth=0.7)
    labels = [f"{v:,}\n({100 * v / total:.1f}%)".replace(",", " ") for v in vals]
    ax.bar_label(bars, labels=labels, padding=4, fontsize=9.5, fontweight="bold", color=COLOR_TEXT)
    ax.set_ylabel("Cantidad de imágenes", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.set_title("HAM10000: distribución real por diagnóstico (n = 10 015)", fontsize=13, fontweight="bold", pad=12, color=COLOR_TEXT)
    ax.set_ylim(0, max(vals) * 1.20)
    _estilizar(ax, "y")
    save(fig, output, "01_clases.png")

    # 02_lesiones.png
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(CLASES))
    v_img = [counts[c] for c in CLASES]
    v_les = [lesions[c] for c in CLASES]
    b1 = ax.bar(x - 0.19, v_img, 0.38, label="Imágenes JPEG (10 015)", color=COLOR, edgecolor="#1e293b", linewidth=0.7)
    b2 = ax.bar(x + 0.19, v_les, 0.38, label="Lesiones únicas (7 470)", color=COLOR_ALT, edgecolor="#1e293b", linewidth=0.7)
    ax.bar_label(b1, padding=3, fontsize=8.8, color=COLOR_TEXT)
    ax.bar_label(b2, padding=3, fontsize=8.8, color=COLOR_TEXT)
    ax.set_xticks(x, CLASES, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Cantidad", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.set_title("Una imagen no equivale a una lesión independiente", fontsize=13, fontweight="bold", pad=12, color=COLOR_TEXT)
    ax.set_ylim(0, max(v_img) * 1.16)
    ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=10, loc="upper left")
    _estilizar(ax, "y")
    save(fig, output, "02_lesiones.png")

    # 03_muestras.png (mosaico vertical 7x3, conserva RGB natural)
    fig, axes = plt.subplots(len(CLASES), 3, figsize=(10.5, 14.5), gridspec_kw={"width_ratios": [0.62, 1.0, 1.0]})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.945, bottom=0.015, wspace=0.10, hspace=0.26)
    examples = {}
    for row_idx, c in enumerate(CLASES):
        candidates = [r for r in rows if r["dx"] == c and r["split"] == "train"]
        examples[c] = candidates[0]

        ax_info = axes[row_idx, 0]
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        ax_info.axis("off")
        ax_info.add_patch(FancyBboxPatch(
            (0.03, 0.06), 0.94, 0.88,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            fc=COLOR_CARD_ALT, ec=COLOR_BORDER, lw=1.2,
        ))
        ax_info.text(0.5, 0.72, c, ha="center", va="center", fontsize=13.5, fontweight="bold", color=COLOR)
        ax_info.text(0.5, 0.44, _DX_NOMBRES.get(c, c), ha="center", va="center",
                     fontsize=9.4, color=COLOR_TEXT, linespacing=1.25)
        ax_info.text(0.5, 0.19, f"{train_counts[c]:,} imgs en train".replace(",", " "),
                     ha="center", va="center", fontsize=8.8, fontweight="bold", color=COLOR_FOURTH)

        for j in range(2):
            row = candidates[j]
            ax_img = axes[row_idx, j + 1]
            with Image.open(Path(data_root) / row["relative_path"]) as image:
                ax_img.imshow(image.convert("RGB"))
            ax_img.set_title(row["image_id"], fontsize=10, fontweight="bold", pad=5, color=COLOR_TEXT)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            for sp in ax_img.spines.values():
                sp.set_edgecolor("#475569")
                sp.set_linewidth(0.8)
    fig.suptitle("Ejemplos reales del subconjunto train por categoría (no constituyen diagnóstico clínico)",
                 fontsize=13.5, fontweight="bold", y=0.985, color=COLOR_TEXT)
    save(fig, output, "03_muestras.png")

    # 04_split.png
    fig, ax = plt.subplots(figsize=(11, 4.7))
    max_split_v = 0
    for j, (split, col, hatch) in enumerate(zip(("train", "val", "test"), PALETTE_SPLIT, HATCHES_SPLIT)):
        c_split = Counter(r["dx"] for r in rows if r["split"] == split)
        vals_s = [c_split[k] for k in CLASES]
        max_split_v = max(max_split_v, max(vals_s))
        bars_s = ax.bar(x + (j - 1) * 0.27, vals_s, 0.27, label=f"{split} (n = {sum(vals_s):,})".replace(",", " "),
                        color=col, edgecolor="#1e293b", linewidth=0.6, hatch=hatch)
        ax.bar_label(bars_s, padding=3, fontsize=8, rotation=90, color=COLOR_TEXT)
    ax.set_xticks(x, CLASES, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Cantidad de imágenes", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.set_title("Partición fija estratificada por grupos (sin cruce de lesión ni SHA-256)", fontsize=13, fontweight="bold", pad=12, color=COLOR_TEXT)
    ax.set_ylim(0, max_split_v * 1.22)
    ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=10, loc="upper left")
    _estilizar(ax, "y")
    save(fig, output, "04_split.png")

    # 05_multiplicidad.png
    multiplicity = Counter(Counter(r["lesion_id"] for r in rows).values())
    ks = list(sorted(multiplicity))
    mvals = [multiplicity[k] for k in ks]
    total_les = sum(mvals)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    bars = ax.bar([str(k) for k in ks], mvals, color=COLOR_FOURTH, width=0.58, edgecolor="#1e293b", linewidth=0.7)
    mlabels = [f"{v:,}\n({100 * v / total_les:.1f}%)".replace(",", " ") for v in mvals]
    ax.bar_label(bars, labels=mlabels, padding=4, fontsize=9.5, fontweight="bold", color=COLOR_TEXT)
    ax.set_xlabel("Número de fotografías asociadas a una misma lesión (lesion_id)", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.set_ylabel("Cantidad de lesiones únicas", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.set_title("Multiplicidad de tomas por lesión en HAM10000 (7 470 lesiones)", fontsize=13, fontweight="bold", pad=12, color=COLOR_TEXT)
    ax.set_ylim(0, max(mvals) * 1.22)
    _estilizar(ax, "y")
    save(fig, output, "05_multiplicidad.png")

    # 06_metadata.png
    ages = [float(r["age"]) for r in rows if r["age"]]
    sex = Counter(r["sex"] or "sin dato" for r in rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(wspace=0.22)
    axes[0].hist(ages, bins=np.arange(0, 91, 5), color=COLOR_ALT, edgecolor="white", linewidth=0.8)
    axes[0].set_xlabel("Edad registrada (años)", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    axes[0].set_ylabel("Cantidad de imágenes", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    axes[0].set_title(f"Distribución de edad ({len(rows)-len(ages)} ausentes)", fontsize=12, fontweight="bold", color=COLOR_TEXT)
    _estilizar(axes[0], "y")

    sex_keys = list(sex.keys())
    sex_vals = list(sex.values())
    sex_colors = [COLOR, COLOR_ALT, COLOR_FOURTH, COLOR_THIRD][:len(sex_keys)]
    sbars = axes[1].bar(sex_keys, sex_vals, color=sex_colors, width=0.55, edgecolor="#1e293b", linewidth=0.7)
    axes[1].bar_label(sbars, labels=[f"{v:,}".replace(",", " ") for v in sex_vals], padding=4, fontsize=10, fontweight="bold", color=COLOR_TEXT)
    axes[1].set_ylabel("Cantidad de imágenes", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    axes[1].set_title("Sexo registrado por imagen (no pacientes únicos)", fontsize=12, fontweight="bold", color=COLOR_TEXT)
    axes[1].set_ylim(0, max(sex_vals) * 1.15)
    _estilizar(axes[1], "y")
    save(fig, output, "06_metadata.png")

    # 07_localizacion.png
    loc = Counter(r["localization"] for r in rows)
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ordered = loc.most_common()
    loc_names = [a for a, _ in ordered][::-1]
    loc_vals = [b for _, b in ordered][::-1]
    hbars = ax.barh(loc_names, loc_vals, color=COLOR, height=0.72, edgecolor="#1e293b", linewidth=0.6)
    ax.bar_label(hbars, labels=[f" {v:,}".replace(",", " ") for v in loc_vals], padding=3, fontsize=9.2, color=COLOR_TEXT)
    ax.set_xlabel("Cantidad de imágenes", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.set_title("Distribución por localización anatómica registrada (15 zonas)", fontsize=13, fontweight="bold", pad=12, color=COLOR_TEXT)
    ax.set_xlim(0, max(loc_vals) * 1.14)
    _estilizar(ax, "x")
    save(fig, output, "07_localizacion.png")

    # 08_preprocesamiento.png (conserva RGB natural)
    sample = examples["mel"]
    with Image.open(Path(data_root) / sample["relative_path"]) as source:
        original = source.convert("RGB")
    resized = letterbox(original)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04, wspace=0.08)
    axes[0].imshow(original)
    axes[0].set_title(f"1. Original ({original.width}×{original.height})", fontsize=11.5, fontweight="bold", pad=8, color=COLOR_TEXT)
    axes[1].imshow(resized)
    axes[1].set_title("2. Letterbox isotrópico (224×224)", fontsize=11.5, fontweight="bold", pad=8, color=COLOR_TEXT)
    tensor = HAMDataset([sample], data_root, augment=True, seed=42)[0][0]
    visible = tensor.numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    axes[2].imshow(np.clip(visible, 0, 1))
    axes[2].set_title("3. Aumento determinista (solo train)", fontsize=11.5, fontweight="bold", pad=8, color=COLOR_TEXT)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#475569")
            sp.set_linewidth(0.8)
    save(fig, output, "08_preprocesamiento.png")

    # 09_robustez.png (conserva RGB natural)
    cond_titles = {
        "clean": "clean (referencia)",
        "noise_002": "noise_002 (σ = 0.02)",
        "noise_005": "noise_005 (σ = 0.05)",
        "blur_1": "blur_1 (r = 1 px)",
        "blur_2": "blur_2 (r = 2 px)",
    }
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 4.0))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.78, bottom=0.04, wspace=0.06)
    for ax, condition in zip(axes, CONDITIONS):
        ax.imshow(perturbar(resized, condition, sample["image_id"], 42))
        ax.set_title(cond_titles[condition], fontsize=11, fontweight="bold", pad=7, color=COLOR_TEXT)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#475569")
            sp.set_linewidth(0.8)
    fig.suptitle(f"Protocolo determinista de robustez ilustrado sobre {sample['image_id']} (train; no representa evaluación real)",
                 fontsize=12.5, fontweight="bold", y=0.95, color=COLOR_TEXT)
    save(fig, output, "09_robustez.png")

    # 10_pipeline.png
    fig, ax = plt.subplots(figsize=(13.5, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("1. Ingesta", "Originales HAM10000\n10 015 JPEG + CSV"),
        ("2. Auditoría", "Correspondencia, SHA-256\ny barrido RMSE < 4.5"),
        ("3. Split fijo", "StratifiedGroupKFold\n70/15/15 · 7 470 grupos"),
        ("4. Entrenamiento", "3 CNN (IMAGENET1K_V1)\nReceta base.json (seed 42)"),
        ("5. Evaluación", "Macro F1, Bal. Acc., AUC\n+ 5 condiciones de ruido"),
    ]
    box_w, box_h = 0.162, 0.45
    centers = np.linspace(0.095, 0.905, len(stages))
    for i, (head, body) in enumerate(stages):
        cx = centers[i]
        y0 = 0.39
        ax.add_patch(FancyBboxPatch((cx - box_w / 2, y0), box_w, box_h,
                                    boxstyle="round,pad=0.01,rounding_size=0.025",
                                    fc=COLOR_CARD, ec=COLOR, lw=1.5))
        ax.add_patch(FancyBboxPatch((cx - box_w / 2, y0 + box_h - 0.13), box_w, 0.13,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    fc=COLOR, ec=COLOR, lw=1.0))
        ax.text(cx, y0 + box_h - 0.065, head, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white")
        ax.text(cx, y0 + (box_h - 0.13) / 2, body, ha="center", va="center",
                fontsize=9.1, color=COLOR_TEXT, linespacing=1.35)
        if i < len(stages) - 1:
            x_start = cx + box_w / 2 + 0.014
            x_end = centers[i + 1] - box_w / 2 - 0.014
            ax.annotate("", xy=(x_end, y0 + box_h / 2), xytext=(x_start, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.5", color=COLOR_FOURTH, lw=2.0))
    ax.add_patch(FancyBboxPatch((0.018, 0.05), 0.964, 0.21,
                                boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=COLOR_CARD_WARM, ec=COLOR_ALT, lw=1.3, ls="--"))
    ax.text(0.5, 0.155,
            "Controles transversales:  0 fugas de lesión ni SHA-256  ·  Lectura directa de JPEG originales\n"
            "Test protegido (--confirmar-test)  ·  Mismo código y configuración Laptop Windows / Google Colab",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color=COLOR_TEXT, linespacing=1.45)
    save(fig, output, "10_pipeline.png")

    # 11_arquitecturas.png
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.09, wspace=0.14)
    specs = [
        (
            "ResNet-18",
            "11 180 103 parámetros · 147 MiB*",
            COLOR,
            [
                ("Entrada RGB 224×224×3", COLOR_CARD_ALT, COLOR_FOURTH),
                ("Stem: Conv 7×7 + BN + MaxPool", COLOR_CARD, COLOR_FOURTH),
                ("4 Etapas Residuales (bloques 3×3)", COLOR_CARD, COLOR_FOURTH),
                ("Global AvgPool → vector 512-d", COLOR_CARD, COLOR_FOURTH),
                ("Cabeza adaptada: Linear(512 → 7)", COLOR_CARD_WARM, COLOR_ALT),
            ],
        ),
        (
            "EfficientNet-B0",
            "4 016 515 parámetros · 208 MiB*",
            COLOR_ALT,
            [
                ("Entrada RGB 224×224×3", COLOR_CARD_ALT, COLOR_FOURTH),
                ("Stem: Conv 3×3 + BN + SiLU", COLOR_CARD, COLOR_FOURTH),
                ("Bloques MBConv + Atención SE", COLOR_CARD, COLOR_FOURTH),
                ("Global AvgPool + Dropout(0.2) → 1280-d", COLOR_CARD, COLOR_FOURTH),
                ("Cabeza adaptada: Linear(1280 → 7)", COLOR_CARD_WARM, COLOR_ALT),
            ],
        ),
        (
            "MobileNetV3-Small",
            "1 525 031 parámetros · 58 MiB*",
            COLOR_THIRD,
            [
                ("Entrada RGB 224×224×3", COLOR_CARD_ALT, COLOR_FOURTH),
                ("Stem: Conv 3×3 + h-swish", COLOR_CARD, COLOR_FOURTH),
                ("Cuellos invertidos DW-Sep + SE", COLOR_CARD, COLOR_FOURTH),
                ("Pool → Linear(576→1024) + h-swish + Drop", COLOR_CARD, COLOR_FOURTH),
                ("Cabeza adaptada: Linear(1024 → 7)", COLOR_CARD_WARM, COLOR_ALT),
            ],
        ),
    ]
    for ax, (title, subtitle, accent, blocks) in zip(axes, specs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.03, 0.02), 0.94, 0.95,
                                    boxstyle="round,pad=0.015,rounding_size=0.03",
                                    fc="white", ec=COLOR_BORDER, lw=1.3))
        ax.text(0.5, 0.92, title, ha="center", va="center", fontsize=12.5, fontweight="bold", color=accent)
        ax.text(0.5, 0.855, subtitle, ha="center", va="center", fontsize=9.3, color=COLOR_FOURTH, fontweight="bold")
        ys = np.linspace(0.73, 0.11, len(blocks))
        for j, (label, fc, ec) in enumerate(blocks):
            is_last = j == len(blocks) - 1
            ax.add_patch(FancyBboxPatch((0.08, ys[j] - 0.052), 0.84, 0.104,
                                        boxstyle="round,pad=0.01,rounding_size=0.022",
                                        fc=fc, ec=ec, lw=1.7 if is_last else 1.2))
            ax.text(0.5, ys[j], label, ha="center", va="center", fontsize=9.2,
                    fontweight="bold" if is_last else "normal", color=COLOR_TEXT)
            if j < len(blocks) - 1:
                ax.annotate("", xy=(0.5, ys[j + 1] + 0.058), xytext=(0.5, ys[j] - 0.058),
                            arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.4", color=COLOR, lw=1.4))
    fig.suptitle("Esquema estructural comparativo de las tres CNN adaptadas (pesos base IMAGENET1K_V1)",
                 y=0.95, fontsize=13.5, fontweight="bold", color=COLOR_TEXT)
    fig.text(0.5, 0.025, "*VRAM asignada medida en prueba sintética GPU (batch 2, FP32, 224×224). Diagrama conceptual de etapas.",
             ha="center", va="center", fontsize=9.3, color=COLOR_FOURTH, style="italic")
    save(fig, output, "11_arquitecturas.png")

    guardar_json(output / "figuras_manifest.json", {
        "source_split": str(split_path), "images": len(rows), "classes": dict(counts),
        "selected_train_examples": {c: r["image_id"] for c, r in examples.items()},
        "note": "Figuras descriptivas reales; sin curvas ni métricas de entrenamiento inventadas."})
    print(f"Once figuras guardadas en {output}", flush=True)


_MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_small": "MobileNetV3-Small",
}
_MODEL_COLORS = {
    "resnet18": "#1b4965",
    "efficientnet_b0": "#e07a5f",
    "mobilenet_v3_small": "#2a9d8f",
}
_MODEL_BAR_COLORS = {
    "resnet18": "#1b4965",
    "efficientnet_b0": "#e07a5f",
    "mobilenet_v3_small": "#2a9d8f",
}
_MODEL_HATCHES = {
    "resnet18": None,
    "efficientnet_b0": None,
    "mobilenet_v3_small": None,
}
_MODEL_MARKERS = {
    "resnet18": "o",
    "efficientnet_b0": "s",
    "mobilenet_v3_small": "^",
}
_EXTRA_COLORS = ("#457b9d", "#d97706", "#7c3aed", "#c1121f", "#0f766e")
_EXTRA_BAR_COLORS = ("#457b9d", "#d97706", "#7c3aed", "#c1121f", "#0f766e")
_EXTRA_HATCHES = (None, None, None, None, None)
_EXTRA_MARKERS = ("D", "v", "P", "X", "*")
_LINE_STYLES = ("-", "--", "-.", (0, (3, 1, 1, 1)))
_CLASS_COLORS = ("#1b4965", "#e07a5f", "#2a9d8f", "#d97706", "#c1121f", "#457b9d", "#7c3aed")
_CLASS_STYLES = (
    (_CLASS_COLORS[0], "--", "o"),
    (_CLASS_COLORS[1], "-.", "s"),
    (_CLASS_COLORS[2], ":", "^"),
    (_CLASS_COLORS[3], (0, (5, 1.5)), "D"),
    (_CLASS_COLORS[4], "-", "v"),
    (_CLASS_COLORS[5], (0, (4, 1.5, 1, 1.5, 1, 1.5)), "P"),
    (_CLASS_COLORS[6], (0, (1.5, 1.5)), "X"),
)


def _ordered_models(models):
    base = {m: i for i, m in enumerate(MODELOS)}
    return sorted(set(models), key=lambda m: (0, base[m], m) if m in base else (1, 0, m))


def _model_color(model, idx):
    return _MODEL_COLORS.get(model, _EXTRA_COLORS[idx % len(_EXTRA_COLORS)])


def _model_bar_color(model, idx):
    return _MODEL_BAR_COLORS.get(model, _EXTRA_BAR_COLORS[idx % len(_EXTRA_BAR_COLORS)])


def _model_hatch(model, idx):
    return _MODEL_HATCHES[model] if model in _MODEL_HATCHES else _EXTRA_HATCHES[idx % len(_EXTRA_HATCHES)]


def _model_marker(model, idx):
    return _MODEL_MARKERS.get(model, _EXTRA_MARKERS[idx % len(_EXTRA_MARKERS)])


def _sufijo_mitigacion(mitigation):
    return f"_{mitigation}" if mitigation != "none" else ""


def _etiqueta_condicion(condition, mitigation):
    return f"{condition}+{mitigation}" if mitigation != "none" else condition


def _orden_condicion(condition, mitigation):
    base_order = {c: i for i, c in enumerate(CONDITIONS)}
    tier = (0 if condition in base_order else 1) if mitigation == "none" else 2
    return (tier, base_order.get(condition, len(base_order)), condition, mitigation)


def _calcular_roc_predicciones(pred_path):
    pred_rows = leer_csv(pred_path)
    if not pred_rows:
        return None
    fpr_grid = np.linspace(0.0, 1.0, 101)
    truths = [row["truth"] for row in pred_rows]
    class_curves = {}
    tprs = []
    omitted = []
    for c in CLASES:
        y_bin = np.array([1 if t == c else 0 for t in truths], dtype=int)
        n_pos = int(y_bin.sum())
        n_neg = len(y_bin) - n_pos
        if n_pos > 0 and n_neg > 0:
            y_score = np.array([float(row[f"p_{c}"]) for row in pred_rows], dtype=float)
            fpr, tpr, _ = roc_curve(y_bin, y_score)
            roc_auc = float(auc(fpr, tpr))
            interp_tpr = np.interp(fpr_grid, fpr, tpr)
            interp_tpr[0] = 0.0
            class_curves[c] = (fpr, tpr, roc_auc)
            tprs.append(interp_tpr)
        else:
            omitted.append(c)
    if len(class_curves) == len(CLASES) and tprs:
        macro_tpr = np.mean(tprs, axis=0)
        macro_tpr[0] = 0.0
        macro_tpr[-1] = 1.0
        macro_auc = float(np.mean([v[2] for v in class_curves.values()]))
    else:
        macro_tpr = None
        macro_auc = None
    return {
        "fpr_grid": fpr_grid,
        "class_curves": class_curves,
        "omitted": omitted,
        "macro_tpr": macro_tpr,
        "macro_auc": macro_auc,
        "n": len(pred_rows),
    }


def _graficar_matriz_confusion(r, mitigation, output):
    suffix = _sufijo_mitigacion(mitigation)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    matrix = np.array(r["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    max_val = matrix.max() if matrix.size else 0
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            ax.text(
                j, i, str(matrix[i, j]), ha="center", va="center", fontsize=10,
                fontweight="bold" if i == j else "normal",
                color="white" if matrix[i, j] > max_val * 0.5 else COLOR_TEXT,
            )
    ax.set_xticks(range(len(CLASES)), CLASES, rotation=45, ha="right", fontsize=10.5)
    ax.set_yticks(range(len(CLASES)), CLASES, fontsize=10.5)
    for sp in ax.spines.values():
        sp.set_color("#475569")
    title_mit = f" | {mitigation}" if mitigation != "none" else ""
    model_lbl = _MODEL_LABELS.get(r["model"], r["model"])
    ax.set_xlabel("Predicción", fontsize=11.5, fontweight="bold", color=COLOR_TEXT)
    ax.set_ylabel("Etiqueta", fontsize=11.5, fontweight="bold", color=COLOR_TEXT)
    ax.set_title(
        f"{model_lbl} | {r['split']} | {r['condition']}{title_mit}",
        fontsize=12, fontweight="bold", pad=10, color=COLOR_TEXT,
    )
    save(fig, output, f"confusion_{r['model']}_{r['seed']}_{r['split']}_{r['condition']}{suffix}.png")


def _graficar_roc_individual(roc_data, r, mitigation, output):
    suffix = _sufijo_mitigacion(mitigation)
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    for i, c in enumerate(CLASES):
        if c in roc_data["class_curves"]:
            fpr, tpr, roc_auc = roc_data["class_curves"][c]
            col, ls, marker = _CLASS_STYLES[i % len(_CLASS_STYLES)]
            is_mel = c == "mel"
            ax.plot(
                fpr, tpr,
                color=col,
                ls=ls,
                marker=marker,
                markevery=0.15,
                markersize=5.2 if is_mel else 4.8,
                markeredgecolor=COLOR_TEXT,
                markeredgewidth=0.5,
                lw=2.1 if is_mel else 1.65,
                alpha=0.95 if is_mel else 0.90,
                zorder=9 if is_mel else 5,
                label=f"{c} (AUC = {roc_auc:.3f})",
            )
    if roc_data["macro_tpr"] is not None:
        ax.plot(
            roc_data["fpr_grid"], roc_data["macro_tpr"],
            color="#0f172a", lw=2.6, ls="-", zorder=10,
            label=f"Macro-promedio OVR (AUC = {roc_data['macro_auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], color=COLOR_LIGHT, lw=1.2, ls=":", label="Azar (AUC = 0.500)")
    if roc_data["omitted"]:
        omitted_str = ", ".join(roc_data["omitted"])
        ax.plot([], [], " ", label=f"Omitidas (sin AUC): {omitted_str}")
        if roc_data["macro_tpr"] is None:
            ax.plot([], [], " ", label="Macro OVR: N/D (clases ausentes)")
        if not roc_data["class_curves"]:
            ax.text(
                0.5, 0.52,
                f"Sin curvas OVR válidas\n(clases omitidas: {omitted_str})",
                ha="center", va="center", fontsize=9.5, color=COLOR_ALT,
                bbox=dict(boxstyle="round,pad=0.35", fc=COLOR_CARD, ec=COLOR_BORDER),
            )
    model_lbl = _MODEL_LABELS.get(r["model"], r["model"])
    title_mit = f" | {mitigation}" if mitigation != "none" else ""
    ax.set_title(
        f"Curvas ROC OVR · {model_lbl} | {r['split']} | {r['condition']}{title_mit}",
        fontsize=12, fontweight="bold", pad=10, color=COLOR_TEXT,
    )
    ax.set_xlabel("Tasa de falsos positivos (FPR)", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    _estilizar(ax, "both")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=9.2)
    save(fig, output, f"roc_{r['model']}_{r['seed']}_{r['split']}_{r['condition']}{suffix}.png")


def _graficar_roc_comparativa(roc_groups, output):
    for (split, condition, mitigation), items in sorted(roc_groups.items()):
        if not items:
            continue
        model_order = {m: i for i, m in enumerate(_ordered_models([rec["model"] for rec, _ in items]))}
        items_sorted = sorted(items, key=lambda item: (model_order[item[0]["model"]], str(item[0]["seed"])))
        model_counts = Counter(rec["model"] for rec, _ in items_sorted)
        seen_per_model = Counter()

        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
        fig.subplots_adjust(wspace=0.22, top=0.84)
        plotted_macro = 0
        plotted_mel = 0

        for rec, roc_data in items_sorted:
            m = rec["model"]
            m_idx = model_order[m]
            seed_idx = seen_per_model[m]
            seen_per_model[m] += 1
            col = _model_color(m, m_idx)
            ls = _LINE_STYLES[(m_idx + seed_idx) % len(_LINE_STYLES)]
            marker = _model_marker(m, m_idx + seed_idx)
            base_lbl = _MODEL_LABELS.get(m, m)
            lbl = f"{base_lbl} (seed {rec['seed']})" if model_counts[m] > 1 else base_lbl

            if roc_data["macro_tpr"] is not None:
                axes[0].plot(
                    roc_data["fpr_grid"], roc_data["macro_tpr"],
                    color=col, ls=ls, marker=marker, markevery=0.15, markersize=5.0,
                    markeredgecolor=COLOR_TEXT, markeredgewidth=0.5, lw=2.1,
                    label=f"{lbl} (AUC = {roc_data['macro_auc']:.4f})",
                )
                plotted_macro += 1
            else:
                axes[0].plot([], [], " ", label=f"{lbl}: N/D (faltan clases)")

            if "mel" in roc_data["class_curves"]:
                fpr_m, tpr_m, auc_m = roc_data["class_curves"]["mel"]
                axes[1].plot(
                    fpr_m, tpr_m,
                    color=col, ls=ls, marker=marker, markevery=0.15, markersize=5.0,
                    markeredgecolor=COLOR_TEXT, markeredgewidth=0.5, lw=2.1,
                    label=f"{lbl} (AUC = {auc_m:.4f})",
                )
                plotted_mel += 1
            else:
                axes[1].plot([], [], " ", label=f"{lbl}: mel N/D")

        axes[0].plot([0, 1], [0, 1], color=COLOR_LIGHT, lw=1.2, ls=":", label="Azar (AUC = 0.5000)")
        axes[1].plot([0, 1], [0, 1], color=COLOR_LIGHT, lw=1.2, ls=":", label="Azar (AUC = 0.5000)")
        if plotted_macro == 0:
            axes[0].text(
                0.5, 0.52, "Macro-ROC no disponible\n(faltan clases en y_true)",
                ha="center", va="center", fontsize=9.5, color=COLOR_ALT,
                bbox=dict(boxstyle="round,pad=0.35", fc=COLOR_CARD, ec=COLOR_BORDER),
            )
        if plotted_mel == 0:
            axes[1].text(
                0.5, 0.52, "ROC de melanoma (mel) no disponible\n(clase ausente o sin negativos)",
                ha="center", va="center", fontsize=9.5, color=COLOR_ALT,
                bbox=dict(boxstyle="round,pad=0.35", fc=COLOR_CARD, ec=COLOR_BORDER),
            )

        for ax, sub_title in zip(axes, ("Macro-ROC promedio (OVR)", "ROC específica de melanoma (mel)")):
            ax.set_xlabel("Tasa de falsos positivos (FPR)", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
            ax.set_ylabel("Tasa de verdaderos positivos (TPR)", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
            ax.set_title(sub_title, fontsize=11.5, fontweight="bold", pad=8, color=COLOR_TEXT)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            _estilizar(ax, "both")
            ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=8.8)

        suffix = _sufijo_mitigacion(mitigation)
        title_mit = f" | {mitigation}" if mitigation != "none" else ""
        fig.suptitle(
            f"Comparación de curvas ROC · {split} | {condition}{title_mit}",
            fontsize=13, fontweight="bold", y=0.96, color=COLOR_TEXT,
        )
        save(fig, output, f"roc_comparativa_{split}_{condition}{suffix}.png")


def _graficar_curvas_historia(runs, output):
    for history in sorted(Path(runs).rglob("history.csv")):
        rows = leer_csv(history)
        if not rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
        fig.subplots_adjust(wspace=0.22, top=0.84)
        epochs = [int(r["epoch"]) for r in rows]
        axes[0].plot(
            epochs, [float(r["train_loss"]) for r in rows],
            color="#1b4965", ls="-", marker="o", markersize=4.5, lw=2.0, label="Train ponderada",
        )
        axes[0].plot(
            epochs, [float(r["val_loss"]) for r in rows],
            color="#e07a5f", ls="--", marker="s", markersize=4.5, lw=2.0, label="Val no ponderada",
        )
        axes[0].set_xlabel("Época", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
        axes[0].set_ylabel("Pérdida", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
        axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0].set_title("Evolución de la pérdida", fontsize=11, fontweight="bold", pad=8, color=COLOR_TEXT)
        _estilizar(axes[0], "y")
        axes[0].legend(frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=9)

        axes[1].plot(
            epochs, [float(r["val_macro_f1"]) for r in rows],
            color="#2a9d8f", ls="-", marker="o", markersize=4.5, lw=2.0, label="Val Macro F1",
        )
        if all(r.get("val_accuracy") not in (None, "") for r in rows):
            axes[1].plot(
                epochs, [float(r["val_accuracy"]) for r in rows],
                color="#457b9d", ls="--", marker="^", markersize=4.5, lw=1.9, label="Val Accuracy",
            )
        axes[1].set_xlabel("Época", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
        axes[1].set_ylabel("Métrica de validación", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
        axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes[1].set_ylim(0, 1.03)
        axes[1].set_title("Desempeño por época", fontsize=11, fontweight="bold", pad=8, color=COLOR_TEXT)
        _estilizar(axes[1], "y")
        axes[1].legend(frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=9, loc="lower right")

        fig.suptitle(f"Curvas de entrenamiento · {history.parent.name}", fontsize=12.5, fontweight="bold", y=0.96, color=COLOR_TEXT)
        save(fig, output, f"curvas_{history.parent.name}.png")


def _mean_or_none(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def _graficar_comparacion_metricas(records, output):
    clean_none = [r for r in records if r["condition"] == "clean" and r["mitigation"] == "none"]
    if not clean_none:
        return
    available_splits = {r["split"] for r in clean_none}
    if "test" in available_splits:
        main_split = "test"
    elif "val" in available_splits:
        main_split = "val"
    else:
        main_split = sorted(available_splits)[0]
    main_rows = [r for r in clean_none if r["split"] == main_split]
    models = _ordered_models([r["model"] for r in main_rows])

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), gridspec_kw={"width_ratios": [1.65, 1.0]})
    fig.subplots_adjust(wspace=0.25, top=0.77, bottom=0.14)

    metrics_spec = (
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Prec."),
        ("macro_f1", "Macro F1"),
        ("balanced_accuracy", "Bal. Acc."),
        ("melanoma_recall", "Recall mel"),
        ("macro_auc_ovr", "Macro AUC"),
    )
    x = np.arange(len(metrics_spec))
    n_models = len(models)
    bar_w = min(0.24, 0.78 / max(1, n_models))
    for idx, m in enumerate(models):
        m_rows = [r for r in main_rows if r["model"] == m]
        vals = [_mean_or_none(m_rows, k) for k, _ in metrics_spec]
        heights = [v if v is not None else 0.0 for v in vals]
        offset = (idx - (n_models - 1) / 2) * bar_w
        bars = axes[0].bar(
            x + offset, heights, bar_w,
            label=_MODEL_LABELS.get(m, m),
            color=_model_bar_color(m, idx),
            hatch=_model_hatch(m, idx),
            edgecolor="#1e293b", linewidth=0.6,
        )
        labels = [f"{v:.3f}" if v is not None else "N/D" for v in vals]
        axes[0].bar_label(
            bars, labels=labels, padding=3, fontsize=7.6,
            rotation=90 if n_models > 2 else 0, color=COLOR_TEXT,
        )
    axes[0].set_xticks(x, [lbl for _, lbl in metrics_spec], fontsize=9.4, fontweight="bold", rotation=10, ha="center")
    axes[0].set_ylabel("Puntuación", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
    axes[0].set_ylim(0, 1.18)
    axes[0].set_title(f"Desempeño predictivo ({main_split}, clean)", fontsize=11.5, fontweight="bold", pad=28, color=COLOR_TEXT)
    _estilizar(axes[0], "y")
    axes[0].legend(
        frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=8.6,
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=min(3, max(1, n_models)),
    )

    lat_vals = {m: _mean_or_none([r for r in main_rows if r["model"] == m], "latency_median_ms") for m in models}
    vram_vals = {
        m: _mean_or_none([r for r in main_rows if r["model"] == m], "peak_training_allocated_mib")
        for m in models
    }
    has_lat = any(v is not None for v in lat_vals.values())
    has_vram = any(v is not None for v in vram_vals.values())
    xm = np.arange(len(models))
    short_labels = [_MODEL_LABELS.get(m, m) for m in models]

    if has_lat and has_vram:
        ax_vram = axes[1].twinx()
        w = 0.34
        lat_h = [lat_vals[m] if lat_vals[m] is not None else 0.0 for m in models]
        vram_h = [vram_vals[m] if vram_vals[m] is not None else 0.0 for m in models]
        b1 = axes[1].bar(
            xm - w / 2, lat_h, w, label="Latencia mediana (ms)",
            color="#1b4965", edgecolor="#1e293b", linewidth=0.6,
        )
        b2 = ax_vram.bar(
            xm + w / 2, vram_h, w, label="VRAM pico (MiB)",
            color="#e07a5f", edgecolor="#1e293b", linewidth=0.6,
        )
        axes[1].bar_label(
            b1, labels=[f"{lat_vals[m]:.2f}" if lat_vals[m] is not None else "N/D" for m in models],
            padding=3, fontsize=8.2, color=COLOR_TEXT,
        )
        ax_vram.bar_label(
            b2, labels=[f"{vram_vals[m]:.0f}" if vram_vals[m] is not None else "N/D" for m in models],
            padding=3, fontsize=8.2, color=COLOR_TEXT,
        )
        axes[1].set_ylabel("Latencia mediana (ms)", fontsize=10, fontweight="bold", color="#1b4965")
        ax_vram.set_ylabel("VRAM pico (MiB)", fontsize=10, fontweight="bold", color="#e07a5f")
        axes[1].set_ylim(0, (max(lat_h) if max(lat_h) > 0 else 1.0) * 1.18)
        ax_vram.set_ylim(0, (max(vram_h) if max(vram_h) > 0 else 1.0) * 1.18)
        _estilizar(axes[1], "y")
        ax_vram.spines["top"].set_visible(False)
        ax_vram.spines["left"].set_visible(False)
        ax_vram.spines["right"].set_color("#e07a5f")
        ax_vram.tick_params(labelsize=9.5, colors=COLOR_TEXT)
        h1, l1 = axes[1].get_legend_handles_labels()
        h2, l2 = ax_vram.get_legend_handles_labels()
        axes[1].legend(
            h1 + h2, l1 + l2, frameon=True, facecolor="white",
            edgecolor=COLOR_BORDER, fontsize=8.5, loc="lower center",
            bbox_to_anchor=(0.5, 1.01), ncol=2,
        )
        axes[1].set_title("Costo computacional (latencia y VRAM)", fontsize=11.5, fontweight="bold", pad=28, color=COLOR_TEXT)
    elif has_lat:
        lat_h = [lat_vals[m] if lat_vals[m] is not None else 0.0 for m in models]
        cols = [_model_bar_color(m, idx) for idx, m in enumerate(models)]
        b1 = axes[1].bar(xm, lat_h, 0.5, color=cols, edgecolor="#1e293b", linewidth=0.6)
        for idx, (bar, m) in enumerate(zip(b1, models)):
            h = _model_hatch(m, idx)
            if h:
                bar.set_hatch(h)
        axes[1].bar_label(
            b1, labels=[f"{lat_vals[m]:.2f}" if lat_vals[m] is not None else "N/D" for m in models],
            padding=3, fontsize=8.5, color=COLOR_TEXT,
        )
        axes[1].set_ylim(0, (max(lat_h) if max(lat_h) > 0 else 1.0) * 1.22)
        axes[1].set_ylabel("Latencia mediana (ms)", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
        axes[1].set_title("Latencia mediana (VRAM pico: N/D)", fontsize=11.5, fontweight="bold", pad=28, color=COLOR_TEXT)
        _estilizar(axes[1], "y")
    elif has_vram:
        vram_h = [vram_vals[m] if vram_vals[m] is not None else 0.0 for m in models]
        cols = [_model_bar_color(m, idx) for idx, m in enumerate(models)]
        b1 = axes[1].bar(xm, vram_h, 0.5, color=cols, edgecolor="#1e293b", linewidth=0.6)
        for idx, (bar, m) in enumerate(zip(b1, models)):
            h = _model_hatch(m, idx)
            if h:
                bar.set_hatch(h)
        axes[1].bar_label(
            b1, labels=[f"{vram_vals[m]:.0f}" if vram_vals[m] is not None else "N/D" for m in models],
            padding=3, fontsize=8.5, color=COLOR_TEXT,
        )
        axes[1].set_ylim(0, (max(vram_h) if max(vram_h) > 0 else 1.0) * 1.22)
        axes[1].set_ylabel("VRAM pico (MiB)", fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
        axes[1].set_title("VRAM pico (Latencia: N/D)", fontsize=11.5, fontweight="bold", pad=28, color=COLOR_TEXT)
        _estilizar(axes[1], "y")
    else:
        axes[1].text(0.5, 0.5, "Sin datos de latencia ni VRAM", ha="center", va="center", fontsize=10, color=COLOR_ALT)
        axes[1].set_xticks([])
        _estilizar(axes[1], None)

    if has_lat or has_vram:
        axes[1].set_xticks(xm, short_labels, fontsize=9.2, fontweight="bold", rotation=12 if len(models) > 2 else 0)

    fig.suptitle(
        f"Comparación global de modelos · partición {main_split} (clean)",
        fontsize=13, fontweight="bold", y=0.97, color=COLOR_TEXT,
    )
    save(fig, output, "comparacion_metricas.png")


def _graficar_comparacion_robustez(records, output):
    by_split = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)

    qualifying_splits = []
    for split, srows in by_split.items():
        per_model = defaultdict(set)
        for r in srows:
            per_model[r["model"]].add((r["condition"], r["mitigation"]))
        if any(len(conds) > 1 for conds in per_model.values()):
            qualifying_splits.append(split)

    if not qualifying_splits:
        return
    if "test" in qualifying_splits:
        rob_split = "test"
    elif "val" in qualifying_splits:
        rob_split = "val"
    else:
        rob_split = sorted(qualifying_splits)[0]

    rob_rows = by_split[rob_split]
    ordered_pairs = sorted(
        {(r["condition"], r["mitigation"]) for r in rob_rows},
        key=lambda pair: _orden_condicion(pair[0], pair[1]),
    )
    cond_labels = [_etiqueta_condicion(cond, mit) for cond, mit in ordered_pairs]
    x = np.arange(len(ordered_pairs))
    models = _ordered_models([r["model"] for r in rob_rows])

    unmit_indices = [i for i, (_, mit) in enumerate(ordered_pairs) if mit == "none"]
    mit_indices = [i for i, (_, mit) in enumerate(ordered_pairs) if mit != "none"]
    sep_x = (max(unmit_indices) + 0.5) if (unmit_indices and mit_indices) else None

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fig.subplots_adjust(wspace=0.22, top=0.84, bottom=0.20)
    panels = (
        ("macro_f1", "Macro F1", "Evolución de Macro F1"),
        ("melanoma_recall", "Recall de melanoma (mel)", "Evolución de Recall mel"),
    )
    for ax, (metric_key, ylabel, title) in zip(axes, panels):
        for idx, m in enumerate(models):
            xs, ys = [], []
            for xi, (cond, mit) in enumerate(ordered_pairs):
                vals = [
                    float(r[metric_key])
                    for r in rob_rows
                    if r["model"] == m
                    and r["condition"] == cond
                    and r["mitigation"] == mit
                    and r.get(metric_key) is not None
                ]
                if vals:
                    xs.append(xi)
                    ys.append(float(np.mean(vals)))
            if xs:
                ax.plot(
                    xs, ys,
                    color=_model_color(m, idx),
                    ls=_LINE_STYLES[idx % len(_LINE_STYLES)],
                    marker=_model_marker(m, idx),
                    markersize=5.8,
                    markeredgecolor=COLOR_TEXT,
                    markeredgewidth=0.5,
                    lw=2.1,
                    label=_MODEL_LABELS.get(m, m),
                )
        if sep_x is not None:
            ax.axvline(sep_x, color=COLOR_LIGHT, ls="--", lw=1.2)
        ax.set_xticks(x, cond_labels, rotation=24, ha="right", fontsize=9.2, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold", color=COLOR_TEXT)
        ax.set_ylim(0, 1.04)
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8, color=COLOR_TEXT)
        _estilizar(ax, "both")
        ax.legend(frameon=True, facecolor="white", edgecolor=COLOR_BORDER, fontsize=8.8, loc="best")

    fig.suptitle(
        f"Robustez frente a perturbaciones y mitigaciones · partición {rob_split}",
        fontsize=13, fontweight="bold", y=0.96, color=COLOR_TEXT,
    )
    save(fig, output, "comparacion_robustez.png")


def comparar(runs, output, evaluation_subdir=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    loaded = []
    for file in sorted(Path(runs).rglob("metrics.json")):
        if any(part.startswith(".") for part in file.relative_to(runs).parts):
            continue
        if evaluation_subdir is not None and file.parent.parent.name != evaluation_subdir:
            continue
        r = json.loads(file.read_text(encoding="utf-8"))
        mitigation = r.get("mitigation", "none") or "none"
        record = {
            "model": r["model"],
            "seed": r["seed"],
            "split": r["split"],
            "condition": r["condition"],
            "mitigation": mitigation,
            "accuracy": r.get("accuracy"),
            "macro_precision": r.get("macro_precision"),
            "macro_f1": r["macro_f1"],
            "balanced_accuracy": r["balanced_accuracy"],
            "melanoma_recall": r["melanoma_recall"],
            "macro_auc_ovr": r["macro_auc_ovr"],
            "latency_median_ms": r["latency_median_ms"],
            "latency_p95_ms": r.get("latency_p95_ms"),
            "peak_training_allocated_mib": r.get("peak_training_allocated_mib"),
            "parameters": r["parameters"],
            "training_seconds": r["training_seconds"],
            "gpu": None if r.get("evaluation_device") == "cpu" else r.get("environment", {}).get("gpu"),
            "split_sha256": r["split_sha256"],
        }
        loaded.append((file, r, mitigation, record))

    records = [entry[3] for entry in loaded]
    if not records:
        raise ValueError("No hay evaluaciones reales. No se fabricarán tablas ni gráficos de resultados.")
    if len({r["split_sha256"] for r in records}) != 1:
        raise ValueError("No se pueden consolidar particiones diferentes en una misma comparación.")
    keys = [(r["model"], r["seed"], r["split"], r["condition"], r["mitigation"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Evaluaciones duplicadas; seleccione --subcarpeta-evaluacion o separe las raíces.")
    hardware = {(r.get("evaluation_device", "cuda" if r.get("environment", {}).get("gpu") else "cpu"),
                 r.get("environment", {}).get("gpu"), r.get("environment", {}).get("torch")) for _, r, _, _ in loaded}
    if len(hardware) > 1:
        raise ValueError("No se compara eficiencia de dispositivos/entornos mezclados; separe las raíces.")

    guardar_csv(output / "comparacion.csv", records)

    roc_groups = defaultdict(list)
    for file, r, mitigation, record in loaded:
        _graficar_matriz_confusion(r, mitigation, output)
        pred_path = file.parent / "predictions.csv"
        if pred_path.is_file():
            roc_data = _calcular_roc_predicciones(pred_path)
            if roc_data is not None:
                _graficar_roc_individual(roc_data, r, mitigation, output)
                roc_groups[(r["split"], r["condition"], mitigation)].append((record, roc_data))

    _graficar_roc_comparativa(roc_groups, output)
    _graficar_curvas_historia(runs, output)
    _graficar_comparacion_metricas(records, output)
    _graficar_comparacion_robustez(records, output)
    print(f"Comparación real guardada: {output / 'comparacion.csv'}", flush=True)
