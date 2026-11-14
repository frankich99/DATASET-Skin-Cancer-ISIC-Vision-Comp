"""Transformaciones comunes y Dataset portable para Windows y Colab."""

import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from . import CLASES

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONDITIONS = ("clean", "noise_002", "noise_005", "blur_1", "blur_2")
MITIGATIONS = ("none", "median_3", "gaussian_1")


def letterbox(image, size=224):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.contain(image, (size, size), method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), tuple(round(x * 255) for x in MEAN))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def perturbar(image, condition, image_id, seed=42):
    if condition not in CONDITIONS:
        raise ValueError(f"Condición desconocida: {condition}")
    if condition.startswith("blur_"):
        return image.filter(ImageFilter.GaussianBlur(radius=int(condition.split("_")[1])))
    if condition.startswith("noise_"):
        sigma = {"noise_002": 0.02, "noise_005": 0.05}[condition]
        key = hashlib.sha256(f"{seed}:{image_id}:{condition}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        array = np.asarray(image, dtype=np.float32) / 255
        array = np.clip(array + rng.normal(0, sigma, array.shape), 0, 1)
        return Image.fromarray(np.rint(array * 255).astype(np.uint8))
    return image


def mitigar(image, mitigation="none"):
    if mitigation not in MITIGATIONS:
        raise ValueError(f"Mitigación desconocida: {mitigation}")
    if mitigation == "median_3":
        return image.filter(ImageFilter.MedianFilter(size=3))
    if mitigation == "gaussian_1":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    return image


class HAMDataset(Dataset):
    def __init__(self, rows, data_root, size=224, augment=False, condition="clean", seed=42, mitigation="none"):
        if mitigation not in MITIGATIONS:
            raise ValueError(f"Mitigación desconocida: {mitigation}")
        self.rows, self.root = list(rows), Path(data_root)
        self.size, self.augment, self.condition, self.seed, self.mitigation = size, augment, condition, seed, mitigation
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row["relative_path"]) as source:
            image = letterbox(source, self.size)
        if self.augment:
            key = hashlib.sha256(f"{self.seed}:{self.epoch}:{row['image_id']}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(key[:8], "little"))
            def uniform(low, high):
                return float(torch.empty(()).uniform_(low, high, generator=generator))
            if torch.rand((), generator=generator).item() < 0.5:
                image = ImageOps.mirror(image)
            if torch.rand((), generator=generator).item() < 0.5:
                image = ImageOps.flip(image)
            angle = uniform(-30, 30)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR,
                                 fillcolor=tuple(round(x * 255) for x in MEAN))
            image = ImageEnhance.Brightness(image).enhance(uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(uniform(0.9, 1.1))
        image = perturbar(image, self.condition, row["image_id"], self.seed)
        image = mitigar(image, self.mitigation)
        tensor = TF.normalize(TF.to_tensor(image), MEAN, STD)
        return tensor, CLASES.index(row["dx"]), row["image_id"]


def worker_seed(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
