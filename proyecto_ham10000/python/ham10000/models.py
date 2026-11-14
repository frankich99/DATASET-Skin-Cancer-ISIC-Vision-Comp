"""Tres arquitecturas, pesos explícitos y adaptación a siete clases."""

from torch import nn
from torchvision import models

from . import CLASES, MODELOS


def construir(name, pretrained=True):
    weights = {
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    }
    if name not in MODELOS:
        raise ValueError(f"Modelo no permitido: {name}")
    model = getattr(models, name)(weights=weights[name] if pretrained else None)
    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, len(CLASES))
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASES))
    return model, str(weights[name]) if pretrained else "sin_preentrenamiento"


def cabeza(model, name):
    return model.fc if name == "resnet18" else model.classifier


def congelar_backbone(model, name, frozen):
    for parameter in model.parameters():
        parameter.requires_grad_(not frozen)
    for parameter in cabeza(model, name).parameters():
        parameter.requires_grad_(True)
    if frozen:
        # Evita actualizar medias y varianzas de BatchNorm durante adaptación de cabeza.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
