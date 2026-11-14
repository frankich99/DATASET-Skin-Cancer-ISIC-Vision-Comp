"""Verifica CUDA y tres CNN con datos sintéticos; no utiliza HAM10000."""

import gc
import sys

import torch
import torchvision
from torchvision.models import efficientnet_b0, mobilenet_v3_small, resnet18


def main():
    print(f"Python: {sys.executable}", flush=True)
    print(f"PyTorch: {torch.__version__}", flush=True)
    print(f"torchvision: {torchvision.__version__}", flush=True)
    print(f"CUDA del paquete: {torch.version.cuda}", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible en este entorno Python.")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    print(f"GPU: {properties.name}", flush=True)
    print(f"VRAM: {properties.total_memory / 1024**3:.2f} GiB", flush=True)
    print(f"Capacidad de cómputo: {capability}", flush=True)
    print(f"Arquitecturas compiladas: {torch.cuda.get_arch_list()}", flush=True)
    print(f"cuDNN: {torch.backends.cudnn.version()}", flush=True)

    a = torch.ones((128, 128), device=device)
    result = a @ a
    torch.cuda.synchronize()
    if not torch.all(result == 128).item():
        raise RuntimeError("La multiplicación en GPU produjo un resultado incorrecto.")
    print("Multiplicación matricial CUDA: OK", flush=True)
    del a, result

    for name, constructor in (
        ("ResNet-18", resnet18),
        ("EfficientNet-B0", efficientnet_b0),
        ("MobileNetV3-Small", mobilenet_v3_small),
    ):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        # No se descargan pesos ni se leen imágenes. Batch 2 permite probar BatchNorm.
        model = constructor(weights=None, num_classes=7).to(device).train()
        images = torch.randn((2, 3, 224, 224), device=device)
        labels = torch.tensor([0, 1], device=device)
        output = model(images)
        if tuple(output.shape) != (2, 7):
            raise RuntimeError(f"Salida incorrecta en {name}: {output.shape}")
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Pérdida no finita en {name}.")
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError(f"Gradientes inválidos en {name}.")
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"{name}: forward/backward OK; memoria asignada máxima {peak:.0f} MiB", flush=True)
        del gradients, loss, output, images, labels, model
        gc.collect()

    torch.cuda.empty_cache()
    print("Verificación completada. No se entrenó con el dataset ni se guardaron pesos.", flush=True)


if __name__ == "__main__":
    main()
