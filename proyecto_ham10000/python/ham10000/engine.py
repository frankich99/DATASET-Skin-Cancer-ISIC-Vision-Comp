"""Entrenamiento reanudable, selección por validación y evaluación separada."""

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from . import CLASES
from .data import HAMDataset, MITIGATIONS, worker_seed
from .io import cargar_split, guardar_csv, guardar_json, sha256
from .metrics import calcular
from .models import cabeza, congelar_backbone, construir


def entorno():
    return {
        "python": platform.python_version(), "system": platform.platform(),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version(),
    }


def semilla(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def guardar_checkpoint(path, state):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def dispositivo(name):
    if torch.device(name).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No hay GPU CUDA. Revise el intérprete/entorno GPU; para evaluar en CPU use --device cpu.")
    return torch.device(name)


def loader(rows, root, config, train=False, condition="clean", mitigation="none"):
    ds = HAMDataset(rows, root, config["image_size"], train, condition, config["seed"], mitigation=mitigation)
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(ds, batch_size=config["batch_size"], shuffle=train,
                      num_workers=config["num_workers"], pin_memory=config["device"] == "cuda",
                      worker_init_fn=worker_seed, generator=generator, drop_last=train)


@torch.inference_mode()
def inferir(model, batches, device):
    model.eval()
    y, p, ids = [], [], []
    for images, labels, image_ids in batches:
        logits = model(images.to(device, non_blocking=True))
        p.extend(logits.softmax(1).cpu().tolist())
        y.extend(labels.tolist())
        ids.extend(image_ids)
    return y, np.asarray(p), ids


def entrenar(name, data_root, split_path, config_path, output_root, resume=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config["epochs"] <= config["head_epochs"] or config["batch_size"] < 2:
        raise ValueError("epochs debe superar head_epochs; batch_size debe ser al menos 2.")
    if config["num_workers"] < 0 or config["patience"] < 1:
        raise ValueError("num_workers y patience no válidos.")
    device = dispositivo(config["device"])
    semilla(config["seed"])
    rows = cargar_split(split_path, data_root)
    split_digest = sha256(split_path)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if len(train_rows) < config["batch_size"] or not val_rows:
        raise ValueError("No hay un lote completo de train o faltan imágenes de validación.")
    run = Path(output_root) / f"{name}_seed{config['seed']}"
    run.mkdir(parents=True, exist_ok=True)
    last_path = run / "last.pt"
    if any(run.iterdir()) and not resume:
        raise FileExistsError(f"Existe {run}; use --resume o una carpeta de salida diferente.")
    if resume and not last_path.exists():
        raise FileNotFoundError(f"No hay checkpoint para reanudar: {last_path}")
    model, weights_name = construir(name, config["pretrained"] and not resume)
    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["config"] != config or checkpoint["split_sha256"] != split_digest:
            raise ValueError("Checkpoint y protocolo/split no coinciden. No se reanuda.")
        if checkpoint["model_name"] != name:
            raise ValueError("Checkpoint de otra arquitectura.")
        finished = checkpoint["epoch"] + 1 >= config["epochs"] or checkpoint["stale"] >= config["patience"]
        record_path = run / "run.json"
        if finished and record_path.is_file() and json.loads(record_path.read_text(encoding="utf-8")).get("status") == "completed":
            print(f"Entrenamiento ya terminado: {run}. No se modifican sus artefactos.", flush=True)
            return run
        weights_name = checkpoint["weights_name"]
        model.load_state_dict(checkpoint["model"])
    model.to(device)
    head_ids = {id(p) for p in cabeza(model, name).parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": config["backbone_lr"]},
        {"params": list(cabeza(model, name).parameters()), "lr": config["head_lr"]},
    ], weight_decay=config["weight_decay"])
    amp = config["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, stale, elapsed = 0, -1.0, 0, 0.0
    history, sessions = [], []
    if resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, stale = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["stale"]
        history, elapsed, sessions = checkpoint["history"], checkpoint["elapsed_seconds"], checkpoint["sessions"]
        restore_rng(checkpoint["rng"])
    sessions.append(entorno())
    train_loader = loader(train_rows, data_root, config, train=True)
    val_loader = loader(val_rows, data_root, config)
    counts = np.bincount([CLASES.index(r["dx"]) for r in train_rows], minlength=len(CLASES))
    if np.any(counts == 0):
        raise ValueError("Train no contiene las siete clases.")
    weights = torch.tensor(len(train_rows) / (len(CLASES) * counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights if config["class_weights"] else None)
    metadata = {"model": name, "weights": weights_name, "config": config,
                "split_sha256": split_digest, "classes": list(CLASES), "sessions": sessions,
                "parameters": sum(p.numel() for p in model.parameters()),
                "status": "training", "train_class_weights": weights.tolist()}
    guardar_json(run / "run.json", metadata)
    for epoch in range(start, config["epochs"]):
        if stale >= config["patience"]:
            break
        begun = time.perf_counter()
        semilla(config["seed"] + epoch)
        train_loader.dataset.epoch = epoch
        model.train()
        frozen = epoch < config["head_epochs"]
        congelar_backbone(model, name, frozen)
        factor = 1.0 if frozen else 0.5 * (1 + math.cos(math.pi * (epoch - config["head_epochs"]) /
                                                       max(1, config["epochs"] - config["head_epochs"])))
        optimizer.param_groups[0]["lr"] = config["backbone_lr"] * factor
        optimizer.param_groups[1]["lr"] = config["head_lr"] * factor
        train_loader.generator.manual_seed(config["seed"] + epoch)
        total_loss, seen = 0.0, 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for images, labels, _ in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            context = torch.amp.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
            with context:
                loss = criterion(model(images), labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError("Pérdida no finita. Revise datos y AMP.")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
        y, probabilities, _ = inferir(model, val_loader, device)
        scores = calcular(y, probabilities)
        val_loss = float(-np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-12, 1)).mean())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - begun
        elapsed += seconds
        improved = scores["macro_f1"] > best
        best, stale = (scores["macro_f1"], 0) if improved else (best, stale + 1)
        history.append({"epoch": epoch + 1, "stage": "head" if frozen else "fine_tuning",
                        "train_loss": total_loss / seen, "val_loss": val_loss,
                        "val_macro_f1": scores["macro_f1"], "val_accuracy": scores["accuracy"],
                        "seconds": seconds,
                        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None})
        state = {"model_name": name, "weights_name": weights_name, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                 "config": config, "split_sha256": split_digest, "epoch": epoch,
                 "best": best, "stale": stale, "history": history, "rng": rng_state(),
                 "elapsed_seconds": elapsed, "sessions": sessions}
        if improved:
            guardar_checkpoint(run / "best.pt", state)
        guardar_checkpoint(last_path, state)
        guardar_csv(run / "history.csv", history)
        print(f"{name} época {epoch+1}: val macro F1={scores['macro_f1']:.4f}; {seconds:.1f}s", flush=True)
    metadata.update(status="completed", training_seconds=elapsed, best_val_macro_f1=best,
                    completed_epochs=len(history), sessions=sessions)
    guardar_json(run / "run.json", metadata)
    return run


def evaluar(checkpoint_path, data_root, split_path, output, split="val", condition="clean", confirm_test=False, mitigation="none",
            device_override=None, num_workers=None):
    if split not in ("val", "test"):
        raise ValueError("Solo se permite evaluar val o test.")
    if split == "test" and not confirm_test:
        raise ValueError("Test está reservado. Añada --confirmar-test solo al congelar el protocolo.")
    if mitigation not in MITIGATIONS:
        raise ValueError(f"Mitigación desconocida: {mitigation}")
    target = Path(output) / (f"{split}_{condition}" if mitigation == "none" else f"{split}_{condition}_{mitigation}")
    if target.exists():
        raise FileExistsError(f"Evaluación existente: {target}. Use otra carpeta; no se sobrescribe.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["split_sha256"] != sha256(split_path):
        raise ValueError("El split difiere del utilizado para entrenar.")
    training_config = checkpoint["config"]
    config = dict(training_config)
    if device_override is not None:
        config["device"] = device_override
    if num_workers is not None:
        if num_workers < 0:
            raise ValueError("num_workers no puede ser negativo.")
        config["num_workers"] = num_workers
    device = dispositivo(config["device"])
    rows = [r for r in cargar_split(split_path, data_root) if r["split"] == split]
    if not rows:
        raise ValueError("La partición elegida no contiene imágenes.")
    model, _ = construir(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    y, probabilities, ids = inferir(model, loader(rows, data_root, config, condition=condition, mitigation=mitigation), device)
    result = calcular(y, probabilities)
    by_id = {r["image_id"]: r for r in rows}
    predictions = [dict(
        image_id=image_id, lesion_id=by_id[image_id]["lesion_id"], group_id=by_id[image_id]["group_id"],
        truth=CLASES[label], prediction=CLASES[int(prob.argmax())],
        **{f"p_{c}": float(prob[i]) for i, c in enumerate(CLASES)}
    ) for image_id, label, prob in zip(ids, y, probabilities)]
    # Latencia de cómputo, sin lectura/transformación; mismo batch 1 en todos los modelos.
    sample = HAMDataset(rows[:1], data_root, config["image_size"])[0][0].unsqueeze(0).to(device)
    model.eval()
    latencies = []
    with torch.inference_mode():
        for i in range(60):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            begun = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if i >= 10:
                latencies.append((time.perf_counter() - begun) * 1000)
    run_record = Path(checkpoint_path).parent / "run.json"
    total_seconds = checkpoint["elapsed_seconds"]
    if run_record.exists():
        record = json.loads(run_record.read_text(encoding="utf-8"))
        if record.get("split_sha256") == checkpoint["split_sha256"] and record.get("config") == training_config:
            total_seconds = record.get("training_seconds", total_seconds)
    peaks = [r["peak_allocated_mib"] for r in checkpoint["history"] if r["peak_allocated_mib"] is not None]
    result.update(model=checkpoint["model_name"], split=split, condition=condition, mitigation=mitigation,
                  seed=config["seed"], split_sha256=checkpoint["split_sha256"],
                   checkpoint_sha256=sha256(checkpoint_path), environment=entorno(),
                   evaluation_device=str(device), evaluation_num_workers=config["num_workers"],
                  latency_median_ms=float(np.median(latencies)), latency_p95_ms=float(np.percentile(latencies, 95)),
                  latency_protocol="batch 1; 10 warmup + 50 mediciones; forward FP32; excluye IO",
                  training_seconds=total_seconds, peak_training_allocated_mib=max(peaks) if peaks else None,
                  parameters=sum(p.numel() for p in model.parameters()))
    # Publicar ambos archivos juntos. Una interrupción no deja una evaluación final a medias.
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}_", dir=target.parent))
    try:
        guardar_csv(staging / "predictions.csv", predictions)
        guardar_json(staging / "metrics.json", result)
        if target.exists():
            raise FileExistsError(f"Evaluación creada por otro proceso: {target}")
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Evaluación guardada: {target}", flush=True)
    return result
