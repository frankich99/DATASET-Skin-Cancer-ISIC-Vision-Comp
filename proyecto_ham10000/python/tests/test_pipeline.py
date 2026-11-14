"""Pruebas de fugas, transformaciones, métricas y checkpoints sin dataset real."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ham10000 import CLASES
from ham10000.data import HAMDataset, letterbox, perturbar
from ham10000.engine import entorno, guardar_checkpoint, restore_rng, rng_state, semilla
from ham10000.io import guardar_csv, guardar_json, leer_csv
from ham10000.metrics import calcular
from ham10000.plots import _calcular_roc_predicciones, comparar
from ham10000.prepare import asignar_splits, exportar_carpetas


class PipelineTests(unittest.TestCase):
    def test_split_reproducible_y_sin_fuga(self):
        rows = [dict(image_id=f"{c}_{j}_{k}", lesion_id=f"{c}_{j}", group_id=f"{c}_{j}", dx=c)
                for c in CLASES for j in range(50) for k in range(1 + (j % 2))]
        a = asignar_splits(rows)
        self.assertEqual(a, asignar_splits(rows))
        for split in ("train", "val", "test"):
            self.assertEqual({r["dx"] for r in a if r["split"] == split}, set(CLASES))

    def test_metrica_perfecta(self):
        result = calcular(list(range(7)), np.eye(7))
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(result["macro_auc_ovr"], 1)
        self.assertEqual(result["balanced_accuracy"], 1)

    def test_clase_ausente_no_inventa_auc(self):
        result = calcular([0, 0], np.full((2, 7), 1/7))
        self.assertIsNone(result["macro_auc_ovr"])
        self.assertIsNone(result["melanoma_recall"])

    def test_letterbox_y_ruido_determinista(self):
        image = letterbox(Image.new("RGB", (600, 450), "red"))
        self.assertEqual(image.size, (224, 224))
        a = np.asarray(perturbar(image, "noise_002", "ISIC_TEST", 42))
        b = np.asarray(perturbar(image, "noise_002", "ISIC_TEST", 42))
        np.testing.assert_array_equal(a, b)

    def test_aumento_independiente_del_rng_del_modelo(self):
        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (30, 20), "red").save(Path(tmp) / "a.jpg")
            ds = HAMDataset([dict(relative_path="a.jpg", dx="mel", image_id="a")], tmp, augment=True)
            a = ds[0][0]
            torch.rand(300)
            self.assertTrue(torch.equal(a, ds[0][0]))

    def test_checkpoint_seguro_rng_y_entorno(self):
        semilla(42)
        state = rng_state()
        expected = torch.rand(5)
        restore_rng(state)
        self.assertTrue(torch.equal(expected, torch.rand(5)))
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "last.pt"
            guardar_checkpoint(file, {"rng": state, "environment": entorno()})
            value = torch.load(file, weights_only=True, map_location="cpu")
            self.assertIn("rng", value)

    def test_exportar_carpetas_sin_fuga(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            rows = []
            for i, c in enumerate(CLASES):
                for s in ("train", "val", "test"):
                    image_id = f"{c}_{s}"
                    Image.new("RGB", (40, 30), (20 * i, 100, 150)).save(raw / f"{image_id}.jpg")
                    rows.append(dict(
                        image_id=image_id, lesion_id=f"L_{image_id}", dx=c,
                        group_id=f"L_{image_id}", split=s, relative_path=f"{image_id}.jpg",
                    ))
            split_path = root / "split.csv"
            from ham10000.io import guardar_csv
            guardar_csv(split_path, rows)
            out = root / "processed"
            manifest, stats = exportar_carpetas(raw, split_path, out)
            self.assertEqual(len(manifest), 21)
            self.assertEqual(stats["conteos_split"], {"train": 7, "val": 7, "test": 7})
            self.assertTrue((out / ".gitkeep").is_file())
            self.assertTrue((out / "dataset_procesado.csv").is_file())
            self.assertTrue((out / "estadisticas.json").is_file())
            for s in ("train", "val", "test"):
                for c in CLASES:
                    self.assertTrue((out / s / c / f"{c}_{s}.jpg").is_file())

    @staticmethod
    def _crear_eval_sintetica(
        eval_dir, model, seed, split, condition, y, probs,
        mitigation=None, peak_vram=256.0, write_predictions=True, split_sha="sha_test_fijo",
    ):
        eval_dir.mkdir(parents=True, exist_ok=True)
        if write_predictions:
            guardar_csv(eval_dir / "predictions.csv", [
                dict(
                    image_id=f"ISIC_{i:04d}",
                    lesion_id=f"L_{i:04d}",
                    group_id=f"L_{i:04d}",
                    truth=CLASES[label],
                    prediction=CLASES[int(prob.argmax())],
                    **{f"p_{c}": float(prob[k]) for k, c in enumerate(CLASES)},
                )
                for i, (label, prob) in enumerate(zip(y, probs))
            ])
        result = calcular(y, probs)
        result.update(
            model=model,
            split=split,
            condition=condition,
            seed=seed,
            split_sha256=split_sha,
            checkpoint_sha256="chk_sha_sintetico",
            environment={"gpu": "SyntheticGPU" if peak_vram is not None else None},
            latency_median_ms=4.5 if model == "resnet18" else 6.2,
            latency_p95_ms=5.8 if model == "resnet18" else 7.9,
            latency_protocol="sintético",
            training_seconds=18.0,
            peak_training_allocated_mib=peak_vram,
            parameters=11180103 if model == "resnet18" else 4016515,
        )
        if mitigation is not None:
            result["mitigation"] = mitigation
        guardar_json(eval_dir / "metrics.json", result)

    @staticmethod
    def _probs_sinteticas(y, strength, rng):
        logits = rng.normal(0.0, 0.4, size=(len(y), len(CLASES)))
        for i, label in enumerate(y):
            logits[i, label] += strength
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def test_comparar_genera_artefactos_completos(self):
        rng = np.random.default_rng(123)
        y_full = np.repeat(np.arange(len(CLASES)), 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            out = root / "resultados"

            specs = [
                ("resnet18", 2.2, 240.0),
                ("efficientnet_b0", 1.8, 310.0),
            ]
            for model, base_strength, vram in specs:
                run_dir = runs / f"{model}_seed42"
                guardar_csv(run_dir / "history.csv", [
                    dict(
                        epoch=1, stage="head", train_loss=1.4, val_loss=1.1,
                        val_macro_f1=0.55, val_accuracy=0.60, seconds=5.0, peak_allocated_mib=vram,
                    ),
                    dict(
                        epoch=2, stage="fine_tuning", train_loss=0.8, val_loss=0.7,
                        val_macro_f1=0.74, val_accuracy=0.78, seconds=6.0, peak_allocated_mib=vram,
                    ),
                ])
                # val/clean sin clave mitigation (compatibilidad hacia atrás -> "none")
                self._crear_eval_sintetica(
                    run_dir / "val_clean", model, 42, "val", "clean",
                    y_full, self._probs_sinteticas(y_full, base_strength - 0.1, rng),
                    mitigation=None, peak_vram=vram,
                )
                # test/clean sin clave mitigation -> "none"
                self._crear_eval_sintetica(
                    run_dir / "test_clean", model, 42, "test", "clean",
                    y_full, self._probs_sinteticas(y_full, base_strength, rng),
                    mitigation=None, peak_vram=vram,
                )
                # test/noise_005 sin mitigación
                self._crear_eval_sintetica(
                    run_dir / "test_noise_005", model, 42, "test", "noise_005",
                    y_full, self._probs_sinteticas(y_full, base_strength - 0.8, rng),
                    mitigation="none", peak_vram=vram,
                )
                # test/noise_005 con mitigación median_3
                self._crear_eval_sintetica(
                    run_dir / "test_noise_005_median_3", model, 42, "test", "noise_005",
                    y_full, self._probs_sinteticas(y_full, base_strength - 0.3, rng),
                    mitigation="median_3", peak_vram=vram,
                )
                # test/blur_1 sin predictions.csv (debe tolerarse sin fallar)
                self._crear_eval_sintetica(
                    run_dir / "test_blur_1", model, 42, "test", "blur_1",
                    y_full, self._probs_sinteticas(y_full, base_strength - 0.5, rng),
                    mitigation=None, peak_vram=vram, write_predictions=False,
                )

            comparar(runs, out)

            comp_csv = out / "comparacion.csv"
            self.assertTrue(comp_csv.is_file())
            rows = leer_csv(comp_csv)
            self.assertEqual(len(rows), 10)
            required_cols = {
                "model", "seed", "split", "condition", "mitigation", "accuracy",
                "macro_precision", "macro_f1", "balanced_accuracy", "melanoma_recall", "macro_auc_ovr",
                "latency_median_ms", "latency_p95_ms", "peak_training_allocated_mib",
                "parameters", "training_seconds", "gpu", "split_sha256",
            }
            self.assertTrue(required_cols.issubset(rows[0].keys()))
            self.assertEqual({r["mitigation"] for r in rows}, {"none", "median_3"})

            expected_files = [
                "confusion_resnet18_42_test_clean.png",
                "confusion_resnet18_42_test_noise_005.png",
                "confusion_resnet18_42_test_noise_005_median_3.png",
                "confusion_efficientnet_b0_42_test_clean.png",
                "confusion_efficientnet_b0_42_test_noise_005_median_3.png",
                "curvas_resnet18_seed42.png",
                "curvas_efficientnet_b0_seed42.png",
                "roc_resnet18_42_test_clean.png",
                "roc_resnet18_42_test_noise_005.png",
                "roc_resnet18_42_test_noise_005_median_3.png",
                "roc_efficientnet_b0_42_test_clean.png",
                "roc_comparativa_val_clean.png",
                "roc_comparativa_test_clean.png",
                "roc_comparativa_test_noise_005.png",
                "roc_comparativa_test_noise_005_median_3.png",
                "comparacion_metricas.png",
                "comparacion_robustez.png",
            ]
            for name in expected_files:
                file_path = out / name
                self.assertTrue(file_path.is_file(), f"Falta gráfico esperado: {name}")
                with Image.open(file_path) as img:
                    self.assertGreater(img.width, 0)
                    self.assertGreater(img.height, 0)

            self.assertFalse((out / "roc_resnet18_42_test_blur_1.png").exists())
            self.assertFalse((out / "roc_comparativa_test_blur_1.png").exists())

    def test_comparar_clase_ausente_en_predictions_sin_fallar_ni_inventar_auc(self):
        rng = np.random.default_rng(99)
        # Faltan "df" (idx 3) y "vasc" (idx 6); "mel" (idx 4) sí está presente.
        y_partial = np.array([0, 0, 1, 1, 2, 2, 4, 4, 5, 5])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = root / "runs"
            out = root / "resultados"
            eval_dir = runs / "resnet18_seed42" / "val_clean"
            self._crear_eval_sintetica(
                eval_dir, "resnet18", 42, "val", "clean",
                y_partial, self._probs_sinteticas(y_partial, 1.5, rng),
                mitigation=None, peak_vram=None,
            )

            roc_data = _calcular_roc_predicciones(eval_dir / "predictions.csv")
            self.assertIsNotNone(roc_data)
            self.assertIn("df", roc_data["omitted"])
            self.assertIn("vasc", roc_data["omitted"])
            self.assertNotIn("df", roc_data["class_curves"])
            self.assertNotIn("vasc", roc_data["class_curves"])
            self.assertIn("mel", roc_data["class_curves"])
            self.assertIsNone(roc_data["macro_tpr"])
            self.assertIsNone(roc_data["macro_auc"])

            comparar(runs, out)

            rows = leer_csv(out / "comparacion.csv")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["mitigation"], "none")
            self.assertEqual(rows[0]["macro_auc_ovr"], "")
            self.assertEqual(rows[0]["peak_training_allocated_mib"], "")
            self.assertTrue((out / "confusion_resnet18_42_val_clean.png").is_file())
            self.assertTrue((out / "roc_resnet18_42_val_clean.png").is_file())
            self.assertTrue((out / "roc_comparativa_val_clean.png").is_file())
            self.assertTrue((out / "comparacion_metricas.png").is_file())
            self.assertFalse((out / "comparacion_robustez.png").exists())

            # Caso extremo: también falta "mel" (solo una clase presente -> ninguna curva OVR válida)
            runs_single = root / "runs_single"
            out_single = root / "resultados_single"
            y_single = np.array([5, 5, 5, 5])
            eval_single = runs_single / "resnet18_seed42" / "val_clean"
            self._crear_eval_sintetica(
                eval_single, "resnet18", 42, "val", "clean",
                y_single, self._probs_sinteticas(y_single, 1.2, rng),
                mitigation=None, peak_vram=None,
            )
            roc_single = _calcular_roc_predicciones(eval_single / "predictions.csv")
            self.assertEqual(roc_single["class_curves"], {})
            self.assertIsNone(roc_single["macro_auc"])
            comparar(runs_single, out_single)
            self.assertTrue((out_single / "roc_resnet18_42_val_clean.png").is_file())
            self.assertTrue((out_single / "roc_comparativa_val_clean.png").is_file())

    def test_roc_predicciones_perfectas_y_consistencia_metrics(self):
        rng = np.random.default_rng(7)
        y_perfect = np.repeat(np.arange(len(CLASES)), 3)
        probs_perfect = np.eye(len(CLASES))[y_perfect]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eval_perf = root / "perf"
            self._crear_eval_sintetica(
                eval_perf, "resnet18", 42, "test", "clean",
                y_perfect, probs_perfect,
            )
            roc_perf = _calcular_roc_predicciones(eval_perf / "predictions.csv")
            metrics_perf = calcular(y_perfect, probs_perfect)
            self.assertEqual(roc_perf["macro_auc"], 1.0)
            self.assertEqual(roc_perf["macro_auc"], metrics_perf["macro_auc_ovr"])

            probs_noisy = self._probs_sinteticas(y_perfect, 1.4, rng)
            eval_noisy = root / "noisy"
            self._crear_eval_sintetica(
                eval_noisy, "resnet18", 42, "test", "clean",
                y_perfect, probs_noisy,
            )
            roc_noisy = _calcular_roc_predicciones(eval_noisy / "predictions.csv")
            metrics_noisy = calcular(y_perfect, probs_noisy)
            self.assertAlmostEqual(roc_noisy["macro_auc"], metrics_noisy["macro_auc_ovr"], places=12)


if __name__ == "__main__":
    unittest.main()
