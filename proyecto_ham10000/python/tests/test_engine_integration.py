"""Integración con imágenes artificiales; nunca entrena HAM10000."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from ham10000 import CLASES
from ham10000.data import HAMDataset, MITIGATIONS, mitigar
from ham10000.engine import entrenar, evaluar
from ham10000 import engine
from ham10000.io import guardar_csv, sha256


class IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Prueba corta de integración GPU")
    def test_train_resume_evaluate_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "imagenes"
            data.mkdir()
            rows = []
            rng = np.random.default_rng(42)
            for c in CLASES:
                for i, split in enumerate(("train", "train", "val", "test")):
                    image_id = f"{c}_{i}"
                    file = data / f"{image_id}.jpg"
                    Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)).save(file)
                    rows.append(dict(image_id=image_id, lesion_id=image_id, group_id=image_id,
                                     dx=c, split=split, relative_path=file.name))
            split_path = root / "split.csv"
            guardar_csv(split_path, rows)
            config = json.loads((Path(__file__).resolve().parents[2] / "configs/base.json").read_text())
            config.update(epochs=2, head_epochs=1, batch_size=7, patience=3, num_workers=0, pretrained=False, image_size=64, device="cuda")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            save = engine.guardar_checkpoint
            def interrupt_after_first_epoch(path, state):
                save(path, state)
                if Path(path).name == "last.pt" and state["epoch"] == 0:
                    raise RuntimeError("interrupción sintética tras primera época")
            with patch.object(engine, "guardar_checkpoint", side_effect=interrupt_after_first_epoch):
                with self.assertRaisesRegex(RuntimeError, "interrupción sintética"):
                    entrenar("resnet18", data, split_path, config_path, root / "runs")
            run = root / "runs/resnet18_seed42"
            self.assertEqual(torch.load(run / "last.pt", weights_only=True, map_location="cpu")["epoch"], 0)
            self.assertTrue((run / "best.pt").is_file())
            entrenar("resnet18", data, split_path, config_path, root / "runs", resume=True)
            self.assertEqual(len(torch.load(run / "last.pt", weights_only=True, map_location="cpu")["history"]), 2)
            before = {p.name: sha256(p) for p in run.iterdir() if p.is_file()}
            entrenar("resnet18", data, split_path, config_path, root / "runs", resume=True)
            self.assertEqual(before, {p.name: sha256(p) for p in run.iterdir() if p.is_file()})
            config_path.write_text(json.dumps(dict(config, num_workers=1)))
            with self.assertRaisesRegex(ValueError, "protocolo/split"):
                entrenar("resnet18", data, split_path, config_path, root / "runs", resume=True)
            config_path.write_text(json.dumps(config))
            result = evaluar(run / "best.pt", data, split_path, root / "eval", device_override="cpu", num_workers=0)
            self.assertEqual(result["n"], 7)
            self.assertEqual(result["evaluation_device"], "cpu")
            self.assertEqual(torch.load(run / "best.pt", weights_only=True, map_location="cpu")["config"]["device"], "cuda")
            with self.assertRaisesRegex(ValueError, "negativo"):
                evaluar(run / "best.pt", data, split_path, root / "bad_workers", num_workers=-1)
            with self.assertRaisesRegex(ValueError, "val o test"):
                evaluar(run / "best.pt", data, split_path, root / "train", split="train")
            with self.assertRaises(ValueError):
                evaluar(run / "best.pt", data, split_path, root / "test", split="test")
            with self.assertRaises(FileExistsError):
                evaluar(run / "best.pt", data, split_path, root / "eval")

    def test_mitigation_filters_and_evaluate_subfolder(self):
        self.assertEqual(MITIGATIONS, ("none", "median_3", "gaussian_1"))
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        arr[8, 8] = (255, 255, 255)
        image = Image.fromarray(arr)

        out_none = mitigar(image, "none")
        self.assertIs(out_none, image)
        np.testing.assert_array_equal(np.asarray(out_none), arr)

        out_median = mitigar(image, "median_3")
        self.assertEqual((out_median.size, out_median.mode), ((16, 16), "RGB"))
        np.testing.assert_array_equal(np.asarray(out_median), np.asarray(mitigar(image, "median_3")))
        self.assertEqual(out_median.getpixel((8, 8)), (0, 0, 0))

        out_gaussian = mitigar(image, "gaussian_1")
        self.assertEqual((out_gaussian.size, out_gaussian.mode), ((16, 16), "RGB"))
        np.testing.assert_array_equal(np.asarray(out_gaussian), np.asarray(mitigar(image, "gaussian_1")))
        center_val = out_gaussian.getpixel((8, 8))[0]
        neighbor_val = out_gaussian.getpixel((8, 7))[0]
        self.assertTrue(0 < center_val < 255)
        self.assertGreater(neighbor_val, 0)

        with self.assertRaises(ValueError):
            mitigar(image, "invalida")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "imagenes"
            data.mkdir()
            rows = []
            rng = np.random.default_rng(123)
            for c in CLASES:
                for i, split in enumerate(("train", "train", "val", "test")):
                    image_id = f"{c}_{i}"
                    file = data / f"{image_id}.jpg"
                    Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)).save(file)
                    rows.append(dict(image_id=image_id, lesion_id=image_id, group_id=image_id,
                                     dx=c, split=split, relative_path=file.name))
            split_path = root / "split.csv"
            guardar_csv(split_path, rows)

            with self.assertRaises(ValueError):
                HAMDataset(rows[:1], data, mitigation="invalida")
            ds_med = HAMDataset(rows[:1], data, size=64, condition="noise_005", mitigation="median_3")
            self.assertTrue(torch.equal(ds_med[0][0], ds_med[0][0]))

            config = json.loads((Path(__file__).resolve().parents[2] / "configs/base.json").read_text(encoding="utf-8"))
            config.update(epochs=2, head_epochs=1, batch_size=7, patience=3, num_workers=0, pretrained=False, image_size=64, device="cpu")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            run = entrenar("resnet18", data, split_path, config_path, root / "runs")
            eval_dir = root / "eval"

            clean_result = evaluar(run / "best.pt", data, split_path, eval_dir)
            self.assertEqual(clean_result["mitigation"], "none")
            self.assertTrue((eval_dir / "val_clean" / "metrics.json").is_file())
            with patch.object(engine, "guardar_json", side_effect=OSError("escritura interrumpida")):
                with self.assertRaises(OSError):
                    evaluar(run / "best.pt", data, split_path, root / "failure")
            self.assertFalse((root / "failure/val_clean").exists())
            self.assertFalse(list((root / "failure").glob(".val_clean_*")))

            mit_result = evaluar(run / "best.pt", data, split_path, eval_dir,
                                 condition="noise_005", mitigation="median_3")
            mit_dir = eval_dir / "val_noise_005_median_3"
            self.assertTrue(mit_dir.is_dir())
            self.assertTrue((mit_dir / "metrics.json").is_file())
            self.assertTrue((mit_dir / "predictions.csv").is_file())
            self.assertTrue((eval_dir / "val_clean" / "metrics.json").is_file())

            clean_metrics = json.loads((eval_dir / "val_clean" / "metrics.json").read_text(encoding="utf-8"))
            mit_metrics = json.loads((mit_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(clean_metrics["mitigation"], "none")
            self.assertEqual(mit_metrics["mitigation"], "median_3")
            self.assertEqual(mit_metrics["condition"], "noise_005")
            self.assertEqual(mit_metrics["split"], "val")
            self.assertEqual(mit_metrics["n"], 7)
            self.assertEqual(mit_result["mitigation"], "median_3")

            with self.assertRaises(ValueError):
                evaluar(run / "best.pt", data, split_path, eval_dir, mitigation="invalida")
            with self.assertRaises(FileExistsError):
                evaluar(run / "best.pt", data, split_path, eval_dir,
                        condition="noise_005", mitigation="median_3")


if __name__ == "__main__":
    unittest.main()
