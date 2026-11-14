"""Regresiones de inicialización local/Colab y artefactos; sin entrenar HAM10000."""

import importlib.util
import json
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest
import zipfile
from unittest.mock import patch

from ham10000 import CLASES
from ham10000.artifacts import verificar_evaluacion
from ham10000.io import guardar_csv, guardar_json

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("notebook_runtime", ROOT / "colab/runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
spec_check = importlib.util.spec_from_file_location("notebook_check", ROOT / "colab/verificar_notebook.py")
check = importlib.util.module_from_spec(spec_check)
spec_check.loader.exec_module(check)


class NotebookRuntimeTests(unittest.TestCase):
    def test_runtime_detection_is_not_package_detection(self):
        self.assertFalse(runtime.es_colab_alojado({}))
        self.assertTrue(runtime.es_colab_alojado({"COLAB_RELEASE_TAG": "release"}))
        self.assertTrue(runtime.es_colab_alojado({"COLAB_BACKEND_VERSION": "1"}))

    def test_local_project_discovery_and_diagnostics(self):
        self.assertEqual(runtime.localizar_proyecto(cwd=ROOT / "colab"), ROOT)
        self.assertEqual(runtime.localizar_proyecto(cwd=ROOT.parent), ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "RUTA_PROYECTO"):
                runtime.localizar_proyecto(explicit=tmp)
            with self.assertRaisesRegex(FileNotFoundError, "RUTA_DATOS"):
                runtime.localizar_datos(ROOT, tmp)
            nested = Path(tmp) / "HAM10000"
            nested.mkdir()
            (nested / "HAM10000_metadata.csv").write_text("image_id")
            self.assertEqual(runtime.localizar_datos(ROOT, tmp), nested)

    def test_zip_atomic_idempotent_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "a.zip", Path(tmp) / "out"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("nested/file.txt", "data")
            self.assertEqual(runtime.extraer_zip(source, target), target)
            self.assertEqual(runtime.extraer_zip(source, target), target)
            self.assertEqual((target / "nested/file.txt").read_text(), "data")
            other = Path(tmp) / "unknown"
            other.mkdir()
            with self.assertRaises(FileExistsError):
                runtime.extraer_zip(source, other)

    def test_zip_rejects_unsafe_paths_links_and_collisions(self):
        cases = ["../escape", "/absolute", "C:/drive", "nested/../../escape", "AUX.txt", "trailing. "]
        with tempfile.TemporaryDirectory() as tmp:
            for i, name in enumerate(cases):
                source, target = Path(tmp) / f"{i}.zip", Path(tmp) / f"out{i}"
                with zipfile.ZipFile(source, "w") as archive:
                    archive.writestr(name, "untrusted")
                with self.assertRaises(ValueError):
                    runtime.extraer_zip(source, target)
                self.assertFalse(target.exists())
            source = Path(tmp) / "link.zip"
            info = zipfile.ZipInfo("link")
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr(info, "target")
            with self.assertRaises(ValueError):
                runtime.extraer_zip(source, Path(tmp) / "link-out")
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("A.txt", "one")
                archive.writestr("a.txt", "two")
            with self.assertRaises(ValueError):
                runtime.extraer_zip(source, Path(tmp) / "collision-out")

    def test_zip_interrupted_write_does_not_publish_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "a.zip", Path(tmp) / "out"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("file.txt", "data")
            with patch.object(runtime.shutil, "copyfileobj", side_effect=OSError("interrupción")):
                with self.assertRaises(OSError):
                    runtime.extraer_zip(source, target)
            self.assertFalse(target.exists())
            self.assertFalse(list(Path(tmp).glob(".out_*")))

    def test_notebook_clean_schema_syntax_and_safe_defaults(self):
        nb = check.verificar(ROOT / "colab/HAM10000_Grupo09.ipynb")
        self.assertIn("Lesiones Cutáneas Pigmentadas", nb["metadata"]["title"])
        helper = next(c for c in nb["cells"] if c["id"] == "utilidades-runtime")
        self.assertEqual("".join(helper["source"]).strip(), (ROOT / "colab/runtime.py").read_text(encoding="utf-8").strip())
        # Ejercitar la orquestación real del notebook sin inferencia ni checkpoints reales.
        evaluation_source = "".join(next(c for c in nb["cells"] if c["id"] == "evaluacion")["source"])
        models = ("resnet18", "efficientnet_b0", "mobilenet_v3_small")
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            for model in models:
                checkpoint = runs / f"{model}_seed42/best.pt"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b"synthetic placeholder; never loaded")
            calls = []
            namespace = dict(Path=Path, EVALUAR=False, CONFIRMAR_TEST=False, SUBCARPETA_EVAL="evaluacion",
                             MODELOS_EVALUAR=list(models), MODELOS=models, DISPOSITIVO_EVAL=None,
                             WORKERS_EVAL=0, RUNS=runs, config={"seed": 42}, DATOS=runs, SPLIT=runs / "split.csv",
                             sha256=lambda p: "synthetic", ejecutar=lambda *args: calls.append(args),
                             torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
            exec(evaluation_source, namespace)
            self.assertEqual(calls, [])
            namespace["EVALUAR"] = True
            exec(evaluation_source, namespace)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all("--confirmar-test" not in c and "val" in c for c in calls))
            calls.clear()
            namespace["CONFIRMAR_TEST"] = True
            exec(evaluation_source, namespace)
            self.assertEqual(len(calls), 21)
            self.assertTrue(all("--confirmar-test" in c and "test" in c and "cpu" in c for c in calls))

    def test_local_initialization_never_imports_colab(self):
        nb = json.loads((ROOT / "colab/HAM10000_Grupo09.ipynb").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            (data / "HAM10000_metadata.csv").write_text("image_id")
            namespace = {}
            for identifier in ("controles", "utilidades-runtime"):
                exec("".join(next(c for c in nb["cells"] if c["id"] == identifier)["source"]), namespace)
            namespace.update(RUTA_PROYECTO=str(ROOT), RUTA_DATOS=str(data))
            namespace["es_colab_alojado"] = lambda: False
            # Un import local de google.colab fallaría aun si estuviera instalado.
            with patch.dict(sys.modules, {"google.colab": None}):
                exec("".join(next(c for c in nb["cells"] if c["id"] == "inicializacion")["source"]), namespace)
            self.assertFalse(namespace["EN_COLAB"])
            self.assertEqual(namespace["DATOS"], data)

    def test_evaluation_complete_vs_truncated_or_wrong_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            rows = [dict(image_id="one", lesion_id="lesion", group_id="group", split="val", dx=CLASES[0])]
            prediction = dict(image_id="one", lesion_id="lesion", group_id="group", truth=CLASES[0], prediction=CLASES[0],
                              **{f"p_{c}": int(i == 0) for i, c in enumerate(CLASES)})
            guardar_csv(target / "predictions.csv", [prediction])
            guardar_json(target / "metrics.json", dict(n=1, checkpoint_sha256="ck", split_sha256="sp", split="val", condition="clean"))
            args = (target, "ck", "sp", "val", "clean", "none", rows)
            self.assertEqual(verificar_evaluacion(*args)["n"], 1)
            with self.assertRaises(ValueError):
                verificar_evaluacion(target, "other", *args[2:])
            guardar_csv(target / "predictions.csv", [prediction, prediction])
            with self.assertRaises(ValueError):
                verificar_evaluacion(*args)
            (target / "metrics.json").unlink()
            with self.assertRaisesRegex(ValueError, "incompleta"):
                verificar_evaluacion(*args)

    def test_colab_branch_simulated_without_real_cloud_authentication(self):
        nb = json.loads((ROOT / "colab/HAM10000_Grupo09.ipynb").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "data"
            data.mkdir()
            (data / "HAM10000_metadata.csv").write_text("image_id")
            namespace = {}
            for identifier in ("controles", "utilidades-runtime"):
                exec("".join(next(c for c in nb["cells"] if c["id"] == identifier)["source"]), namespace)
            namespace.update(RUTA_PROYECTO=str(ROOT), RUTA_DATOS=str(data), RUTA_DRIVE=str(base / "drive"))
            namespace["es_colab_alojado"] = lambda: True
            original_path = namespace["Path"]
            namespace["Path"] = lambda value: original_path(base / "content" if value == "/content/ham10000_sesion" else value)
            namespace["Path"].cwd = original_path.cwd
            fake_colab = types.ModuleType("google.colab")
            mounted = []
            fake_colab.drive = types.SimpleNamespace(mount=mounted.append)
            with patch.dict(sys.modules, {"google.colab": fake_colab}), patch.dict(runtime.os.environ):
                exec("".join(next(c for c in nb["cells"] if c["id"] == "inicializacion")["source"]), namespace)
            self.assertEqual(mounted, ["/content/drive"])
            self.assertEqual(namespace["RUNS"], base / "drive/runs")

    def test_comparison_rejects_duplicates_and_selects_subfolder(self):
        from ham10000 import plots
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = dict(model="resnet18", seed=42, split="val", condition="clean", macro_f1=0.1,
                           balanced_accuracy=0.1, melanoma_recall=0.1, macro_auc_ovr=0.5,
                           latency_median_ms=1, parameters=1, training_seconds=1, split_sha256="split")
            for folder in ("evaluacion", "evaluacion_cpu"):
                guardar_json(root / "runs/resnet18_seed42" / folder / "val_clean/metrics.json", metrics)
            with self.assertRaisesRegex(ValueError, "duplicadas"):
                plots.comparar(root / "runs", root / "out")
            with patch.object(plots, "_graficar_matriz_confusion"), patch.object(plots, "_graficar_roc_comparativa"), \
                 patch.object(plots, "_graficar_curvas_historia"), patch.object(plots, "_graficar_comparacion_metricas"), \
                 patch.object(plots, "_graficar_comparacion_robustez"):
                plots.comparar(root / "runs", root / "out", evaluation_subdir="evaluacion")
            self.assertEqual(len((root / "out/comparacion.csv").read_text().splitlines()), 2)
            metrics.update(model="efficientnet_b0", environment={"gpu": "otra GPU", "torch": "other"})
            guardar_json(root / "runs/efficientnet_b0_seed42/evaluacion/val_clean/metrics.json", metrics)
            with self.assertRaisesRegex(ValueError, "mezclados"):
                plots.comparar(root / "runs", root / "mixed", evaluation_subdir="evaluacion")


if __name__ == "__main__":
    unittest.main()
