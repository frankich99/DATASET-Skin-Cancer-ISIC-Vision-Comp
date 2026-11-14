"""Interfaz local/Colab: preparar, entrenar, evaluar, figuras y comparación."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from ham10000 import MODELOS
from ham10000.data import MITIGATIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("preparar")
    prepare.add_argument("--datos", type=Path, required=True)
    prepare.add_argument("--salida", type=Path, required=True)
    prepare.add_argument("--semilla", type=int, default=42)
    validate = commands.add_parser("validar")
    validate.add_argument("--datos", type=Path, required=True)
    validate.add_argument("--split", type=Path, required=True)
    export = commands.add_parser("exportar")
    export.add_argument("--datos", type=Path, required=True)
    export.add_argument("--split", type=Path, required=True)
    export.add_argument("--salida", type=Path, required=True)
    export.add_argument("--tamano", type=int, default=224)
    export.add_argument("--calidad", type=int, default=95)
    train = commands.add_parser("entrenar")
    train.add_argument("--modelo", choices=MODELOS, required=True)
    train.add_argument("--datos", type=Path, required=True)
    train.add_argument("--split", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--salida", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluar")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--datos", type=Path, required=True)
    evaluate.add_argument("--split", type=Path, required=True)
    evaluate.add_argument("--salida", type=Path, required=True)
    evaluate.add_argument("--particion", choices=("val", "test"), default="val")
    evaluate.add_argument("--condicion", choices=("clean", "noise_002", "noise_005", "blur_1", "blur_2"), default="clean")
    evaluate.add_argument("--mitigacion", choices=MITIGATIONS, default="none")
    evaluate.add_argument("--confirmar-test", action="store_true")
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default=None,
                          help="Dispositivo de evaluación; no cambia el protocolo del checkpoint.")
    evaluate.add_argument("--num-workers", type=int, default=None,
                          help="Workers solo para evaluación; no cambia el entrenamiento.")
    figures = commands.add_parser("figuras")
    figures.add_argument("--datos", type=Path, required=True)
    figures.add_argument("--split", type=Path, required=True)
    figures.add_argument("--salida", type=Path, required=True)
    compare = commands.add_parser("comparar")
    compare.add_argument("--runs", type=Path, required=True)
    compare.add_argument("--salida", type=Path, required=True)
    compare.add_argument("--subcarpeta-evaluacion", default=None,
                         help="Selecciona, por ejemplo, evaluacion o evaluacion_cpu sin mezclar mediciones.")
    args = parser.parse_args()
    if args.command == "preparar":
        from ham10000.prepare import preparar
        preparar(args.datos, args.salida, args.semilla)
    elif args.command == "validar":
        from ham10000.io import cargar_split, sha256
        rows = cargar_split(args.split, args.datos)
        for row in rows:
            if sha256(args.datos / row["relative_path"]) != row["sha256"]:
                raise ValueError(f"Imagen modificada: {row['image_id']}")
        print(f"{len(rows)} imágenes verificadas contra el split fijo; sin modificar la partición.")
    elif args.command == "exportar":
        from ham10000.prepare import exportar_carpetas
        exportar_carpetas(args.datos, args.split, args.salida, args.tamano, args.calidad)
    elif args.command == "entrenar":
        from ham10000.engine import entrenar
        entrenar(args.modelo, args.datos, args.split, args.config, args.salida, args.resume)
    elif args.command == "evaluar":
        from ham10000.engine import evaluar
        evaluar(args.checkpoint, args.datos, args.split, args.salida,
                args.particion, args.condicion, args.confirmar_test,
                mitigation=args.mitigacion, device_override=args.device, num_workers=args.num_workers)
    elif args.command == "figuras":
        from ham10000.plots import figuras_dataset
        figuras_dataset(args.datos, args.split, args.salida)
    elif args.command == "comparar":
        from ham10000.plots import comparar
        comparar(args.runs, args.salida, evaluation_subdir=args.subcarpeta_evaluacion)


if __name__ == "__main__":
    main()
