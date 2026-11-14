"""Crea un ZIP portable del proyecto sin datos originales ni checkpoints."""

from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root.parent / "proyecto_ham10000_colab.zip"
    temp = destination.with_suffix(".zip.tmp")
    ignored_suffixes = {".aux", ".log", ".toc", ".lof", ".lot", ".out", ".bbl", ".blg", ".pyc", ".pt", ".pth", ".nav", ".snm", ".vrb"}
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob("*")):
            relative = file.relative_to(root)
            if not file.is_file() or any(part in {"__pycache__", ".venv", ".ipynb_checkpoints"} for part in relative.parts):
                continue
            if relative.parts[:2] == ("data", "raw") and file.name != ".gitkeep":
                continue
            if relative.parts[0] in {"runs", "resultados_notebook"} or file.suffix in ignored_suffixes:
                continue
            # Solo el resumen publicado; no confundirlo con una nueva ejecución en Colab.
            if relative.parts[0] == "resultados" and relative.as_posix() != "resultados/comparacion.csv":
                continue
            if file.name.startswith("evidencias_notebook_") or file.name.endswith(".tmp"):
                continue
            if relative.parts[:3] in {("data", "processed", "train"), ("data", "processed", "val"), ("data", "processed", "test")}:
                continue
            archive.write(file, (Path(root.name) / relative).as_posix())
    with zipfile.ZipFile(temp) as archive:
        if archive.testzip():
            raise RuntimeError("El ZIP generado no superó la comprobación CRC.")
    temp.replace(destination)
    print(f"Código portable: {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")
    print("El dataset y la .venv de Windows NO están incluidos.")


if __name__ == "__main__":
    main()
