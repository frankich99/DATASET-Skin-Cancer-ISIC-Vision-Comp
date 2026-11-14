"""Compila el informe ampliado y el borrador IEEE; comprueba al menos 25 páginas."""

import re
import shutil
import subprocess
import sys
from pathlib import Path


def ejecutar(command, cwd):
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", timeout=180)
    if result.returncode:
        print(result.stdout[-10000:])
        raise RuntimeError(f"Falló: {' '.join(command)}")
    return result.stdout


def main():
    root = Path(__file__).resolve().parent
    pdflatex, bibtex = shutil.which("pdflatex"), shutil.which("bibtex")
    if not pdflatex or not bibtex:
        raise RuntimeError("Se requiere MiKTeX/TeX Live con pdflatex y bibtex. Alternativa: subir latex a Overleaf.")
    ejecutar([sys.executable, str(root / "generar_tablas.py")], root)
    pages = {}
    for name in ("main", "paper_ieee"):
        command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", name + ".tex"]
        ejecutar(command, root)
        ejecutar([bibtex, name], root)
        ejecutar(command, root)
        log = ejecutar(command, root)
        match = re.search(r"Output written on .*?\((\d+) pages?", log, re.S)
        if not match:
            raise RuntimeError(f"No se pudo comprobar el número de páginas de {name}.")
        pages[name] = int(match.group(1))
        print(f"{name}.pdf: {pages[name]} páginas")
    if pages["main"] < 25:
        raise RuntimeError("El informe todavía no alcanza las 25 páginas solicitadas; revise contenido y compilación.")
    print("Compilación completa: informe técnico ampliado y artículo IEEE actualizados con resultados reales.")


if __name__ == "__main__":
    main()
