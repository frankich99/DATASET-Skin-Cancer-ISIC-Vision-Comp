"""Valida el cuaderno; opcionalmente lo ejecuta localmente sin operaciones experimentales."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile


def verificar(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if (notebook["nbformat"], notebook["nbformat_minor"]) != (4, 5):
        raise ValueError("Formato de notebook inesperado")
    ids = [c["id"] for c in notebook["cells"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Identificadores repetidos")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=cell["id"])
            if cell["outputs"] or cell["execution_count"] is not None:
                raise ValueError("El notebook distribuido debe estar sin salidas ni ejecuciones")
    controls = next(c for c in notebook["cells"] if c["id"] == "controles")
    values = {}
    for node in ast.parse("".join(controls["source"])).body:
        if isinstance(node, ast.Assign):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    for flag in ("ENTRENAR", "EVALUAR", "CONFIRMAR_TEST", "GENERAR_FIGURAS", "EXPORTAR_EVIDENCIAS", "INSTALAR_COMUNES"):
        if values[flag] is not False:
            raise ValueError(f"El control seguro {flag} no está desactivado")
    if values["RUTA_PROYECTO"] or values["RUTA_DATOS"] or values["RUTA_RUNS"]:
        raise ValueError("El cuaderno distribuido no debe contener rutas personales activas")
    print(f"JSON, estructura, AST y controles seguros: {len(notebook['cells'])} celdas")
    return notebook


def snapshot(project):
    files = list((project / "runs").rglob("*")) + list((project / "resultados").rglob("*"))
    values = {}
    for p in files:
        if p.is_file() and p.suffix in (".pt", ".csv", ".json"):
            with p.open("rb") as stream:
                values[str(p.relative_to(project))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ejecutar-local", action="store_true")
    parser.add_argument("--salida", type=Path, help="Directorio externo para notebook ejecutado/reporte")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    file = root / "HAM10000_Grupo09.ipynb"
    verificar(file)
    try:
        import nbformat
    except ImportError:
        if args.ejecutar_local:
            raise RuntimeError("Instale python/requirements-notebook.txt con el mismo intérprete.")
        print("nbformat no instalado: validación del esquema oficial pendiente; AST/JSON comprobados.")
        return
    nb = nbformat.read(file, as_version=4)
    nbformat.validate(nb)
    print("Esquema oficial nbformat: válido")
    if not args.ejecutar_local:
        return
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    project = root.parent
    before = snapshot(project)
    output = args.salida.resolve() if args.salida else Path(tempfile.mkdtemp(prefix="ham10000_notebook_"))
    if output.is_relative_to(project):
        raise ValueError("Use un directorio externo al proyecto para las salidas de verificación.")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kernel_", dir=output) as tmp:
        kernels = Path(tmp)
        kernel = kernels / "ham10000-verificacion"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "HAM10000 verificación aislada", "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(kernel_name=kernel.name, kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
        client = NotebookClient(nb, km=manager, timeout=900, allow_errors=False,
                                resources={"metadata": {"path": str(root)}})
        try:
            client.execute()
        finally:
            nbformat.write(nb, output / "HAM10000_local_ejecutado.ipynb")
    after = snapshot(project)
    if before != after:
        raise RuntimeError("La inspección modificó artefactos experimentales; revise sus hashes.")
    count = sum(c.cell_type == "code" for c in nb.cells)
    report = dict(status="ok", code_cells=count, interpreter=sys.executable, runtime="local",
                  experimental_artifacts_unchanged=len(before),
                  cloud_execution_verified=False, notebook_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (output / "verificacion.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{count} celdas de código ejecutadas localmente; {len(before)} artefactos sin cambios. Salida: {output}")


if __name__ == "__main__":
    main()
