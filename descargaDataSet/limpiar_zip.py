"""Elimina carpetas de imágenes duplicadas por mayúsculas/minúsculas en el ZIP de HAM10000."""

import argparse
import collections
import os
from pathlib import Path
import sys
import zipfile


def limpiar(origen, destino):
    origen = Path(origen).resolve()
    destino = Path(destino).resolve()
    if not origen.is_file():
        raise FileNotFoundError(origen)
    if destino.exists():
        raise FileExistsError(f"El destino ya existe: {destino}")
    temp = destino.with_name(destino.name + ".tmp")

    with zipfile.ZipFile(origen) as zin:
        infos = zin.infolist()
        grupos = collections.defaultdict(list)
        for info in infos:
            if info.filename.lower().endswith(".jpg"):
                grupos[os.path.basename(info.filename).lower()].append(info)
        if not grupos:
            raise ValueError("El ZIP no contiene JPG.")

        for clave, grupo in grupos.items():
            if len(grupo) != 2:
                raise ValueError(f"{clave}: se esperaban 2 copias, hay {len(grupo)}.")
            if len({(i.CRC, i.file_size) for i in grupo}) != 1:
                raise ValueError(f"{clave}: las copias no son idénticas.")

        conservar = set()
        for info in infos:
            if info.filename.lower().endswith(".jpg"):
                carpeta = info.filename.split("/")[0]
                if not carpeta.startswith("HAM10000_"):
                    continue
                conservar.add(info.filename)

        print(f"Entradas originales: {len(infos)}")
        print(f"JPG únicos: {len(grupos)}")
        print(f"JPG a conservar: {len(conservar)}")

        with zipfile.ZipFile(
            temp, "w", compression=zipfile.ZIP_STORED, allowZip64=True
        ) as zout:
            escritos = 0
            for info in infos:
                es_jpg = info.filename.lower().endswith(".jpg")
                if es_jpg and info.filename not in conservar:
                    continue
                datos = zin.read(info)
                zout.writestr(info, datos, compress_type=zipfile.ZIP_STORED)
                escritos += 1
                if escritos % 1000 == 0:
                    print(f"  {escritos} archivos...", flush=True)

        print(f"Entradas escritas: {escritos}")
        print("Verificando CRC...", flush=True)
        with zipfile.ZipFile(temp) as check:
            if len(check.infolist()) != escritos:
                raise ValueError("El número de entradas no coincide.")
            malo = check.testzip()
            if malo:
                raise ValueError(f"CRC incorrecto: {malo}")

    temp.replace(destino)
    print(f"ZIP limpio: {destino}")
    print(f"Tamaño: {destino.stat().st_size / 1073741824:.3f} GiB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("origen", type=Path)
    parser.add_argument("destino", type=Path)
    args = parser.parse_args()
    try:
        limpiar(args.origen, args.destino)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        Path(str(args.destino) + ".tmp").unlink(missing_ok=True)
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
