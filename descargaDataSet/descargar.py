"""Descarga HAM10000 con reanudación HTTP y verificación del ZIP."""

import argparse
import email.utils
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import time
import urllib.error
import urllib.request
import zipfile


URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "kmader/skin-cancer-mnist-ham10000?datasetVersionNumber=2"
)
BLOCK = 1024 * 1024


class DownloadError(Exception):
    """Error que requiere intervención, no más reintentos."""


def verificar(path):
    print("\nVerificando integridad CRC de todos los archivos del ZIP...", flush=True)
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist():
                raise DownloadError("El ZIP está vacío.")
            bad = archive.testzip()
            if bad:
                raise DownloadError(f"ZIP dañado: {bad}. Se conserva el archivo.")
    except zipfile.BadZipFile as exc:
        raise DownloadError("No es un ZIP válido. Se conserva para revisión.") from exc


def descargar(url, destino, timeout=30, verificar_zip=True):
    destino = Path(destino).resolve()
    destino.parent.mkdir(parents=True, exist_ok=True)
    partial = destino.with_name(destino.name + ".part")
    state_path = destino.with_name(destino.name + ".resume.json")
    lock_path = destino.with_name(destino.name + ".lock")
    try:
        lock = lock_path.open("x")
    except FileExistsError as exc:
        raise DownloadError(
            f"Descarga bloqueada por {lock_path}. No ejecutes dos procesos. "
            "Si el anterior terminó abruptamente, elimina SOLO ese .lock."
        ) from exc
    try:
        lock.write(str(os.getpid()))
        lock.close()
        if destino.exists():
            if verificar_zip:
                verificar(destino)
            print(f"Archivo existente; no se sobrescribe: {destino}")
            return destino
        state = {}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise DownloadError(f"Metadatos ilegibles: {state_path}") from exc
            if state.get("url") != url:
                raise DownloadError("La descarga parcial corresponde a otra URL.")
        if partial.exists() and partial.stat().st_size and not state:
            raise DownloadError("Archivo .part sin metadatos; usa otro destino.")
        failures = 0
        last_report = 0.0
        last_bytes = partial.stat().st_size if partial.exists() else 0
        last_time = time.monotonic()
        while True:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {
                "User-Agent": "HAM10000-Resumable-Downloader/1.0",
                "Accept-Encoding": "identity",
                "Cache-Control": "no-cache",
            }
            if offset:
                headers["Range"] = f"bytes={offset}-"
                validator = state.get("etag") or state.get("last_modified")
                if validator:
                    headers["If-Range"] = validator
            request = urllib.request.Request(url, headers=headers)
            retry_after = 0
            try:
                # Cada reintento usa la URL pública, no un enlace firmado vencido.
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    status = response.status
                    encoding = response.headers.get("Content-Encoding", "identity")
                    if encoding.lower() != "identity":
                        raise DownloadError("Compresión HTTP inesperada; no se mezcla con el parcial.")
                    content_type = response.headers.get("Content-Type", "").lower()
                    if "text/html" in content_type or "application/json" in content_type:
                        raise DownloadError("Kaggle devolvió una página o un error, no el ZIP.")
                    if status == 206:
                        match = re.fullmatch(
                            r"bytes (\d+)-(\d+)/(\d+)",
                            response.headers.get("Content-Range", ""),
                        )
                        if not match:
                            raise DownloadError("Content-Range ausente o inválido.")
                        start, end, total = map(int, match.groups())
                        if start != offset or end < start or end >= total:
                            raise DownloadError("El servidor devolvió un rango incompatible.")
                        old_etag = state.get("etag")
                        new_etag = response.headers.get("ETag")
                        if offset and old_etag and new_etag and old_etag != new_etag:
                            raise DownloadError("El archivo remoto cambió; usa otro destino.")
                        if offset and state.get("total") not in (None, total):
                            raise DownloadError("El tamaño remoto cambió; usa otro destino.")
                        expected = end - start + 1
                        mode = "ab" if offset else "wb"
                    elif status == 200:
                        length = response.headers.get("Content-Length")
                        if length is None or not length.isdigit() or int(length) <= 0:
                            raise DownloadError("El servidor no indicó un tamaño válido.")
                        total = expected = int(length)
                        if offset:
                            print("\nServidor ignoró Range o cambió el archivo. Reinicio seguro.")
                        offset = 0
                        mode = "wb"
                    else:
                        raise DownloadError(f"Respuesta HTTP inesperada: {status}")
                    etag = response.headers.get("ETag")
                    if etag and etag.startswith("W/"):
                        etag = None
                    new_state = {
                        "url": url,
                        "total": total,
                        "etag": etag,
                        "last_modified": response.headers.get("Last-Modified"),
                    }
                    free = shutil.disk_usage(destino.parent).free
                    if free < total - offset + 16 * BLOCK:
                        raise DownloadError("Espacio insuficiente para completar la descarga.")
                    received = 0
                    with partial.open(mode) as output:
                        # Truncar antes de cambiar metadatos evita mezclar versiones tras un corte.
                        state = new_state
                        temp_state = state_path.with_name(state_path.name + ".tmp")
                        temp_state.write_text(json.dumps(state), encoding="utf-8")
                        temp_state.replace(state_path)
                        try:
                            while True:
                                chunk = response.read(BLOCK)
                                if not chunk:
                                    break
                                if received + len(chunk) > expected:
                                    raise DownloadError("Respuesta mayor que el rango anunciado.")
                                output.write(chunk)
                                received += len(chunk)
                                failures = 0
                                now = time.monotonic()
                                current = offset + received
                                if now - last_report >= 1:
                                    speed = max(0, current - last_bytes) / max(now - last_time, 0.001)
                                    print(
                                        f"\r{100 * current / total:6.2f}% | "
                                        f"{current / BLOCK:,.1f}/{total / BLOCK:,.1f} MiB | "
                                        f"{speed / BLOCK:.2f} MiB/s       ",
                                        end="", flush=True,
                                    )
                                    last_report = last_time = now
                                    last_bytes = current
                        finally:
                            output.flush()
                            os.fsync(output.fileno())
                    if received != expected:
                        raise ConnectionError("Conexión cerrada antes de completar el rango.")
                    if partial.stat().st_size == total:
                        break
                    continue
            except urllib.error.HTTPError as exc:
                if exc.code == 416:
                    match = re.fullmatch(r"bytes \*/(\d+)", exc.headers.get("Content-Range", ""))
                    if match and offset == int(match.group(1)) and state.get("total") == offset:
                        exc.close()
                        break
                    exc.close()
                    raise DownloadError("Rango no válido. Conserva el parcial y revisa el destino.")
                if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                    exc.close()
                    raise DownloadError(f"HTTP {exc.code}. Revisa acceso público a Kaggle.") from exc
                value = exc.headers.get("Retry-After", "")
                try:
                    retry_after = max(0, int(value))
                except ValueError:
                    try:
                        retry_after = max(0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
                    except (ValueError, TypeError, OverflowError):
                        pass
                exc.close()
                reason = f"HTTP {exc.code}"
            except (urllib.error.URLError, ConnectionError, TimeoutError,
                    socket.timeout, http.client.HTTPException) as exc:
                reason = str(exc)
            failures += 1
            delay = max(retry_after, min(30, 2 ** min(failures, 5)))
            print(f"\nConexión interrumpida: {reason}. Reintento en {delay:.0f} s. Parcial conservado.", flush=True)
            time.sleep(delay)
        if verificar_zip:
            verificar(partial)
        partial.replace(destino)
        state_path.unlink(missing_ok=True)
        print(f"\nDescarga completa: {destino}")
        return destino
    finally:
        lock.close()
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destino", type=Path,
        default=Path(__file__).resolve().parent / "HAM10000.zip",
        help="ZIP de salida; por defecto junto al script.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Espera máxima de red en segundos.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout debe ser mayor que cero")
    try:
        descargar(URL, args.destino, timeout=args.timeout)
    except KeyboardInterrupt:
        print("\nPausado. Ejecuta el mismo comando para continuar.")
        return 130
    except (DownloadError, OSError, zipfile.BadZipFile) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
