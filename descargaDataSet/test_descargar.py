"""Pruebas sin Internet para la descarga reanudable."""

from contextlib import contextmanager, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from descargar import BLOCK, DownloadError, descargar


@contextmanager
def servidor(payload, *, cortar=False, ignorar_range=False, html=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def do_GET(self):
            rango = self.headers.get("Range")
            requests.append(rango)
            offset = int(rango.split("=")[1].split("-")[0]) if rango else 0
            if offset and not ignorar_range:
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {offset}-{len(payload)-1}/{len(payload)}")
            else:
                offset = 0
                self.send_response(200)
            self.send_header("Content-Type", "text/html" if html else "application/zip")
            self.send_header("ETag", '"test-v1"')
            self.send_header("Content-Length", str(len(payload) - offset))
            self.end_headers()
            if cortar and len(requests) == 1:
                self.wfile.write(payload[:BLOCK + 123])
            else:
                self.wfile.write(payload[offset:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/dataset.zip", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class DescargaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("imagen.bin", b"a" * (BLOCK * 2 + 100))
        cls.payload = buffer.getvalue()

    def test_corte_y_reanudacion(self):
        with tempfile.TemporaryDirectory() as tmp, servidor(self.payload, cortar=True) as (url, requests):
            path = Path(tmp) / "datos.zip"
            with patch("descargar.time.sleep"), redirect_stdout(io.StringIO()):
                descargar(url, path)
            self.assertEqual(path.read_bytes(), self.payload)
            self.assertTrue(any(value and value.startswith("bytes=") for value in requests))
            self.assertFalse(path.with_name("datos.zip.part").exists())
            self.assertFalse(path.with_name("datos.zip.lock").exists())

    def test_reabrir_script_y_servidor_ignora_range(self):
        with tempfile.TemporaryDirectory() as tmp, servidor(self.payload, ignorar_range=True) as (url, requests):
            import json
            path = Path(tmp) / "datos.zip"
            path.with_name("datos.zip.part").write_bytes(self.payload[:1000])
            path.with_name("datos.zip.resume.json").write_text(json.dumps({
                "url": url, "total": len(self.payload), "etag": '"test-v1"',
            }), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                descargar(url, path)
            self.assertEqual(requests[0], "bytes=1000-")
            self.assertEqual(path.read_bytes(), self.payload)

    def test_rechaza_pagina_html(self):
        with tempfile.TemporaryDirectory() as tmp, servidor(b"<html>login</html>", html=True) as (url, _):
            path = Path(tmp) / "datos.zip"
            with self.assertRaises(DownloadError):
                descargar(url, path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_name("datos.zip.lock").exists())

    def test_zip_existente_no_sobrescribe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "datos.zip"
            path.write_bytes(self.payload)
            with patch("descargar.urllib.request.urlopen") as request, redirect_stdout(io.StringIO()):
                descargar("http://no-usar.example/", path)
            request.assert_not_called()
            self.assertEqual(path.read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
