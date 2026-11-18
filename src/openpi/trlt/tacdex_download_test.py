"""Actual local HTTP range/resume/URL-refresh contract checks, CPU only."""

from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import threading

import pytest

from openpi.trlt import tacdex_download
from openpi.trlt.tacdex_source import write_json


def test_resumes_committed_ranges_and_refreshes_expired_url(tmp_path, monkeypatch):
    chunk = 65536
    monkeypatch.setattr(tacdex_download, "CHUNK_BYTES", chunk)
    monkeypatch.setattr(tacdex_download.time, "sleep", lambda _: None)
    content = bytes(range(256)) * 800 + b"last bytes"
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/expired":
                self.send_response(403)
                self.end_headers()
                return
            start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
            requests.append((start, end))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self.wfile.write(content[start : end + 1])

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    destination = tmp_path / "archive.zip"
    partial = destination.with_suffix(".zip.ranges.part")
    partial.write_bytes(content[:chunk])
    done = [True] + [False] * ((len(content) + chunk - 1) // chunk - 1)
    identity = {"total_bytes": len(content), "sha256": "expected-source-hash", "chunk_bytes": chunk}
    write_json(destination.with_suffix(".zip.ranges.json"), {"identity": identity, "done": done})
    calls = []

    def provider():
        calls.append(True)
        suffix = "expired" if len(calls) == 1 else "fresh"
        return f"http://127.0.0.1:{server.server_port}/{suffix}"

    try:
        tacdex_download.download_ranges(
            destination, provider, total_bytes=len(content), expected_sha256="expected-source-hash", workers=2
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert destination.read_bytes() == content
    assert not partial.exists()
    assert len(calls) >= 2
    assert all(start >= chunk for start, _ in requests)


def test_rejects_conflicting_resume_identity(tmp_path):
    destination = tmp_path / "archive.zip"
    write_json(destination.with_suffix(".zip.ranges.json"), {"identity": {}, "done": []})
    with pytest.raises(ValueError, match="provenance differs"):
        tacdex_download.download_ranges(destination, lambda: "unused", total_bytes=2, expected_sha256="new-hash")
