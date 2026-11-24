"""
Test the resumable download of retrieve_http_file against a local HTTP server.

The server honours ``Range`` requests (206 with ``Content-Range``, 416 past the
end) and can close the connection halfway through its first responses or ignore
``Range`` altogether.
"""

import os
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from retrieve_http_file import download

PAYLOAD = bytes(range(256)) * 4000  # ~1 MB


class Handler(BaseHTTPRequestHandler):
    """Serves PAYLOAD at /file with Range support; the server's flags make it misbehave."""

    def do_GET(self):
        """Answer a GET, honouring Range unless ignore_range, hanging up halfway while drops > 0."""
        server = self.server
        server.ranges.append(self.headers.get("Range"))
        if self.path != "/file":
            self.send_error(404)
            return
        start = 0
        requested = self.headers.get("Range")
        if requested and not server.ignore_range:
            start = int(requested.split("=")[1].split("-")[0])
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(PAYLOAD)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if server.drops > 0:  # announce the whole body, send half, hang up
            server.drops -= 1
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, *args):
        """Keep the test output quiet."""


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.ranges, srv.drops, srv.ignore_range = [], 0, False
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def url(srv, path="/file"):
    return f"http://127.0.0.1:{srv.server_address[1]}{path}"


def fetch(srv, out, **kwargs):
    return download(url(srv), out, backoff=0, timeout=5, chunk=4096, **kwargs)


def test_plain_download(server, tmp_path):
    out = fetch(server, tmp_path / "data" / "file.bin")
    assert out.read_bytes() == PAYLOAD
    assert server.ranges == [None]
    assert not (tmp_path / "data" / "file.bin.part").exists()


def test_resumes_after_broken_connections(server, tmp_path):
    server.drops = 2
    out = fetch(server, tmp_path / "file.bin")
    assert out.read_bytes() == PAYLOAD
    # first request full, then two resumes from the bytes already received
    half = len(PAYLOAD) // 2
    quarter = half + (len(PAYLOAD) - half) // 2
    assert server.ranges == [None, f"bytes={half}-", f"bytes={quarter}-"]


def test_starts_over_when_range_is_ignored(server, tmp_path):
    server.drops, server.ignore_range = 1, True
    out = fetch(server, tmp_path / "file.bin")
    assert out.read_bytes() == PAYLOAD
    assert server.ranges == [None, f"bytes={len(PAYLOAD) // 2}-"]


def test_complete_part_file_is_kept(server, tmp_path):
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD)  # finished, but not renamed by an interrupted run
    out = fetch(server, tmp_path / "file.bin")
    assert out.read_bytes() == PAYLOAD
    assert server.ranges == [f"bytes={len(PAYLOAD)}-"]


def test_oversized_part_file_starts_over(server, tmp_path):
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD + b"junk")
    out = fetch(server, tmp_path / "file.bin")
    assert out.read_bytes() == PAYLOAD
    assert server.ranges == [f"bytes={len(PAYLOAD) + 4}-", None]


def test_missing_file_fails_without_output(server, tmp_path):
    with pytest.raises(urllib.error.HTTPError):
        download(url(server, "/missing"), tmp_path / "file.bin", backoff=0, timeout=5)
    assert not (tmp_path / "file.bin").exists()
    assert server.ranges == [None]  # a 404 is not retried


def test_gives_up_after_attempts(server, tmp_path):
    server.drops = 5
    with pytest.raises(RuntimeError, match="in 3 attempts"):
        fetch(server, tmp_path / "file.bin", attempts=3)
    assert not (tmp_path / "file.bin").exists()
    assert (tmp_path / "file.bin.part").exists()  # kept for the next run to resume
