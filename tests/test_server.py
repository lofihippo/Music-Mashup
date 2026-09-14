"""Integration tests for the HTTP server endpoints (upload/analysis/render)."""

import io
import json
import threading
import urllib.request
import urllib.error
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import numpy as np
import soundfile as sf

from musicmashup import server


@pytest.fixture
def server_client(tmp_path, monkeypatch):
    """Run the handler on an ephemeral port; monkeypatch work dirs to tmp."""
    # Isolate storage so tests don't pollute the repo's work/ dir.
    monkeypatch.setattr(server, "WORK_DIR", Path(tmp_path))
    monkeypatch.setattr(server, "UPLOAD_DIR", Path(tmp_path) / "uploads")
    monkeypatch.setattr(server, "OUTPUT_DIR", Path(tmp_path) / "outputs")
    server._ensure_dirs()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base
    finally:
        httpd.shutdown()
        httpd.server_close()


def _tone_wav() -> bytes:
    sr = 44100
    t = np.arange(int(1.0 * sr)) / sr
    data = np.repeat((0.3 * np.sin(2 * np.pi * 440 * t))[:, None], 2, axis=1)
    buf = io.BytesIO()
    sf.write(buf, data.astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def _multipart(field, filename, content):
    boundary = "XBOUNDARY"
    parts = []
    parts.append(f"--{boundary}\r\n")
    parts.append(f'Content-Disposition: form-data; name="{field}"; '
                 f'filename="{filename}"\r\n')
    parts.append("Content-Type: application/octet-stream\r\n\r\n")
    body = "".join(parts).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    return body, boundary


def _upload(host, content, filename="a.wav", sid="sess"):
    body, boundary = _multipart("files", filename, content)
    req = urllib.request.Request(
        f"{host}/upload", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "X-Session-Id": sid})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_health(server_client):
    with urllib.request.urlopen(server_client + "/api/health") as r:
        data = json.loads(r.read())
    assert data["ok"] is True
    assert data["backend"] == "numpy/soundfile"


def test_upload_and_analysis(server_client):
    wav = _tone_wav()
    res = _upload(server_client, wav, "song.wav", sid="s1")
    assert res["uploads"][0]["name"] == "song.wav"
    up_id = res["uploads"][0]["id"]
    # analysis by id
    req = urllib.request.Request(
        f"{server_client}/analysis", data=json.dumps({"id": up_id}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-Session-Id": "s1"})
    with urllib.request.urlopen(req) as r:
        info = json.loads(r.read())
    assert info["duration_s"] == pytest.approx(1.0, abs=0.05)


def test_upload_rejects_path_traversal_filename(server_client):
    wav = _tone_wav()
    # malicious filename with path separators
    res = _upload(server_client, wav, "../../evil.wav", sid="s2")
    up_id = res["uploads"][0]["id"]
    # must be a plain hex id, and no files written outside upload dir
    assert up_id.isalnum()
    assert len(up_id) == 32
    written = list(server.UPLOAD_DIR.iterdir())
    assert all(not p.name.startswith("..") for p in written)
    # the malicious name is preserved only as display name
    assert res["uploads"][0]["name"] == "evil.wav"


def test_upload_duplicate_id_not_overwriting(server_client):
    wav = _tone_wav()
    a = _upload(server_client, wav, "same.wav", sid="s3")
    b = _upload(server_client, wav, "same.wav", sid="s3")
    assert a["uploads"][0]["id"] != b["uploads"][0]["id"]


def test_analysis_requires_matching_session(server_client):
    wav = _tone_wav()
    res = _upload(server_client, wav, "song.wav", sid="owner")
    up_id = res["uploads"][0]["id"]
    req = urllib.request.Request(
        f"{server_client}/analysis", data=json.dumps({"id": up_id}).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-Session-Id": "intruder"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req)
    assert ei.value.code == 404


def test_upload_size_cap(server_client):
    # Header content-length > cap should be rejected without reading.
    big_len = server.Handler.MAX_UPLOAD_BYTES + 1
    req = urllib.request.Request(
        f"{server_client}/upload", data=b"x" * 16, method="POST",
        headers={"Content-Type": "multipart/form-data; boundary=XB",
                 "Content-Length": str(big_len), "X-Session-Id": "s4"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req)
    assert ei.value.code == 413


def test_malformed_json_returns_400(server_client):
    req = urllib.request.Request(
        f"{server_client}/analysis", data=b"{not-json", method="POST",
        headers={"Content-Type": "application/json", "X-Session-Id": "s5"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req)
    assert ei.value.code == 400


def test_render_roundtrip(server_client):
    wav = _tone_wav()
    res = _upload(server_client, wav, "a.wav", sid="s6")
    up_id = res["uploads"][0]["id"]
    payload = {"name": "Mix", "crossfade_s": 0.0, "detect": "off",
               "clips": [{"source": up_id}]}
    req = urllib.request.Request(
        f"{server_client}/render", data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "X-Session-Id": "s6"})
    with urllib.request.urlopen(req) as r:
        meta = json.loads(r.read())
    assert meta["duration_s"] == pytest.approx(1.0, abs=0.05)
    # fetch the rendered audio
    with urllib.request.urlopen(server_client + "/audio/" + meta["file"]) as r:
        audio_bytes = r.read()
    assert len(audio_bytes) > 44  # WAV header + samples
