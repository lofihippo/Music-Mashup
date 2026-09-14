"""Web server for Music Mashup.

A small stdlib-only HTTP server that:

* serves the static web UI,
* accepts uploaded audio (stored per-session under ``uploads/``),
* runs analysis (silence / beats / BPM),
* lets the client edit splice points, crossfade, gain and fades,
* renders the final mix back to an audio file and streams it.

No framework dependency: uses ``http.server`` and ``ThreadingHTTPServer``.
"""

from __future__ import annotations

import json
import os
import time
import uuid
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import analysis, audio, splice
from .model import Collection, Clip

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
WORK_DIR = BASE_DIR / "work"
UPLOAD_DIR = WORK_DIR / "uploads"
OUTPUT_DIR = WORK_DIR / "outputs"

_sample_rate = audio.TARGET_SR
_lock = threading.Lock()
_sessions: dict = {}  # session_id -> {"uploads": {id: bytes}, ...}


class BadJSONError(ValueError):
    """Raised for a malformed JSON request body."""


def _ensure_dirs() -> None:
    for d in (WORK_DIR, UPLOAD_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _now_s() -> float:
    return time.time()


class Handler(BaseHTTPRequestHandler):
    server_version = "MusicMashup/0.2"

    # ---- helpers ---------------------------------------------------------
    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise BadJSONError(f"malformed JSON body: {exc}") from exc

    def _session(self) -> str:
        sid = self.headers.get("X-Session-Id") or ""
        # allow ?session= on query for audio streaming
        if not sid:
            from urllib.parse import urlparse, parse_qs

            q = parse_qs(urlparse(self.path).query)
            sid = (q.get("session") or [""])[0]
        return sid

    def log_message(self, fmt, *args):  # quieter logs
        return

    # ---- handlers --------------------------------------------------------
    def do_POST(self):
        path = self.path.split("?")[0].split("#")[0]

        if path == "/upload":
            self._handle_upload()
            return
        if path == "/analysis":
            self._handle_analysis()
            return
        if path == "/render":
            self._handle_render()
            return
        self._send_json({"error": "not found"}, 404)

    def do_GET(self):
        path = self.path.split("?")[0].split("#")[0]

        if path in ("/", "/index.html", "/static/index.html"):
            self._serve_file(STATIC_DIR / "index.html", "text/html")
            return
        if path == "/api/health":
            self._send_json({
                "ok": True,
                "analysis_available": analysis.has_analysis(),
                "sample_rate": _sample_rate,
                "backend": "numpy/soundfile",
            })
            return
        if path.startswith("/audio/"):
            self._serve_output(path)
            return
        # static assets
        rel = path.lstrip("/")
        candidate = (STATIC_DIR / rel).resolve()
        if str(candidate).startswith(str(STATIC_DIR.resolve())):
            ctype = "application/javascript" if candidate.suffix == ".js" \
                else "text/css" if candidate.suffix == ".css" \
                else "text/html" if candidate.suffix == ".html" else "application/octet-stream"
            self._serve_file(candidate, ctype)
            return
        self._send_json({"error": "not found"}, 404)

    # ---- concrete handlers ----------------------------------------------
    MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB per request

    def _handle_upload(self):
        ctype = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", 0) or 0)
        if not ctype.startswith("multipart/form-data") or length <= 0:
            self._send_json({"error": "expected multipart upload"}, 400)
            return
        if length > self.MAX_UPLOAD_BYTES:
            self._send_json({"error": "upload too large"}, 413)
            return
        body = self.rfile.read(length)
        boundary = ctype.split("boundary=", 1)[1].strip('"').encode("ascii")
        parts = self._parse_multipart(body, boundary)
        sid = self._session()
        _sessions.setdefault(sid, {"uploads": {}})

        result = []
        max_total = self.MAX_UPLOAD_BYTES
        total = 0
        for name, fname, data in parts:
            if not data:
                continue
            total += len(data)
            if total > max_total:
                self._send_json({"error": "upload too large"}, 413)
                return
            suffix = os.path.splitext(fname)[1].lower()
            if suffix not in audio.SUPPORTED_EXTENSIONS:
                result.append({"name": fname, "error": f"unsupported type {suffix}"})
                continue
            # Use a server-generated id as the storage key; never trust the
            # client filename for filesystem paths or dict keys.
            upload_id = uuid.uuid4().hex
            display_name = os.path.basename(fname) or "audio"
            with _lock:
                _sessions[sid]["uploads"][upload_id] = data
            out = UPLOAD_DIR / f"{upload_id}{suffix}"
            out.write_bytes(data)
            result.append({"id": upload_id, "name": display_name,
                           "size": len(data)})
        self._send_json({"uploads": result, "session": sid})

    def _parse_multipart(self, body: bytes, boundary: bytes):
        parts = []
        delim = b"--" + boundary
        sections = body.split(delim)
        for sec in sections:
            if not sec.startswith(b"\r\n"):
                continue
            head, _, payload = sec[2:].partition(b"\r\n\r\n")
            payload = payload.rstrip(b"\r\n")
            name = ""
            fname = ""
            for line in head.split(b"\r\n"):
                line_s = line.decode("latin1", "ignore")
                if "name=" in line_s:
                    name = line_s.split('name="', 1)[1].split('"', 1)[0]
                if "filename=" in line_s:
                    fname = line_s.split('filename="', 1)[1].split('"', 1)[0]
            parts.append((name, fname, payload))
        return parts

    def _handle_analysis(self):
        try:
            payload = self._read_json()
        except BadJSONError as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        sid = self._session()
        upload_id = payload.get("id", "")
        uploads = _sessions.get(sid, {}).get("uploads", {})
        if not upload_id or upload_id not in uploads:
            self._send_json({"error": "unknown upload"}, 404)
            return
        data = uploads[upload_id]
        result = self._analyze(data)
        result["id"] = upload_id
        self._send_json(result)

    def _analyze(self, raw: bytes) -> dict:
        # Write to temp file so soundfile can decode by header.
        tmp = WORK_DIR / f"tmp_{uuid.uuid4().hex}.{_pick_ext(raw)}"
        tmp.write_bytes(raw)
        try:
            arr = audio.read(str(tmp), _sample_rate)
        finally:
            tmp.unlink(missing_ok=True)
        info = analysis.analyze(arr, _sample_rate)
        info["duration_s"] = round(len(arr) / _sample_rate, 3)
        return info

    def _handle_render(self):
        try:
            payload = self._read_json()
        except BadJSONError as exc:
            self._send_json({"error": str(exc)}, 400)
            return
        sid = self._session()
        uploads = _sessions.get(sid, {}).get("uploads", {})
        clips = payload.get("clips", [])
        detect = payload.get("detect", "off")
        # rebuild clip objects with explicit source names and cut points
        collection = Collection(name=payload.get("name", "Mix"))
        for cdata in clips:
            collection.add(Clip.from_dict(cdata))

        # load sources by upload id (enforced per session)
        source_index = {}
        loaded = {}
        for cdata in clips:
            src = cdata.get("source", "")
            if src in loaded or src not in uploads:
                continue
            tmp = WORK_DIR / f"tmp_{uuid.uuid4().hex}.{_pick_ext(uploads[src])}"
            tmp.write_bytes(uploads[src])
            try:
                loaded[src] = audio.read(str(tmp), _sample_rate)
            finally:
                tmp.unlink(missing_ok=True)
            source_index[src] = loaded[src]

        # fill unset bounds from detection for auto-splice modes
        if detect in ("silence", "beat"):
            for clip in collection.clips:
                if clip.is_explicit() or clip.source not in source_index:
                    continue
                auto = splice.auto_splice_clip(
                    clip, source_index[clip.source], sr=_sample_rate,
                    detect=detect)
                if auto.detected_start is not None:
                    clip.start = auto.detected_start
                if auto.detected_end is not None:
                    clip.end = auto.detected_end
                if not clip.is_explicit():
                    # nothing detected; use the whole file
                    clip.start = 0.0
                    clip.end = audio.duration_s(source_index[clip.source], _sample_rate)

        crossfade_s = float(payload.get("crossfade_s", 0.0))
        normalize = bool(payload.get("normalize", False))
        target_lufs = float(payload.get("target_lufs", -16.0))

        if not source_index:
            self._send_json({"error": "no sources loaded"}, 400)
            return

        out = splice.render(collection, source_index, sr=_sample_rate,
                            normalize=normalize, target_lufs=target_lufs,
                            crossfade_s=crossfade_s)

        fname = f"mix_{uuid.uuid4().hex}.wav"
        out_path = OUTPUT_DIR / fname
        audio.write(str(out_path), out, _sample_rate)

        # also make a small int16 preview for <audio>
        # (audio element can play wav directly)
        meta = {
            "name": collection.name,
            "file": fname,
            "duration_s": round(len(out) / _sample_rate, 3),
            "channels": int(out.shape[1]) if out.ndim > 1 else 1,
        }
        self._send_json(meta)

    def _serve_output(self, path: str) -> None:
        fname = path.split("/audio/", 1)[1].split("?")[0]
        fp = OUTPUT_DIR / os.path.basename(fname)
        if not fp.exists():
            self._send_json({"error": "not found"}, 404)
            return
        self._serve_file(fp, "audio/wav")

    def _serve_file(self, fp: Path, ctype: str) -> None:
        if not fp.exists():
            self._send_json({"error": "not found"}, 404)
            return
        data = fp.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


def _pick_ext(raw: bytes) -> str:
    if raw[:4] == b"ID3" or raw[:3] == b"\xff\xfb" or raw[:3] == b"\xff\xf3":
        return "mp3"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "wav"
    if raw[:4] == b"fLaC":
        return "flac"
    if raw[:4] == b"OggS":
        return "ogg"
    if raw[:4] == b"\x00\x00\x00\x18" and raw[8:12] == b"ftyp":
        return "m4a"
    return "wav"


def run(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    _ensure_dirs()
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Music Mashup serving at {url}  (Ctrl-C to stop)")
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


def main():
    import argparse

    p = argparse.ArgumentParser(description="Music Mashup web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    run(args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
