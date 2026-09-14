"""Audio I/O: read audio into NumPy arrays, write back out.

Uses `soundfile` (libsndfile), which supports this project's formats (MP3,
WAV, FLAC, OGG, AIFF, ...) *and* runs on modern Python (3.14), where the
legacy ``pydub``/``audioop`` stack no longer builds. Everything is represented
as a float32 numpy array shaped ``(n_samples, channels)`` at a shared sample
rate so the engine, analysis, and web serving all use one representation.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple
from math import gcd

import numpy as np
import soundfile as sf

TARGET_SR = 44100

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".caf",
    ".m4a", ".w64", ".wma", ".opus",
}


class AudioError(RuntimeError):
    """Generic audio error with a friendly message."""


def read(path: str, target_sr: int = TARGET_SR) -> np.ndarray:
    """Read an audio file into a float32 array ``(n_samples, channels)``.

    The result is resampled to ``target_sr``.
    """
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioError(
            f"Could not decode {os.path.basename(path)}: {exc}") from exc
    if data.ndim == 1:
        data = data[:, None]
    if sr != target_sr:
        data = resample(data, sr, target_sr)
    return np.ascontiguousarray(data, dtype=np.float32)


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample a ``(n, ch)`` array from ``sr_in`` to ``sr_out``."""
    if sr_in == sr_out or data.shape[0] == 0:
        return data
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    out_len = int(round(data.shape[0] * up / down))
    r = resample_poly(data, up, down, axis=0)
    if r.shape[0] > out_len:
        r = r[:out_len]
    return np.ascontiguousarray(r, dtype=np.float32)


def write(path: str, data: np.ndarray, sr: int = TARGET_SR,
          subtype: str = "PCM_16") -> None:
    """Write a ``(n, ch)`` float32 array to an audio file (format by ext)."""
    clip = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
    sf.write(path, clip, sr, subtype=subtype)


def duration_s(data: np.ndarray, sr: int = TARGET_SR) -> float:
    return len(data) / float(sr)


def to_int16(data: np.ndarray) -> np.ndarray:
    """Convert float [-1, 1] to int16 bytes for playback/transmission."""
    return (np.clip(data, -1, 1) * 32767).astype(np.int16)


def mono(data: np.ndarray) -> np.ndarray:
    """Return a mono (average of channels) array."""
    if data.ndim == 2 and data.shape[1] > 1:
        return data.mean(axis=1, keepdims=False)
    return data.reshape(-1)


def trim_silence_edges(data: np.ndarray, sr: int = TARGET_SR,
                       threshold_db: float = -40.0,
                       pad_ms: float = 100) -> Tuple[int, int]:
    """Return ``(start_idx, end_idx)`` that trim leading/trailing low energy."""
    import librosa

    m = mono(data)
    frame_len = max(1, int(0.025 * sr))
    hop = max(1, int(0.010 * sr))
    rms = librosa.feature.rms(y=m, frame_length=frame_len, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-12))
    voiced = db >= threshold_db
    if not voiced.any():
        return 0, len(data)
    first = int(np.argmax(voiced))
    last = len(voiced) - 1 - int(np.argmax(voiced[::-1]))
    pad = int(pad_ms / 1000.0 * sr)
    return max(0, first * hop - pad), min(len(data), (last + 1) * hop + pad)
