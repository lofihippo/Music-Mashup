"""Audio analysis: silence and beat detection for automatic splice points.

These helpers operate on the numpy float32 arrays produced by :mod:`.audio`.
They degrade gracefully when analysis deps are missing, returning conservative
defaults so the engine still works with explicit in/out points.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def has_analysis() -> bool:
    try:
        import librosa  # noqa: F401
        return True
    except Exception:
        return False


def _require_librosa():
    import librosa  # noqa: F401
    return librosa


def detect_silence(data: np.ndarray, sr: int,
                   threshold_db: float = -40.0,
                   min_silence_ms: int = 500) -> List[Tuple[float, float]]:
    """Return list of (start_s, end_s) silence spans in ``data``."""
    if not has_analysis() or data is None or len(data) == 0:
        return []
    librosa = _require_librosa()
    mono = audio_mono(data)
    frame_len = max(1, int(0.025 * sr))
    hop = max(1, int(0.010 * sr))
    rms = librosa.feature.rms(y=mono, frame_length=frame_len, hop_length=hop)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    is_silent = db < threshold_db
    spans = []
    i = 0
    n = len(is_silent)
    while i < n:
        if is_silent[i]:
            j = i
            while j < n and is_silent[j]:
                j += 1
            start_s = i * hop / sr
            end_s = j * hop / sr
            if (end_s - start_s) * 1000 >= min_silence_ms:
                spans.append((start_s, end_s))
            i = j
        else:
            i += 1
    return spans


def detect_beats(data: np.ndarray, sr: int,
                 bpm: Optional[float] = None) -> List[float]:
    """Return beat timestamps (seconds) for ``data``."""
    if not has_analysis() or data is None or len(data) == 0:
        return []
    librosa = _require_librosa()
    y = audio_mono(data)
    kwargs = {}
    if bpm:
        kwargs["start_bpm"] = float(bpm)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, **kwargs)
    beats = np.asarray(beats).ravel()
    if beats.size == 0:
        return []
    times = librosa.frames_to_time(beats, sr=sr)
    return [float(t) for t in times]


def estimate_bpm(data: np.ndarray, sr: int) -> Optional[float]:
    """Estimate tempo in BPM for ``data``, or None."""
    if not has_analysis() or data is None or len(data) == 0:
        return None
    librosa = _require_librosa()
    y = audio_mono(data)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    arr = np.ravel(tempo)
    return float(arr[0]) if arr.size else None


def analyze(data: np.ndarray, sr: int = 44100) -> dict:
    """Return a summary dict describing the clip's structure."""
    from .audio import duration_s

    if not has_analysis() or data is None or len(data) == 0:
        return {"available": False, "bpm": None, "beats": [],
                "silence_spans": [], "duration_s": duration_s(data, sr)}
    return {
        "available": True,
        "bpm": estimate_bpm(data, sr),
        "beats": detect_beats(data, sr),
        "silence_spans": detect_silence(data, sr),
        "duration_s": duration_s(data, sr),
    }


def audio_mono(data: np.ndarray) -> np.ndarray:
    """Mono (channel-averaged) 1-D array."""
    if data.ndim == 2 and data.shape[1] > 1:
        return data.mean(axis=1)
    return data.reshape(-1)
