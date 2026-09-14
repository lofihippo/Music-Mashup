"""Audio analysis: silence and beat detection for automatic splice points.

These helpers operate on the numpy float32 arrays produced by :mod:`.audio`.
They degrade gracefully when analysis deps are missing, returning conservative
defaults so the engine still works with explicit in/out points.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .audio import duration_s


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


# ---- highlight ("best part") detection -------------------------------------
# Find the loudest crescendo moments in a track and a window that captures the
# build-up into the peak (hook / drop), which is what a mashup wants to splice.

def _rms_db(data: np.ndarray, sr: int, hop_ms: int = 10,
            frame_ms: int = 25) -> np.ndarray:
    """Return a dB RMS energy envelope (mono), one value per hop frame."""
    librosa = _require_librosa()
    y = audio_mono(data)
    frame_len = max(1, int(frame_ms / 1000.0 * sr))
    hop = max(1, int(hop_ms / 1000.0 * sr))
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def _find_spaced_peaks(db: np.ndarray, sr: int, hop_ms: int,
                       n: int, min_gap_s: float) -> List[int]:
    """Return up to ``n`` frame indices that are local maxima, spaced apart.

    Peaks are ranked by loudness but rejected if they sit within ``min_gap_s``
    of an already-chosen higher peak, so we don't select the same section twice.
    """
    n = max(1, int(n))
    min_gap = max(1, int(min_gap_s * 1000 / hop_ms))  # in frames
    # candidate frames that are local maxima
    peaks = []
    for i in range(1, len(db) - 1):
        if db[i] >= db[i - 1] and db[i] >= db[i + 1]:
            peaks.append((db[i], i))
    peaks.sort(reverse=True)  # loudest first

    chosen = []
    for _, idx in peaks:
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(idx)
            if len(chosen) >= n:
                break
    return sorted(chosen)


def best_peaks(data: np.ndarray, sr: int, n: int = 1,
               min_gap_s: float = 5.0,
               hop_ms: int = 10, frame_ms: int = 25) -> List[float]:
    """Return up to ``n`` peak times (seconds) spaced at least ``min_gap_s``.

    These are the loudest, well-separated moments used as highlight anchors.
    """
    if not has_analysis() or data is None or len(data) == 0:
        return []
    db = _rms_db(data, sr, hop_ms=hop_ms, frame_ms=frame_ms)
    if db.size == 0:
        return []
    idx = _find_spaced_peaks(db, sr, hop_ms, n, min_gap_s)
    hop_s = hop_ms / 1000.0
    return [float(i * hop_s) for i in idx]


def highlight_window(data: np.ndarray, sr: int,
                     before_s: float = 2.5, after_s: float = 1.0,
                     min_gap_s: float = 5.0) -> Optional[Tuple[float, float]]:
    """Return ``(start_s, end_s)`` for the single best highlight window.

    Cuts ``before_s`` up to the loudest peak and ``after_s`` past it, clamped to
    the clip bounds. Returns None if the clip is too short to yield a window.
    """
    if not has_analysis() or data is None or len(data) < sr:  # < 1s
        return None
    peaks = best_peaks(data, sr, n=1, min_gap_s=min_gap_s)
    if not peaks:
        return None
    peak = peaks[0]
    dur = duration_s(data, sr)
    start = max(0.0, peak - before_s)
    end = min(dur, peak + after_s)
    if end - start < 0.5:  # too small to be useful
        return None
    return (start, end)
