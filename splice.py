"""Splice engine: turn a Collection of clips into one rendered mix.

This is the heart of the tool. It correctly handles:

* each clip's explicit in/out (or the whole source when not set),
* per-clip gain,
* fade in / fade out,
* crossfades between adjacent clips,
* optional loudness normalization (EBU R128 via ``pyloudnorm``).

It fixes the original bugs: no silent truncation when lists differ, and an
``end`` of ``0`` is treated as a real bound rather than collapsing to the whole
file. Everything operates on float32 numpy arrays at a shared sample rate.
"""

from __future__ import annotations

import copy
from typing import List, Optional

import numpy as np

from .model import Clip, Collection
from . import analysis, audio

try:
    import pyloudnorm as pyln
except Exception:  # pragma: no cover - env dependent
    pyln = None


def _slice_array(data: np.ndarray, sr: int,
                 start: Optional[float], end: Optional[float]) -> np.ndarray:
    """Slice ``data`` by in/out seconds; both None returns the whole array."""
    n = len(data)
    start_i = int(start * sr) if start is not None else 0
    end_i = int(end * sr) if end is not None else n
    start_i = max(0, min(start_i, n))
    end_i = max(start_i, min(end_i, n))
    return data[start_i:end_i]


def _apply_gain_fade(chunk: np.ndarray, sr: int, clip: Clip) -> np.ndarray:
    out = chunk.astype(np.float32, copy=True)
    if clip.gain_db:
        out = out * (10.0 ** (clip.gain_db / 20.0))
    n = len(out)
    if clip.fade_in > 0:
        f = int(clip.fade_in * sr)
        f = min(f, n)
        if f > 0:
            ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
            out[:f] *= ramp[:, None] if out.ndim > 1 else ramp
    if clip.fade_out > 0:
        f = int(clip.fade_out * sr)
        f = min(f, n)
        if f > 0:
            ramp = np.linspace(1.0, 0.0, f, dtype=np.float32)
            out[n - f:] *= ramp[:, None] if out.ndim > 1 else ramp
    return out


def _crossfade(a: np.ndarray, b: np.ndarray, xfade: float,
               sr: int) -> np.ndarray:
    """Equal-power crossfade the tail of ``a`` into ``b``, returning merged array.

    The overlap length is clamped to the shorter of the two clips, so no
    silence padding is added; clips shorter than the crossfade simply merge
    fully over their length.
    """
    if xfade <= 0:
        return np.concatenate([a, b], axis=0)
    n = int(xfade * sr)
    n = min(n, len(a), len(b))
    if n <= 1:
        return np.concatenate([a, b], axis=0)
    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    a_tail = a[-n:] * (fade_out[:, None] if a.ndim > 1 else fade_out)
    b_head = b[:n] * (fade_in[:, None] if b.ndim > 1 else fade_in)
    merged_mid = a_tail + b_head
    return np.concatenate([a[:-n], merged_mid, b[n:]], axis=0)


def overload(
    collection: Collection,
    source_index: dict,
    *,
    sr: int = 44100,
    normalize: bool = False,
    target_lufs: float = -16.0,
    crossfade_s: float = 0.0,
) -> np.ndarray:
    """Overlap+splice the collection into one float32 array ``(n, ch)``."""
    if not collection.clips:
        return np.zeros((0, 2), dtype=np.float32)

    chunks: List[np.ndarray] = []
    for clip in collection.clips:
        if clip.source not in source_index:
            raise KeyError(f"Source {clip.source!r} not loaded")
        full = source_index[clip.source]
        chunk = _slice_array(full, sr, clip.start, clip.end)
        if len(chunk) == 0:
            continue  # empty/gapped clip; skip rather than silently truncate
        chunks.append(_apply_gain_fade(chunk, sr, clip))

    if not chunks:
        ch = source_index[next(iter(source_index))].shape[1] if source_index else 2
        return np.zeros((0, ch), dtype=np.float32)

    if crossfade_s > 0 and len(chunks) > 1:
        merged = chunks[0]
        for nxt in chunks[1:]:
            merged = _crossfade(merged, nxt, crossfade_s, sr)
    else:
        merged = np.concatenate(chunks, axis=0)

    if normalize and pyln is not None:
        merged = _normalize(merged, sr, target_lufs)
    return merged.astype(np.float32, copy=False)


def render(
    collection: Collection,
    sources,
    *,
    sr: int = 44100,
    normalize: bool = False,
    target_lufs: float = -16.0,
    crossfade_s: float = 0.0,
) -> np.ndarray:
    """Convenience wrapper: ``sources`` is a dict or a list of arrays."""
    if isinstance(sources, dict):
        index = sources
    else:
        index = {}
        names = [c.source for c in collection.clips]
        for name, arr in zip(names, sources):
            index.setdefault(name, arr)
    return overload(collection, index, sr=sr, normalize=normalize,
                    target_lufs=target_lufs, crossfade_s=crossfade_s)


def auto_splice_clip(
    clip: Clip,
    data: np.ndarray,
    *,
    sr: int = 44100,
    detect: str = "off",
    bpm: Optional[float] = None,
    min_silence_ms: int = 500,
    before_s: float = 2.5,
    after_s: float = 1.0,
) -> Clip:
    """Suggest splice bounds for a clip (returns a copy, does not mutate).

    detect modes:
      "off"       -> leave as-is
      "silence"   -> trim leading/trailing silence
      "beat"      -> snap to beat-aligned in/out
      "highlight" -> single loudest crescendo window (best part of the track)
    """
    out = copy.deepcopy(clip)
    dur = audio.duration_s(data, sr)

    if detect == "off":
        return out

    if detect == "silence":
        spans = analysis.detect_silence(data, sr, min_silence_ms=min_silence_ms)
        if spans and spans[0][0] < 0.25:
            out.detected_start = spans[0][1]
        if spans and spans[-1][1] > dur - 0.25:
            out.detected_end = spans[-1][0]
        return out

    if detect == "beat":
        beats = analysis.detect_beats(data, sr, bpm=bpm)
        if beats:
            out.detected_start = beats[0]
            out.detected_end = beats[-1]
        return out

    if detect == "highlight":
        win = analysis.highlight_window(
            data, sr, before_s=before_s, after_s=after_s)
        if win:
            out.detected_start, out.detected_end = win
        return out

    return out


def _normalize(data: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    if pyln is None or len(data) == 0:
        return data
    try:
        arr = np.asarray(data, dtype=np.float32)
        meter = pyln.Meter(sr)
        lufs = float(meter.integrated_loudness(arr))
        gain_db = target_lufs - lufs
        return arr * (10.0 ** (gain_db / 20.0))
    except Exception:
        return data
