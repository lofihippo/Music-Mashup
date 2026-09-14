"""Shared test helpers: synthesize deterministic audio fixtures with numpy.

No binary assets in the repo — test clips are generated on the fly.
"""

from __future__ import annotations

import numpy as np


def make_tone(duration_s: float, sr: int = 44100, freq: float = 440.0,
              channels: int = 2, amplitude: float = 0.3,
              sr_offset: int = 0) -> np.ndarray:
    """Generate a sine clip: `(n, channels)` float32."""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    data = np.repeat(wave[:, None], channels, axis=1).astype(np.float32)
    return data


def make_pulse_train(duration_s: float, sr: int = 44100, bpm: float = 120.0,
                     channels: int = 2) -> np.ndarray:
    """Generate a clip with sharp transient pulses spaced at `bpm`."""
    n = int(duration_s * sr)
    data = np.zeros((n, channels), dtype=np.float32)
    beat_len = sr * (60.0 / bpm)
    i = 0
    while i < n:
        end = min(n, i + int(0.02 * sr))
        data[i:end] = 0.5
        i += int(beat_len)
    return data


def silence(duration_s: float, sr: int = 44100, channels: int = 2) -> np.ndarray:
    n = int(duration_s * sr)
    return np.zeros((n, channels), dtype=np.float32)


def make_crescendo_burst(duration_s: float, sr: int = 44100,
                         burst_center_s: float = None,
                         burst_width_s: float = 1.0,
                         channels: int = 2) -> np.ndarray:
    """A quiet bed with one short loud burst (clear RMS peak to detect).

    Used to validate highlight detection finds the burst and cuts around it.
    """
    n = int(duration_s * sr)
    data = np.full((n, channels), 0.05, dtype=np.float32)  # quiet bed
    if burst_center_s is None:
        burst_center_s = duration_s * 0.6
    center = int(burst_center_s * sr)
    half = int((burst_width_s / 2.0) * sr)
    start = max(0, center - half)
    end = min(n, center + half)
    if end > start:
        win = np.hanning(end - start)[:, None]  # smooth window for a clear peak
        data[start:end] += 0.9 * win
    return np.ascontiguousarray(data, dtype=np.float32)
