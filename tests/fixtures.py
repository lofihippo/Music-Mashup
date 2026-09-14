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
