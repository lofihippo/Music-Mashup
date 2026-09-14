import numpy as np
import pytest

from musicmashup import audio
from tests.fixtures import make_tone


def test_write_read_roundtrip(tmp_path):
    data = make_tone(1.0)
    path = str(tmp_path / "tone.wav")
    audio.write(path, data)
    back = audio.read(path)
    assert back.shape == data.shape
    assert abs(len(back) / 44100.0 - 1.0) < 0.01


def test_write_read_mp3_flac(tmp_path):
    data = make_tone(1.0)
    for ext in ("flac",):
        path = str(tmp_path / f"tone.{ext}")
        audio.write(path, data)
        back = audio.read(path)
        assert back.shape[1] == 2


def test_trim_silence_edges():
    from tests.fixtures import silence
    sr = 44100
    body = make_tone(2.0, sr=sr)
    padded = np.concatenate([silence(0.5, sr), body, silence(0.5, sr)], axis=0)
    s, e = audio.trim_silence_edges(padded, sr)
    assert s > 0.3 * sr
    assert e < len(padded) - 0.3 * sr
    assert s < e
