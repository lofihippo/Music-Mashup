import numpy as np
import pytest

from musicmashup import splice
from musicmashup.model import Collection, Clip
from tests.fixtures import make_tone, silence


def test_concat_full_clips():
    a = make_tone(1.0)
    b = make_tone(1.0, freq=660)
    c = Collection(name="t")
    c.add(Clip(source="a"))
    c.add(Clip(source="b"))
    out = splice.render(c, {"a": a, "b": b})
    assert out.shape[0] == a.shape[0] + b.shape[0]
    assert out.shape[1] == 2


def test_slices_in_out():
    a = make_tone(3.0)
    c = Collection(name="t")
    c.add(Clip(source="a", start=1.0, end=2.0))
    out = splice.render(c, {"a": a})
    assert abs(len(out) / 44100.0 - 1.0) < 0.01  # 1 second slice


def test_end_zero_is_respected_not_whole():
    # end=0.0 is a real bound (empty), not "use the whole file".
    a = make_tone(3.0)
    c = Collection(name="t")
    c.add(Clip(source="a", start=1.0, end=0.0))
    out = splice.render(c, {"a": a})
    assert out.shape[0] == 0


def test_mismatched_lists_no_truncation():
    # zip-based original would silently drop; ours raises for missing source.
    a = make_tone(1.0)
    c = Collection(name="t")
    c.add(Clip(source="a"))
    c.add(Clip(source="missing"))
    with pytest.raises(KeyError):
        splice.render(c, {"a": a})


def test_crossfade_reduces_total():
    a = make_tone(2.0)
    b = make_tone(2.0)
    c = Collection(name="t")
    c.add(Clip(source="a"))
    c.add(Clip(source="b"))
    total = splice.render(c, {"a": a, "b": b})
    xf = splice.render(c, {"a": a, "b": b}, crossfade_s=0.5)
    assert xf.shape[0] < total.shape[0]
    assert abs(len(xf) / 44100.0 - 3.5) < 0.05  # 4s - 0.5s


def test_gain_db_applies():
    a = make_tone(1.0, amplitude=0.2)
    c = Collection(name="t")
    c.add(Clip(source="a", gain_db=6.0))  # ~2x
    out = splice.render(c, {"a": a})
    # peak should roughly double
    assert abs(np.abs(out).max() - 0.4) < 0.05


def test_auto_splice_silence_trims_edges():
    sr = 44100
    body = make_tone(2.0, sr=sr)
    padded = np.concatenate([silence(0.8, sr), body, silence(0.8, sr)], axis=0)
    clip = Clip(source="a")
    auto = splice.auto_splice_clip(clip, padded, sr=sr, detect="silence")
    assert auto.detected_start is not None and auto.detected_start > 0.3
    assert auto.detected_end is not None and auto.detected_end < len(padded) / sr - 0.3
    # original clip not mutated
    assert clip.start is None


def test_empty_collection_returns_zeros():
    c = Collection(name="empty")
    out = splice.render(c, {})
    assert out.shape[0] == 0


def test_skip_gapped_clip():
    a = make_tone(1.0)
    b = make_tone(1.0)
    c = Collection(name="t")
    c.add(Clip(source="a"))
    c.add(Clip(source="b", start=5.0, end=5.0))  # empty slice
    out = splice.render(c, {"a": a, "b": b})
    assert out.shape[0] == a.shape[0]  # only the first contributes
