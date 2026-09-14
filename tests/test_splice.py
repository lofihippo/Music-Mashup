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


# ---- highlight / best-part auto-splice -------------------------------------
def test_highlight_window_finds_burst_center():
    from musicmashup import analysis
    from tests.fixtures import make_crescendo_burst

    sr = 44100
    dur = 8.0
    burst = 5.0  # burst center at 5s
    data = make_crescendo_burst(dur, sr=sr, burst_center_s=burst)
    win = analysis.highlight_window(data, sr)
    assert win is not None
    start, end = win
    # window should straddle the burst (loudest) center
    assert start < burst < end
    assert end - start > 0.5


def test_highlight_window_before_after():
    from musicmashup import analysis
    from tests.fixtures import make_crescendo_burst

    sr = 44100
    burst = 5.0
    data = make_crescendo_burst(8.0, sr=sr, burst_center_s=burst)
    start, end = analysis.highlight_window(data, sr, before_s=2.5, after_s=1.0)
    # starts ~2.5s before peak, ends ~1s after (clamped to clip bounds)
    assert abs((burst - start) - 2.5) < 0.3
    assert abs((end - burst) - 1.0) < 0.3


def test_auto_splice_highlight_sets_bounds():
    from tests.fixtures import make_crescendo_burst

    sr = 44100
    burst = 5.0
    data = make_crescendo_burst(8.0, sr=sr, burst_center_s=burst)
    clip = Clip(source="a")
    auto = splice.auto_splice_clip(clip, data, sr=sr, detect="highlight")
    assert auto.detected_start is not None
    assert auto.detected_end is not None
    assert auto.detected_start < burst < auto.detected_end


def test_highlight_renders_short_window():
    from tests.fixtures import make_crescendo_burst

    sr = 44100
    data = make_crescendo_burst(6.0, sr=sr, burst_center_s=4.0)
    c = Collection(name="hl")
    c.add(Clip(source="a"))
    c.add(Clip(source="b"))
    data2 = make_crescendo_burst(6.0, sr=sr, burst_center_s=3.5)
    # render with highlight detect applied
    auto_a = splice.auto_splice_clip(Clip(source="a"), data, sr=sr, detect="highlight")
    auto_b = splice.auto_splice_clip(Clip(source="b"), data2, sr=sr, detect="highlight")
    # transfer detected bounds to start/end (as the CLI/server does)
    for a in (auto_a, auto_b):
        a.start = a.detected_start
        a.end = a.detected_end
    c2 = Collection(name="hl2")
    c2.add(auto_a)
    c2.add(auto_b)
    out = splice.render(c2, {"a": data, "b": data2}, sr=sr)
    # each highlight is ~3.5s (2.5 before + 1 after), so total < 8s
    assert out.shape[0] / sr < 8.0
    assert out.shape[0] / sr > 3.0
