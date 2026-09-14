from musicmashup.model import Collection, Clip


def test_collection_roundtrip():
    c = Collection(name="My Mix")
    c.add(Clip(source="a.mp3", start=1.5, end=10.0, gain_db=-3, fade_in=0.5))
    c.add(Clip(source="b.wav", fade_out=1.0))
    d = c.to_dict()
    back = Collection.from_dict(d)
    assert back.name == "My Mix"
    assert len(back.clips) == 2
    assert back.clips[0].start == 1.5
    assert back.clips[0].gain_db == -3


def test_clip_is_explicit():
    assert not Clip(source="a").is_explicit()
    assert Clip(source="a", start=1.0).is_explicit()
    assert Clip(source="a", end=2.0).is_explicit()


def test_detected_distinct_from_explicit():
    c = Clip(source="a", start=1.0, detected_start=3.0)
    assert c.is_explicit()
    assert c.detected_start == 3.0
