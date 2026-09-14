"""Playlist / collection model.

A "mashup" here is really a *series*: an ordered list of source clips pulled
from one or more audio files, each with optional in/out cut points, crossfade,
and gain. The whole thing serializes to JSON so the same collection can be
edited in the web UI or driven from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, List


def ms(value: Optional[float]) -> Optional[int]:
    """Convert seconds to whole milliseconds; None stays None."""
    if value is None:
        return None
    return int(round(value * 1000))


@dataclass
class Clip:
    """One slice of a source audio file used in the mix.

    ``start``/``end`` are in seconds. When both are ``None`` the whole file is
    used (auto-splice may fill them in). ``detected_start``/``detected_end``
    record what automatic detection found, so explicit user values can be
    distinguished from auto values.
    """

    source: str
    start: Optional[float] = None
    end: Optional[float] = None
    fade_in: float = 0.0
    fade_out: float = 0.0
    gain_db: float = 0.0

    detected_start: Optional[float] = None
    detected_end: Optional[float] = None

    def is_explicit(self) -> bool:
        """True if the user gave at least one explicit boundary."""
        return self.start is not None or self.end is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Clip":
        return cls(**data)


@dataclass
class Collection:
    """An ordered series of clips that compose one mashup."""

    name: str = "Untitled Mix"
    clips: List[Clip] = field(default_factory=list)

    def add(self, clip: Clip) -> None:
        self.clips.append(clip)

    def sample_rate(self) -> int:
        return 44100

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Collection":
        clips = [Clip.from_dict(c) for c in data.get("clips", [])]
        return cls(name=data.get("name", "Untitled Mix"), clips=clips)
