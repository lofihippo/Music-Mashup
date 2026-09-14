"""Command-line interface for Music Mashup.

Splice a folder of audio files into a single mix, optionally trimming silence
and applying crossfades/loudness normalization. A convenient scriptable path
for "splice this whole collection end-to-end".
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import analysis, audio, splice
from .model import Collection, Clip


def build_playlist(paths, *, mode="off", sr=44100):
    """Return (Collection, source_index) from a list of audio files."""
    collection = Collection(name="Auto Mix")
    source_index = {}
    for path in paths:
        name = os.path.basename(path)
        arr = audio.read(path, sr)
        source_index[name] = arr
        clip = Clip(source=name, start=0.0, end=audio.duration_s(arr, sr))
        if mode == "highlight":
            clip = splice.auto_splice_clip(clip, arr, sr=sr, detect=mode)
            # auto-detected bounds override the placeholder full-file bounds
            if clip.detected_start is not None:
                clip.start = clip.detected_start
            if clip.detected_end is not None:
                clip.end = clip.detected_end
        collection.add(clip)
    return collection, source_index


def main(argv=None):
    p = argparse.ArgumentParser(description="Splice a series of songs into a mix.")
    p.add_argument("sources", nargs="+", help="audio files or a directory")
    p.add_argument("-o", "--output", default="mashup.wav")
    p.add_argument("--detect", choices=["highlight", "off"], default="highlight",
                   help="automatic splice-point selection "
                        "(highlight = best/loudest part of each track)")
    p.add_argument("--crossfade", type=float, default=0.0,
                   help="crossfade seconds between clips")
    p.add_argument("--normalize", action="store_true",
                   help="normalize loudness (EBU R128)")
    p.add_argument("--target-lufs", type=float, default=-16.0)
    p.add_argument("--sr", type=int, default=44100)
    args = p.parse_args(argv)

    files = []
    for s in args.sources:
        sp = Path(s)
        if sp.is_dir():
            for ext in audio.SUPPORTED_EXTENSIONS:
                files.extend(sorted(sp.glob(f"*{ext}")))
        else:
            files.append(sp)
    files = sorted(set(files))

    if not files:
        print("No audio files found.")
        return 1

    collection, source_index = build_playlist(
        [str(f) for f in files], mode=args.detect, sr=args.sr)

    print(f"Splicing {len(collection.clips)} clips "
          f"(detect={args.detect}, crossfade={args.crossfade}s) ...")
    out = splice.render(collection, source_index, sr=args.sr,
                        normalize=args.normalize,
                        target_lufs=args.target_lufs,
                        crossfade_s=args.crossfade)
    audio.write(args.output, out, args.sr)
    print(f"Wrote {args.output} "
          f"({audio.duration_s(out, args.sr):.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
