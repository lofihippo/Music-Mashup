import os
import sys

# Ensure the repo root (the `musicmashup` package dir's parent) is importable
# so `import musicmashup` works regardless of where pytest is invoked from.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)
