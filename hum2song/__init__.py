"""hum2song — hum a melody, get a song.

Four stages, each one an artifact you can inspect before the next runs:

    audio  → melody → abc → song

The package deliberately does no inference itself: pitch tracking runs on librosa,
ABC is written and round-trip validated here, and synthesis is delegated to a
YuE2 install (the official `yue2.cli generate`). Every stage writes a manifest so
a song is reproducible from its inputs.
"""

from __future__ import annotations

__version__ = "0.1.0"

STAGES = ("audio", "melody", "abc", "song")
