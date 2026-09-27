"""Synthesise audio fixtures: sine-wave melodies at exact pitches and times.

Every melody test runs against audio whose ground truth is known by construction,
so a failure is a tracking bug rather than a bad recording.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24_000


def sine(freq: float, seconds: float, rate: int = SAMPLE_RATE, amp: float = 0.8) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    # Short attack/release ramps: an instant onset is a click that onset detection
    # will find but a human mouth never produces, and we want to test the realistic
    # case, not the easy one.
    env = np.ones_like(t)
    ramp = int(0.005 * rate)
    if ramp * 2 < len(env):
        env[:ramp] = np.linspace(0, 1, ramp)
        env[-ramp:] = np.linspace(1, 0, ramp)
    return (amp * env * np.sin(2 * np.pi * freq * t)).astype("float32")


def melody_audio(notes: list[tuple[float, float]], rate: int = SAMPLE_RATE,
                 gap: float = 0.02) -> np.ndarray:
    """Concatenate (freq_hz, seconds) notes with small silent gaps between them."""
    parts = [sine(f, s, rate) for f, s in notes]
    silence = np.zeros(int(gap * rate), dtype="float32")
    out: list[np.ndarray] = []
    for p in parts:
        out.append(p)
        out.append(silence)
    return np.concatenate(out) if out else np.zeros(0, dtype="float32")


def midi_to_hz(midi: int) -> float:
    return float(440.0 * 2 ** ((midi - 69) / 12))


# "Twinkle Twinkle" opening in C major — the canonical tracking fixture. MIDI
# numbers, quarter notes at 100 BPM (0.6 s each), one note per pitch.
TWINKLE = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]


def twinkle_audio(bpm: float = 100.0, rate: int = SAMPLE_RATE) -> np.ndarray:
    beat = 60.0 / bpm
    return melody_audio([(midi_to_hz(m), beat) for m in TWINKLE], rate)
