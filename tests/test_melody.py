"""End-to-end melody tracking on synthetic audio with known ground truth.

pyin on 8.4 s of sine melody takes ~20 s on CPU, so there is one tracking test,
not a suite of them. It is the only place where a wrong answer means the tracker
is wrong rather than the fixture is wrong.
"""

from __future__ import annotations

import pytest

pytest.importorskip("librosa", reason="tracking needs librosa")

from hum2song import melody as M
from tests import fixtures


@pytest.fixture(scope="module")
def twinkle() -> M.Melody:
    audio = fixtures.twinkle_audio(bpm=100.0)
    return M.track(audio, fixtures.SAMPLE_RATE)


def test_recovers_every_pitch_of_a_known_tune(twinkle):
    assert twinkle.pitches == fixtures.TWINKLE


def test_tempo_key_meter_from_a_known_construction(twinkle):
    # Constructed at exactly 100 BPM in C major. Tempo folds to the 60-180 band,
    # key comes from the Krumhansl-Schmuckler profile, meter defaults to 4/4.
    assert abs(twinkle.tempo - 100.0) <= 6.0
    assert twinkle.key == "C major"
    assert twinkle.meter == "4/4"
    assert twinkle.diagnostics["tempo_source"] == "onsets"


def test_notes_carry_their_quantised_grid(twinkle):
    assert all(n.length_beats > 0 for n in twinkle.notes)
    # quarter notes at the estimated tempo: each note ≈ 1 beat ± folding slack
    assert all(0.5 <= n.length_beats <= 2.5 for n in twinkle.notes)
    # onsets are monotonic and non-overlapping after the shorten-pass
    for a, b in zip(twinkle.notes, twinkle.notes[1:]):
        assert a.beats + a.length_beats <= b.beats + 1e-6


def test_voiced_probability_is_reported_honestly(twinkle):
    assert twinkle.diagnostics["mean_voiced_probability"] > 0.5
    assert all(0.0 <= n.probability <= 1.0 for n in twinkle.notes)


def test_unpitched_audio_raises_not_garbage():
    import numpy as np
    noise = (np.random.default_rng(7).standard_normal(fixtures.SAMPLE_RATE * 3) * 0.5).astype("float32")
    with pytest.raises(ValueError, match="no pitched notes"):
        M.track(noise, fixtures.SAMPLE_RATE)


def test_melody_serialises_round_trip(twinkle):
    restored = M.Melody.from_dict(twinkle.to_dict())
    assert restored.pitches == twinkle.pitches
    assert restored.tempo == twinkle.tempo and restored.key == twinkle.key
