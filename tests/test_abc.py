"""ABC stage: notation correctness, built from synthetic Melody objects.

The pitch tracker is exercised end to end in test_melody.py (slow, one test).
These tests build `Melody` structures by hand so a failure names the notation bug
rather than a tracking wobble.
"""

from __future__ import annotations

import pytest

from hum2song import abc as A
from hum2song.melody import Melody, Note


def _melody(midi_beats: list[tuple[int, float, float]], *, tempo=100.0, key="C", meter="4/4") -> Melody:
    """`(midi, onset_beats, length_beats)` → a Melody with those exact notes."""
    notes = [Note(midi=m, onset=b * 0.6, duration=l * 0.6, probability=0.9,
                  beats=b, length_beats=l) for m, b, l in midi_beats]
    return Melody(notes=notes, sample_rate=24000, duration=len(notes) * 0.6, tempo=tempo,
                  key=f"{key} major", meter=meter, abc_key=key)


def test_renders_the_native_yue2_shape():
    mel = _melody([(60, 0, 1), (62, 1, 1), (64, 2, 1), (65, 3, 1)])
    score = A.from_melody(mel, title="test")
    text = A.render(score)
    assert text.splitlines()[0] == "X:1"
    assert "T:test" in text
    assert 'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"' in text
    assert 'V: Ins clef=treble name="Ins Melody" snm="Inst."' in text
    assert "M:4/4" in text and "L:1/16" in text and "Q:1/4=100" in text and "K:C" in text
    # The accompaniment voice rests for the whole section, as in YuE2's examples.
    assert "\nZ1|\n" in text
    assert A.validate(text) == []


def test_quarter_notes_become_4_sixteenths():
    mel = _melody([(60, 0, 1), (62, 1, 1), (64, 2, 1), (65, 3, 1)])
    text = A.render(A.from_melody(mel))
    vocal = [l for l in text.splitlines() if l.startswith("C")][0]
    assert vocal == "C4D4E4F4|"


def test_measures_always_sum_to_the_meter():
    # 5 quarter notes in 4/4: measure 2 holds one note and must be rest-padded.
    mel = _melody([(60, i, 1) for i in range(5)])
    score = A.from_melody(mel)
    for section in score.sections:
        for measure in section.measures:
            assert sum(t.sixteenths for t in measure) == 16
    assert A.validate(A.render(score)) == []


def test_gaps_become_rests_not_collapsed_rhythm():
    # note, 2-beat gap, note: the gap must be notated, not silently closed.
    mel = _melody([(60, 0, 1), (64, 3, 1)])
    text = A.render(A.from_melody(mel))
    assert "z8" in text  # two beats of eighth rests, i.e. 8 sixteenths
    assert A.validate(text) == []


def test_pitch_spelling_round_trips_through_midi():
    for midi in (36, 48, 59, 60, 61, 72, 84, 95):
        pitch = A._pitch(midi)
        assert A._abc_to_midi(pitch) == midi, f"{pitch} -> {A._abc_to_midi(pitch)} != {midi}"


def test_octave_case_convention():
    # ABC: C4 = "C" (uppercase, no marks); C5 = "c"; C3 = "C,"; C6 = "c'"
    assert A._pitch(60) == "C"
    assert A._pitch(72) == "c"
    assert A._pitch(48) == "C,"
    assert A._pitch(84) == "c'"


def test_key_override_is_validated():
    mel = _melody([(60, 0, 1), (62, 1, 1), (64, 2, 1), (65, 3, 1)])
    with pytest.raises(ValueError, match="key must be"):
        A.from_melody(mel, key="H")
    with pytest.raises(ValueError, match="meter must be"):
        A.from_melody(mel, meter="7/8")


def test_chord_symbols_are_rejected_by_validation():
    mel = _melody([(60, 0, 1), (62, 1, 1), (64, 2, 1), (65, 3, 1)])
    text = A.render(A.from_melody(mel))
    planted = text.replace("K:C", 'K:C\n%C "G7"')  # a comment is not a chord...
    assert A.validate(planted) == []
    planted = text.replace("C4D4E4F4|", '"G7"C4D4E4F4|')  # ...this is
    assert any("chord" in p for p in A.validate(planted))


def test_validate_catches_a_broken_measure():
    mel = _melody([(60, 0, 1), (62, 1, 1), (64, 2, 1), (65, 3, 1)])
    text = A.render(A.from_melody(mel))
    broken = text.replace("C4D4E4F4|", "C4D4E4|")  # 12 sixteenths, not 16
    problems = A.validate(broken)
    assert any("does not sum" in p for p in problems)


def test_sections_must_cover_every_note():
    mel = _melody([(60, i, 1) for i in range(8)])
    with pytest.raises(ValueError, match="unnotated"):
        A.from_melody(mel, sections=[("verse", [(0, 3)])])  # notes 4-7 orphaned
    score = A.from_melody(mel, sections=[("verse", [(0, 3)]), ("chorus", [(4, 7)])])
    assert [s.name for s in score.sections] == ["verse", "chorus"]
    assert A.validate(A.render(score)) == []


def test_split_sections_is_positional_and_honest():
    mel = _melody([(60, i, 1) for i in range(9)])
    parts = A.split_sections(mel, 3)
    assert [name for name, _ in parts] == ["part 1", "part 2", "part 3"]
    covered = [i for _, ranges in parts for lo, hi in ranges for i in range(lo, hi + 1)]
    assert covered == list(range(9))


def test_inspect_reports_shape():
    mel = _melody([(60, i, 1) for i in range(8)])
    info = A.inspect(A.render(A.from_melody(mel)))
    assert info["notes"] == 8 and info["measures"] == 2 and info["problems"] == []
    assert info["headers"]["M"] == "4/4" and info["headers"]["K"] == "C"


def test_3_4_meter_pads_to_12_sixteenths():
    mel = _melody([(60, 0, 1), (62, 1, 1), (64, 2, 1), (65, 3, 1), (67, 4, 1), (69, 5, 1)],
                  meter="3/4")
    score = A.from_melody(mel)
    assert len(score.sections[0].measures) == 2
    for measure in score.sections[0].measures:
        assert sum(t.sixteenths for t in measure) == 12
    assert A.validate(A.render(score), meter="3/4") == []


def test_write_and_read_round_trip(tmp_path):
    mel = _melody([(60, 0, 1), (62, 1, 1), (64, 2, 1), (65, 3, 1)])
    path = A.write(A.from_melody(mel, title="round"), tmp_path / "out.abc")
    assert A.read(path) == A.render(A.from_melody(mel, title="round"))
    assert path.read_bytes().count(b"\r") == 0  # LF, like every ABC consumer expects
