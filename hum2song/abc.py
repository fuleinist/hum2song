"""Stage 3: melody → ABC in YuE2's native notation.

The target format is what YuE2's own examples and SheetSage2's melody-only export
produce, not general ABC:

    X:1
    T:<title>
    M:4/4
    L:1/16
    Q:1/4=88
    V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
    V: Ins clef=treble name="Ins Melody" snm="Inst."
    K:C
    % verse
    V: Vocal
    E2G2A2G2E2D2C4|...
    V: Ins
    Z4|

Three properties matter and are enforced here, because `cot="melody"` is a symbolic
*condition* on generation rather than a hard constraint on the audio:

- **Chord-free.** No `"C"`-style harmony annotations anywhere. Passing chords with
  `cot="melody"` conditions on a harmony the model was not told to follow.
- **Both voices present.** `V: Vocal` carries the melody; `V: Ins` carries
  multi-measure rests (`Z<n>`) so the score has the two-voice shape the model was
  trained on. Dropping the voice silently is not the same as an empty one.
- **One measure per line, exact L:1/16 durations.** Every line sums to the meter,
  which is what makes the round-trip check in `validate` meaningful.

This module writes ABC and reads it back. It never shells out, so it has no
dependency beyond the standard library.
"""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

# YuE2's examples and SheetSage2's exporter both use 1/16 as the unit note.
UNIT = 16

# Spelling inside a key signature: ABC uses the same letters as the key, with
# accidentals implied by K:. `K:Eb` means E is flat unless marked `=E`.
_SHARP_KEYS = {"G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7}
_FLAT_KEYS = {"F": 1, "Bb": 2, "Eb": 3, "Ab": 4, "Db": 5, "Gb": 6, "Cb": 7}
_ORDER_SHARP = ("F", "C", "G", "D", "A", "E", "B")
_ORDER_FLAT = ("B", "E", "A", "D", "G", "C", "F")
# Semitone of each letter, relative to C.
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PC_LETTER = {0: "C", 2: "D", 4: "E", 5: "F", 7: "G", 9: "A", 11: "B"}


@dataclasses.dataclass(frozen=True)
class Token:
    """One notated event inside a measure."""

    kind: str            # "note" | "rest"
    text: str            # the ABC text, e.g. "E2", "z4"
    sixteenths: int      # duration in units of L:1/16


@dataclasses.dataclass
class Score:
    title: str
    key: str             # "C", "Eb", "A", "F#"...
    meter: str           # "4/4", "3/4"
    unit: int            # 16
    tempo: float
    sections: list["Section"]
    warnings: list[str] = dataclasses.field(default_factory=list)

    @property
    def measures_per_section(self) -> list[int]:
        return [len(s.measures) for s in self.sections]


@dataclasses.dataclass
class Section:
    name: str            # "verse", "chorus", "bridge", ...
    measures: list[list[Token]]

    @property
    def token_count(self) -> int:
        return sum(len(m) for m in self.measures)


def render(score: Score) -> str:
    """Render a `Score` to ABC text in YuE2's native two-voice shape."""
    lines = ["X:1", f"T:{score.title}", f"M:{score.meter}", f"L:1/{score.unit}",
             f"Q:1/4={int(round(score.tempo))}",
             'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"',
             'V: Ins clef=treble name="Ins Melody" snm="Inst."',
             f"K:{score.key}"]
    for section in score.sections:
        lines.append(f"% {section.name}")
        lines.append("V: Vocal")
        for measure in section.measures:
            lines.append("".join(t.text for t in measure) + "|")
        lines.append("V: Ins")
        # The accompanying voice is silent: one multi-measure rest covering the
        # whole section, which is what YuE2's own examples write.
        lines.append(f"Z{len(section.measures)}|")
    return "\n".join(lines) + "\n"


def write(score: Score, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(score), encoding="utf-8", newline="\n")
    return path


def from_melody(melody, *, title: str = "", sections: list[tuple[str, list[int]]] | None = None,
                key: str | None = None, meter: str | None = None,
                tempo: float | None = None) -> Score:
    """Notate a `melody.Melody` as an ABC `Score`.

    `sections` splits the note list into named parts — `[("verse", [0, 7]),
    ("chorus", [8, 15])]` as inclusive index pairs. Without it the whole melody is
    one `% melody` section, which is the honest default: nothing here can hear a
    chorus.
    """
    notes = melody.notes
    if not notes:
        raise ValueError("the melody has no notes to notate")

    key = key or melody.abc_key
    meter = meter or melody.meter
    tempo = tempo or melody.tempo
    if not re.fullmatch(r"[A-G][b#]?", key):
        raise ValueError(f"key must be a letter with at most one accidental; got {key!r}")
    if meter not in ("2/4", "3/4", "4/4", "6/8"):
        raise ValueError(f"meter must be 2/4, 3/4, 4/4 or 6/8; got {meter!r}")

    warnings = list(melody.warnings)
    beats_per_measure = _beats_per_measure(meter)
    sixteenths_per_measure = int(round(beats_per_measure * UNIT / 4))

    parts = sections or [("melody", [(0, len(notes) - 1)])]
    score_sections: list[Section] = []
    used: set[int] = set()
    for name, ranges in parts:
        measures: list[list[Token]] = []
        for lo, hi in ranges:
            if not (0 <= lo <= hi < len(notes)):
                raise ValueError(
                    f"section {name!r} covers notes {lo}-{hi}, outside 0-{len(notes) - 1}")
            used.update(range(lo, hi + 1))
            laid, warns = _lay_out(notes[lo:hi + 1], sixteenths_per_measure)
            measures.extend(laid)
            warnings.extend(warns)
        score_sections.append(Section(name=name, measures=measures))

    missing = sorted(set(range(len(notes))) - used)
    if missing:
        raise ValueError(f"sections leave note(s) unnotated: {missing}")

    return Score(title=title or "", key=key, meter=meter, unit=UNIT, tempo=float(tempo),
                 sections=score_sections, warnings=warnings)


def _beats_per_measure(meter: str) -> float:
    num, den = (float(x) for x in meter.split("/"))
    return num * 4.0 / den  # quarter-note beats per measure


def _lay_out(notes, sixteenths_per_measure: int) -> tuple[list[list[Token]], list[str]]:
    """Place quantised notes onto measures, padding gaps with rests."""
    warnings: list[str] = []
    measures: list[list[Token]] = [[]]
    cursor = 0.0  # sixteenth-note position from the start of the tune
    for note in notes:
        onset = note.beats * UNIT / 4.0
        length = max(int(round(note.length_beats * UNIT / 4.0)), 1)
        if onset > cursor + 1e-6:
            # A gap longer than the quantisation grid: notate it as a rest so the
            # rhythm the user sang survives, rather than closing the hole.
            gap = int(round(onset - cursor))
            measures, cursor = _emit_rest(measures, cursor, gap, sixteenths_per_measure)
        elif onset < cursor - 1e-6:
            warnings.append(
                f"note {note.name} starts {cursor - onset:.2f} sixteenths before the previous "
                "one ends; the earlier note was shortened"
            )
            length = max(length - int(round(cursor - onset)), 1)
        measures, cursor = _emit_note(measures, cursor, note, length, sixteenths_per_measure, warnings)

    # Pad the final measure out to the bar line: an incomplete last measure makes
    # the score ambiguous to a reader that counts measures.
    if cursor % sixteenths_per_measure:
        pad = sixteenths_per_measure - int(cursor % sixteenths_per_measure)
        measures, cursor = _emit_rest(measures, cursor, pad, sixteenths_per_measure)
    if not measures[-1]:
        measures.pop()
    return measures, warnings


def _open_measure_if_needed(measures: list, cursor: float, per_measure: int) -> None:
    """Start a new measure when the cursor sits on a bar line with content behind."""
    if cursor > 0 and int(cursor) % per_measure == 0 and measures[-1]:
        measures.append([])


def _emit_note(measures, cursor, note, length, per_measure, warnings):
    """Write one note, splitting it across bar lines if it does not fit.

    A note that would cross a bar line is split into two attacks rather than tied,
    and the split is reported: a tie would be the musically correct notation, but
    this writer emits no ties, so the warning is the honest record of the
    approximation instead of a silent rhythm change.
    """
    remaining = length
    while remaining > 0:
        _open_measure_if_needed(measures, cursor, per_measure)
        space = per_measure - int(cursor % per_measure)
        take = min(space, remaining)
        for part in _duration_parts(take):
            measures[-1].append(Token("note", _pitch(note.midi) + _duration_text(part), part))
        cursor += take
        remaining -= take
        if remaining > 0:
            warnings.append(
                f"note {note.name} of {length} sixteenths crossed a bar line; "
                "it is split into two attacks rather than tied"
            )
    return measures, cursor


def _emit_rest(measures, cursor, gap, per_measure):
    remaining = gap
    while remaining > 0:
        _open_measure_if_needed(measures, cursor, per_measure)
        space = per_measure - int(cursor % per_measure)
        take = min(space, remaining)
        for part in _duration_parts(take):
            measures[-1].append(Token("rest", "z" + _duration_text(part), part))
        cursor += take
        remaining -= take
    return measures, cursor


def _pitch(midi: int) -> str:
    """MIDI number → ABC pitch, spelled absolutely (no key-signature dependency).

    The native dialect's octave convention, per the YuE2 ABC reference: uppercase
    `C` is C4 (middle C), lowercase `c` is C5, commas drop an octave (`C,` = C3)
    and apostrophes raise one (`c'` = C6). Black keys are written as sharps
    (`^F`) so the text means the same pitch under any K:.
    """
    pc = midi % 12
    octave = midi // 12 - 1  # MIDI 60 = C4
    if pc in _PC_LETTER:
        letter, accidental = _PC_LETTER[pc], ""
    else:
        # Sharps for the black keys: a flat spelling would depend on the key
        # signature, and this writer emits accidentals explicitly instead.
        letter, accidental = _PC_LETTER[pc - 1], "^"
    if octave <= 4:
        text = letter.upper() + "," * (4 - octave)
    else:
        text = letter.lower() + "'" * (octave - 5)
    return accidental + text


# The native dialect's supported duration multipliers relative to L: — anything
# else has to be expressed as several tokens (10 units = C8-C2 or C8C2).
_SUPPORTED_MULTIPLIERS = (32, 24, 16, 12, 8, 6, 4, 3, 2, 1)


def _duration_parts(sixteenths: int) -> list[int]:
    """Greedy split of a duration into supported multipliers of L:1/16."""
    if sixteenths <= 0:
        raise ValueError(f"duration must be positive; got {sixteenths}")
    parts: list[int] = []
    remaining = sixteenths
    for mult in _SUPPORTED_MULTIPLIERS:
        while remaining >= mult:
            parts.append(mult)
            remaining -= mult
    return parts


def _duration_text(sixteenths: int) -> str:
    """Sixteenth count → ABC length suffix relative to L:1/16."""
    if sixteenths not in _SUPPORTED_MULTIPLIERS and sixteenths != 1:
        raise ValueError(
            f"{sixteenths} is not a supported duration multiplier; "
            "split it with _duration_parts first"
        )
    return "" if sixteenths == 1 else str(sixteenths)


_PITCH_RE = re.compile(r"([_=^]*)([A-G])(,*)|([_=^]*)([a-g])('*)")
_DURATION_RE = re.compile(r"^([0-9]*)/?([0-9]*)")


def _abc_to_midi(text: str) -> int | None:
    """Parse an ABC pitch token back to a MIDI number (no key signature applied).

    Used by the tests and by anything that wants to read a score back as pitches;
    the writer itself never needs it, because accidentals are written explicitly.
    """
    m = _PITCH_RE.match(text)
    if not m:
        return None
    if m.group(2):
        letter, accidentals, octave_marks = m.group(2), m.group(1), m.group(3)
        octave = 4 - len(octave_marks)   # "C" = C4, "C," = C3
    else:
        letter, accidentals, octave_marks = m.group(5), m.group(4), m.group(6)
        octave = 5 + len(octave_marks)   # "c" = C5, "c'" = C6
    pc = _LETTER_PC.get(letter.upper())
    if pc is None:
        return None
    for mark in accidentals:
        pc += {"^": 1, "_": -1, "=": 0}.get(mark, 0)
    return (octave + 1) * 12 + pc


def validate(abc: str, *, meter: str = "4/4", unit: int = UNIT) -> list[str]:
    """Structural checks on ABC text. Returns a list of problems, empty when clean.

    This is a notation check, not a musical one: it confirms the file has the shape
    YuE2's melody condition expects and that every measure adds up. It cannot say
    whether the tune is good, and a passing validate does not mean the transcription
    matched what was sung — see `docs/honest-limits.md`.
    """
    problems: list[str] = []
    lines = [l for l in abc.splitlines() if l.strip() and not l.strip().startswith("%")]
    if not lines:
        return ["the ABC is empty"]

    headers = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip() for l in lines if ":" in l
               and l.split(":", 1)[0].strip() in ("X", "T", "M", "L", "Q", "K")}
    for required in ("X", "M", "L", "K"):
        if required not in headers:
            problems.append(f"missing required header {required}:")
    if headers.get("M") and headers["M"] != meter:
        problems.append(f"M: is {headers['M']}, expected {meter}")
    if headers.get("L") and headers["L"] != f"1/{unit}":
        problems.append(f"L: is {headers['L']}, expected 1/{unit}")
    if headers.get("K") and not re.fullmatch(r"[A-G][b#]?\s*(major|minor|maj|min)?", headers["K"]):
        problems.append(f"K: {headers['K']!r} is not a key letter with an optional mode")

    voices = [l for l in lines if l.startswith("V:")]
    names = [re.search(r"^V:\s*(\w+)", v).group(1) for v in voices if re.search(r"^V:\s*(\w+)", v)]
    if "Vocal" not in names:
        problems.append("no `V: Vocal` voice — YuE2's melody condition expects it")
    if "Ins" not in names:
        problems.append("no `V: Ins` voice — the native shape carries both, Ins resting")

    # Chord symbols in the music lines (comments don't count — an ABC % line is
    # inert text, and quoting a chord name in a comment is how examples document
    # what NOT to emit).
    music = "\n".join(l for l in lines if not l.strip().startswith("%"))
    if re.search(r'"[A-G][#b]?(m|maj|min|dim|aug|sus|7|9|add)?\d*"', music):
        problems.append("chord symbols found: cot=\"melody\" must be given a chord-free score")

    per_measure = _beats_per_measure(meter) * unit / 4.0  # one quarter beat = unit/4 of L:
    current = None
    for line in lines:
        if line.startswith("V:"):
            current = re.search(r"^V:\s*(\w+)", line).group(1)
            continue
        if current == "Ins" and re.fullmatch(r"Z\d+\|", line.strip()):
            continue  # a multi-measure rest covers the section by design
        if current != "Vocal":
            continue
        total = 0.0
        for _text, length in _tokens(line):
            total += length
        if total and abs(total - per_measure) > 1e-6:
            problems.append(
                f"measure does not sum to {per_measure:g} L: units (got {total:g}): "
                f"{line.strip()[:60]}"
            )
    return problems


def _tokens(line: str):
    """Yield `(text, sixteenths)` for each note or rest in a measure line."""
    body = line.split("|")[0]
    for m in re.finditer(r"([_=^]*(?:[A-G],*|[a-g]'*|z))([0-9]*/?[0-9]*)?", body):
        head, dur = m.group(1), m.group(2) or ""
        if head[-1] == "z":
            yield "z" + dur, _parse_duration(dur)
        else:
            yield head + dur, _parse_duration(dur)


def _parse_duration(text: str) -> float:
    """ABC length suffix → duration in units of L: (empty = 1, `2` = 2, `1/2` = 0.5)."""
    if not text:
        return 1.0
    m = _DURATION_RE.match(text)
    num = int(m.group(1)) if m.group(1) else 1
    den = int(m.group(2)) if m.group(2) else 1
    return max(num / den, 1 / UNIT)


def read(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def inspect(abc: str) -> dict:
    """Summary of an ABC score, for the CLI and for manifests."""
    lines = abc.splitlines()
    headers = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip()
               for l in lines if ":" in l and l.split(":", 1)[0].strip() in ("X", "T", "M", "L", "Q", "K")}
    sections = [l.strip()[2:].strip() for l in lines if l.strip().startswith("%")]
    vocal_lines = [l for l in lines if l and not l[0] in "XTMLQKVZ%|" and "|" in l]
    notes = sum(1 for l in vocal_lines for t, _ in _tokens(l))
    return {"headers": headers, "sections": sections, "measures": len(vocal_lines),
            "notes": notes, "chars": len(abc), "problems": validate(abc, meter=headers.get("M", "4/4"))}


def split_sections(melody, n: int) -> list[tuple[str, list[int]]]:
    """Split a note list into `n` equal named sections, as inclusive index pairs.

    Only used when the caller asks for a structure the recording cannot reveal. The
    names are positional (`part 1`, `part 2`) on purpose: calling one of them
    "chorus" would assert something nothing here can hear.
    """
    total = len(melody.notes)
    if n <= 1 or total < n:
        return [("melody", [(0, max(total - 1, 0))])]
    size = math.ceil(total / n)
    out = []
    for i in range(n):
        lo = i * size
        hi = min(lo + size - 1, total - 1)
        if lo > hi:
            break
        out.append((f"part {i + 1}", [(lo, hi)]))
    return out
