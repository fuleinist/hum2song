"""Stage 2: audio → melody.

Pitch tracking with `librosa.pyin` (a probabilistic YIN — the right choice for one
monophonic voice or whistle; `piptrack` assumes an instrument spectrum), then note
segmentation, grid quantisation, and key/tempo/meter estimates.

Everything here is measurable, and the measurement is returned rather than hidden:
each note carries its voiced probability, so a shaky hum is visible as a low number
instead of silently becoming a wrong pitch.

The output is a plain data structure (`Melody`) that `abc.py` renders. Nothing in
this module knows about ABC.
"""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np

# One hop everywhere in this module, so frame indices from pyin, beat_track and
# onset_strength all address the same time grid. librosa's own default, stated
# explicitly because three functions have to agree on it.
HOP_LENGTH = 512
FRAME_LENGTH = 1024

# Krumhansl-Schmuckler key profiles, the standard correlation baseline for key
# estimation from a pitch-class histogram. Major first, then natural minor.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# ABC key signatures use these spellings; # / b both appear in real tunes.
_KEY_TO_ABC = {"C#": "C#", "D#": "Eb", "F#": "F#", "G#": "Ab", "A#": "Bb"}


@dataclasses.dataclass(frozen=True)
class Note:
    """One sounding note on the quantised grid."""

    midi: int            # 0-127
    onset: float         # seconds
    duration: float      # seconds
    probability: float   # mean pyin voiced probability, 0-1
    beats: float = 0.0   # onset in quarter-note beats
    length_beats: float = 0.0

    @property
    def name(self) -> str:
        return f"{_PITCH_NAMES[self.midi % 12]}{self.midi // 12 - 1}"


@dataclasses.dataclass
class Melody:
    notes: list[Note]
    sample_rate: int
    duration: float
    tempo: float
    key: str              # e.g. "C major", "A minor"
    meter: str            # "4/4" or "3/4"
    abc_key: str          # the letter to put after K: — "C", "Eb", "A"
    warnings: list[str] = dataclasses.field(default_factory=list)
    diagnostics: dict = dataclasses.field(default_factory=dict)

    @property
    def pitches(self) -> list[int]:
        return [n.midi for n in self.notes]

    def range(self) -> tuple[int, int] | None:
        if not self.notes:
            return None
        return min(self.pitches), max(self.pitches)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["notes"] = [dataclasses.asdict(n) for n in self.notes]
        d["range"] = self.range()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Melody":
        notes = [Note(**n) for n in d.pop("notes")]
        d.pop("range", None)
        return cls(notes=notes, **d)


def track(samples: np.ndarray, sample_rate: int, *,
          fmin: float = 65.0, fmax: float = 1400.0,
          voiced_threshold: float = 0.35,
          min_note_seconds: float = 0.07,
          grid: str = "1/8") -> Melody:
    """Follow one pitched line through `samples` and return a `Melody`.

    `voiced_threshold` is the pyin probability below which a frame counts as
    unvoiced — raising it drops breaths and raspberries at the cost of splitting
    soft notes. `grid` quantises onsets and lengths (``1/4``, ``1/8``, ``1/16``).
    """
    import librosa

    warnings: list[str] = []
    if samples.size / sample_rate < 0.5:
        warnings.append("input is under 0.5 s; a melody this short will not survive quantisation")

    f0, voiced_flag, voiced_prob = librosa.pyin(
        samples, fmin=fmin, fmax=fmax, sr=sample_rate,
        frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH, fill_na=np.nan,
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sample_rate, hop_length=HOP_LENGTH)

    raw = _segment(f0, voiced_flag, voiced_prob, times, sample_rate, voiced_threshold, min_note_seconds)
    if not raw:
        raise ValueError(
            "no pitched notes found: the recording is too quiet, too noisy, or not monophonic. "
            "Re-record closer to the mic and hum one line at a time."
        )

    # Tempo for a hummed line: the gaps between note onsets ARE the beat grid.
    # librosa.beat_track needs percussive onsets and returns garbage (0 BPM) on a
    # smooth monophonic sine, so it is only the fallback when there are too few
    # notes to estimate from.
    tempo, tempo_source = _tempo_from_onsets(raw, samples, sample_rate)
    if tempo_source == "beat_track":
        warnings.append(f"too few notes for onset-interval tempo; beat_track fallback gave {tempo:.1f} BPM")

    key, mode = _estimate_key(raw)
    meter, meter_score = _estimate_meter(raw)
    notes, grid_warnings = _quantise(raw, tempo, grid)
    warnings.extend(grid_warnings)

    if len(notes) < 3:
        warnings.append(f"only {len(notes)} note(s) survived quantisation; hum something longer")
    voiced = float(np.nanmean(voiced_prob[voiced_flag])) if voiced_flag.any() else 0.0
    if voiced < 0.5:
        warnings.append(f"mean voiced probability is low ({voiced:.2f}); pitches are shaky")

    diagnostics = {
        "frames": int(len(f0)),
        "voiced_frames": int(voiced_flag.sum()),
        "mean_voiced_probability": round(voiced, 4),
        "raw_segments": len(raw),
        "quantised_notes": len(notes),
        "grid": grid,
        "hop_length": HOP_LENGTH,
        "tempo_source": tempo_source,
        "meter_score": round(meter_score, 4),
        "pitch_classes": dict(Counter(n.midi % 12 for n in notes)),
    }
    return Melody(notes=notes, sample_rate=sample_rate,
                  duration=float(samples.size) / sample_rate, tempo=round(tempo, 1),
                  key=f"{key} {mode}", meter=meter, abc_key=_KEY_TO_ABC.get(key, key),
                  warnings=warnings, diagnostics=diagnostics)


def _segment(f0, voiced_flag, voiced_prob, times, sample_rate,
             threshold: float, min_seconds: float) -> list[dict]:
    """Group consecutive frames into raw (unquantised) note segments."""
    midi = np.where(np.isfinite(f0) & (f0 > 0), librosa_note(f0), np.nan)
    rounded = np.round(midi)
    segments: list[dict] = []
    start = None

    for i in range(len(rounded)):
        ok = bool(voiced_flag[i]) and np.isfinite(rounded[i]) and float(voiced_prob[i]) >= threshold
        if not ok:
            if start is not None:
                segments.append(_close(segments, start, i, rounded, voiced_prob, times, sample_rate))
                start = None
            continue
        if start is None:
            start = i
        elif abs(rounded[i] - rounded[start]) >= 1.0 and _mean(rounded, start, i) != rounded[i]:
            # A semitone move is a new note; jitter within a held note is not.
            segments.append(_close(segments, start, i, rounded, voiced_prob, times, sample_rate))
            start = i
    if start is not None:
        segments.append(_close(segments, start, len(rounded), rounded, voiced_prob, times, sample_rate))

    kept = [s for s in segments if s["duration"] >= min_seconds]
    if len(kept) < len(segments):
        # Dropping sub-threshold blips can leave a note that was really two: rejoin
        # consecutive segments of the same pitch separated by a dropped blip only.
        kept = _rejoin(kept)
    return kept


def _close(_segments, start: int, end: int, rounded, voiced_prob, times, sample_rate) -> dict:
    pitches = rounded[start:end]
    probs = voiced_prob[start:end]
    onset = float(times[start])
    offset = float(times[end - 1]) if end - 1 < len(times) else float(times[-1])
    # Add one hop so the last frame's duration is counted, not its onset.
    offset += float(times[1] - times[0]) if len(times) > 1 else 0.0
    counts = Counter(int(p) for p in pitches if np.isfinite(p))
    midi = counts.most_common(1)[0][0] if counts else int(round(float(np.nanmean(pitches))))
    return {
        "midi": int(midi),
        "onset": onset,
        "duration": max(offset - onset, 1.0 / sample_rate),
        "probability": float(np.nanmean(probs)) if len(probs) else 0.0,
        "frames": int(end - start),
    }


def _rejoin(segments: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in segments:
        if out and out[-1]["midi"] == s["midi"] and s["onset"] - (out[-1]["onset"] + out[-1]["duration"]) < 0.06:
            prev = out[-1]
            prev["duration"] = s["onset"] + s["duration"] - prev["onset"]
            prev["probability"] = float((prev["probability"] + s["probability"]) / 2)
            prev["frames"] += s["frames"]
        else:
            out.append(dict(s))
    return out


def _mean(rounded, a: int, b: int):
    vals = rounded[a:b]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else np.nan


def librosa_note(f0):
    """f0 (Hz) → fractional MIDI, vectorised, NaN-safe."""
    import librosa
    return librosa.hz_to_midi(np.where(f0 > 0, f0, np.nan))


def _estimate_key(segments: list[dict]) -> tuple[str, str]:
    """Krumhansl-Schmuckler over a duration-weighted pitch-class histogram."""
    hist = np.zeros(12)
    for s in segments:
        hist[s["midi"] % 12] += s["duration"]
    if hist.sum() == 0:
        return "C", "major"
    best = ("C", "major", -2.0)
    for shift in range(12):
        for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            r = _corr(np.roll(profile, shift), hist)
            if r > best[2]:
                best = (_PITCH_NAMES[shift], mode, float(r))
    return best[0], best[1]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 0 else 0.0


def _tempo_from_onsets(segments: list[dict], samples: np.ndarray, sample_rate: int
                       ) -> tuple[float, str]:
    """Tempo from note onsets; the median inter-onset interval is one beat.

    A hum has no drums, so the note grid is the only beat evidence there is. The
    median interval is converted to BPM assuming one note per beat, then folded
    into 60-180 BPM (a 0.6 s note is 100 BPM, not 25 or 400). Fold, never clip:
    clipping invents a tempo that was never in the signal.
    """
    import librosa

    if len(segments) >= 4:
        onsets = np.array([s["onset"] for s in segments])
        intervals = np.diff(onsets)
        intervals = intervals[intervals > 0.05]  # drop re-detections of one note
        if intervals.size >= 3:
            median = float(np.median(intervals))
            if median > 0:
                bpm = 60.0 / median
                # The median inter-onset interval is taken as one beat. That
                # cannot tell quarter-note humming from eighth-note humming —
                # both are self-consistent — so fold into 60-180 BPM, the range
                # where a sung line lives, and let --tempo override. Folding,
                # not clipping: clipping invents a tempo the signal never had.
                while bpm < 60.0:
                    bpm *= 2.0
                while bpm >= 180.0:
                    bpm /= 2.0
                return round(bpm, 1), "onsets"

    tempo, _frames = librosa.beat.beat_track(
        y=samples, sr=sample_rate, hop_length=HOP_LENGTH, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    if not (20.0 <= tempo <= 300.0):
        tempo = 120.0  # the notation default; the warning tells the user
    return round(tempo, 1), "beat_track"


def _estimate_meter(segments: list[dict]) -> tuple[str, float]:
    """3/4 vs 4/4 from note-duration periodicity; returns `(meter, score)`.

    A waltz groups notes in threes, so per-note durations correlate with
    themselves at lag 3. On a hum with no dynamic accents this is weak evidence —
    which is why the score is returned and stored in diagnostics, and why the CLI
    exposes `--meter` to override it. Default 4/4 whenever nothing beats 0.25.
    """
    if len(segments) < 9:
        return "4/4", 0.0
    d = np.array([s["duration"] for s in segments], dtype=float)
    d -= d.mean()
    norm = float(d @ d)
    if norm <= 0:
        return "4/4", 0.0
    best = ("4/4", 0.0)
    for lag, meter in ((3, "3/4"), (4, "4/4"), (2, "2/4")):
        if len(d) > lag:
            score = float(d[:-lag] @ d[lag:]) / norm
            if score > best[1]:
                best = (meter, score)
    return (best[0], best[1]) if best[1] > 0.25 else ("4/4", best[1])


def _quantise(segments: list[dict], tempo: float, grid: str) -> tuple[list[Note], list[str]]:
    """Snap onsets and durations onto a rhythmic grid measured in quarter beats."""
    warnings: list[str] = []
    try:
        num, den = (int(x) for x in grid.split("/"))
    except ValueError:
        raise ValueError(f"grid must look like '1/8'; got {grid!r}")
    step = num / den  # fraction of a quarter note per grid cell
    beats_per_second = tempo / 60.0

    notes: list[Note] = []
    for s in segments:
        b_on = s["onset"] * beats_per_second
        b_len = max(s["duration"] * beats_per_second, step)
        q_on = round(b_on / step) * step
        q_len = max(round(b_len / step) * step, step)
        if abs(q_len - b_len) / b_len > 0.5:
            warnings.append(
                f"{_PITCH_NAMES[s['midi'] % 12]} at {s['onset']:.2f} s: {b_len:.2f} beats "
                f"quantised to {q_len:.2f} — the grid is coarser than the rhythm"
            )
        notes.append(Note(midi=s["midi"], onset=s["onset"], duration=s["duration"],
                          probability=round(s["probability"], 4),
                          beats=round(q_on, 4), length_beats=round(q_len, 4)))

    # A note whose onset lands before the previous note ends is impossible to
    # notate monophonically: shorten the earlier one rather than losing a pitch.
    for a, b in zip(notes, notes[1:]):
        if a.beats + a.length_beats > b.beats + 1e-6:
            a.length_beats = max(step, round((b.beats - a.beats) / step) * step)
    if warnings:
        warnings = warnings[:5] + ([f"... and {len(warnings) - 5} more"] if len(warnings) > 5 else [])
    return notes, warnings


def load_audio(path: Path | str, **kwargs) -> tuple[np.ndarray, int]:
    from . import audio
    return audio.load(path, **kwargs)
