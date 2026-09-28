# hum2song

[![CI](https://github.com/fuleinist/hum2song/actions/workflows/ci.yml/badge.svg)](https://github.com/fuleinist/hum2song/actions/workflows/ci.yml)

Hum a melody. Get a song.

```
you, humming ──▶ pitch tracking ──▶ ABC notation ──▶ YuE2 ──▶ a real song
   (mic)         librosa/pyin      chord-free,       3B model    (audio.flac)
                                   YuE2-native
```

Inspired by [Dsdaw](https://www.huiwanai.com/dsdaw/) — "make music like chatting" —
but narrowed to the one thing a chat box can't do for you: turn the tune in your
head into a **symbolic, editable score**, then hand that score to a singing model.
Everything runs locally; nothing is uploaded anywhere.

## Why ABC in the middle

YuE2 generates songs from `style + lyrics`, and optionally from an **ABC score** as
a symbolic melody condition (`cot="melody"`). Going through ABC instead of straight
audio→audio buys three things:

1. **You can see and edit the tune** before it becomes a song — it's plain text.
2. **The model gets a clean melodic skeleton** rather than your recording's noise,
   breath and pitch wobble.
3. **Every run is reproducible**: the manifest records tempo, key, grid and seed.

The trade-off, stated up front: `cot="melody"` is a *condition*, not a constraint.
YuE2 follows the tune loosely; expect the contour and phrase structure, not a
note-perfect cover. See [docs/honest-limits.md](docs/honest-limits.md).

## Install

Two environments, because the two model families pin different dependencies
(the same split the official YuE docs draw between SheetSage2 and YuE2):

```bash
# 1. transcription side (this repo)
git clone https://github.com/fuleinist/hum2song && cd hum2song
python -m venv .venv && .venv/Scripts/pip install -e ".[test]"   # Windows
ffmpeg -version        # needed only for browser recordings (webm/m4a)

# 2. synthesis side (YuE2, separate venv, 24 GB GPU)
git clone https://github.com/multimodal-art-projection/YuE && cd YuE
python -m venv .venv && .venv/Scripts/pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

Point hum2song at the YuE2 interpreter (or let it find `YuE/.venv` on its own):

```bash
set HUM2SONG_YUE2_PYTHON=G:/dev/AI/YuE/.venv/Scripts/python.exe   # Windows
export HUM2SONG_YUE2_PYTHON=/path/to/YuE/.venv/bin/python         # POSIX
```

YuE2-3B and YuE2-Vae weights download on first use from
[`m-a-p/YuE2-3B`](https://huggingface.co/m-a-p/YuE2-3B) (~7 GB, CC BY-NC 4.0 —
non-commercial).

## Use

### Record

```bash
hum2song record            # opens http://127.0.0.1:8388 — press ●, hum, stop
```

The recorder is a localhost page: your browser captures audio, POSTs it to the
server on your own machine, and it lands in `recordings/`. No account, no upload,
no network. Hum **one line** — this is a monophonic tracker, not a polyphonic one.
5–30 seconds, reasonably steady pitch, close to the mic.

### Transcribe and notate

```bash
hum2song melody recordings/hum-….webm -o run1/     # → melody.json + manifest
hum2song abc -o run1/ --title "My tune"            # → melody.abc
```

`melody` runs pyin pitch tracking, estimates tempo from your note onsets, key
(Krumhansl-Schmuckler) and meter, quantises onto a 1/8 grid, and reports what it
is unsure about — a low voiced-probability or a coarse-grid warning is the tool
telling you the recording, not lying to you.

`abc` renders YuE2's native dialect: two voices (`Vocal` melody, `Ins` resting),
`L:1/16`, one measure per line, **no chord symbols**, and every measure validated
to sum to the meter before it is written.

Open `run1/melody.abc` in any ABC viewer ([abcjs editor](https://editor.drawthedots.com/),
EasyABC) to see and edit the tune. Fix a wrong note as text; it's faster than
re-humming.

`abc` also sets the **register** the melody is written in — what you want when the
tune was hummed low and the singer you have in mind sits higher:

```bash
hum2song abc -o run1/ --transpose 5        # a fourth up; contour kept, key moves with it
hum2song abc -o run1/ --vocal-band D4-G5   # fold into a register, e.g. an alto lead
```

`--transpose` shifts every note by the same interval, so the tune's shape is exact
and the key signature moves with it (`C` up 5 lands on `F`). `--vocal-band` takes
`LO-HI` as note names or MIDI numbers and folds whatever falls outside back in by
octaves; that folding **does** alter the contour of the notes it touches, and a band
must be at least an octave wide, because octave displacement cannot reach a narrower
window. Neither flag chooses the singer — see
[docs/honest-limits.md](docs/honest-limits.md).

### Sing it

```bash
hum2song song -o run1/ \
  --style "English, warm piano pop, expressive female voice, acoustic piano, rounded bass, light drums, 96 BPM" \
  --lyrics "[Verse]
Neon fades along the lane
Footsteps keep the time of rain

[Chorus]
Let the day come into view
Every road begins with you"
```

→ `run1/song-<ts>/audio.flac` plus the full YuE2 artifact set. On Windows the
default `--backend torch-eager` is required (official PyTorch Windows builds lack
flash attention); ~2.5 min for a ~50 s song on an RTX 3090.

### Or all of it at once

```bash
hum2song all recordings/hum-….webm -o run1/ --title "My tune" --style "…" --lyrics "…"
```

## The run directory

One directory per tune, one manifest that chains the stages:

```
run1/
  input.24k.wav     # trimmed, normalized input
  melody.json       # notes with onsets, durations, voiced probabilities
  melody.abc        # the editable score
  request.json      # exact YuE2 style/lyrics/cot/seed
  song-<ts>/        # YuE2 artifacts: audio.flac, result.json, score.abc, latents
  manifest.json     # every stage's inputs, outputs, warnings, timings
```

Stages are independently runnable — re-notate after editing the melody, or
re-sing the same ABC with a different style — because each reads the previous
stage's files, not its memory.

## Tests

```bash
.venv/Scripts/python -m pytest -q        # 26 tests, no GPU, no models, ~2 s
```

The melody test runs the real tracker against a synthesized "Twinkle Twinkle"
whose 14 pitches are known by construction, so a failure is a tracking bug, not a
bad recording. ABC tests build melodies by hand and check the notation invariants
(measure sums, pitch round-trips, chord-freeness). The song stage is tested only
for its failure modes — a missing YuE2 must produce an actionable message, never a
traceback or a fake success.

## Honest limits

- `cot="melody"` conditions, it does not constrain — the output follows your
  tune's shape, not its notes exactly.
- pyin tracks **one** pitched line. Two voices, chords or a busy accompaniment
  will produce garbage; hum naked.
- Tempo is your median note spacing folded into 60–180 BPM: quarter-note humming
  and eighth-note humming are indistinguishable to it. Override with `--tempo`.
- Meter detection is weak on a hum (no accents to hear); `--meter` exists because
  the guess is a guess.
- For transcribing **existing songs** (not your hums), SheetSage2 is the better
  tool — it is what the official YuE2 cover workflow uses. This repo's tracker is
  the light path that works with zero model downloads.
- YuE2 weights are **CC BY-NC 4.0**. What you make with them inherits that:
  non-commercial use.

Full list: [docs/honest-limits.md](docs/honest-limits.md).

## Tests

CI runs the suite on **Ubuntu and Windows**, on **Python 3.10 and 3.12**, for every
push to `main` and every pull request
([workflow](.github/workflows/ci.yml)). Nothing in the suite needs a GPU or a YuE2
install: the transcription side is CPU-only, and synthesis is deliberately kept in a
separate environment.

```bash
pip install -e ".[test]"
pytest -q
```

One test in `tests/test_melody.py` runs pyin end to end, so the suite takes roughly
25 s on CPU rather than a fraction of a second. The rest construct `Melody` objects
by hand, so a failure names the notation bug instead of a tracking wobble.

## Credits

- [YuE2](https://github.com/multimodal-art-projection/YuE) — m-a-p, Apache-2.0 code / CC BY-NC weights
- [librosa](https://librosa.org/) — ISC; pyin after Mauch & Dixon (2014)
- [Dsdaw](https://www.huiwanai.com/dsdaw/) — the "music like chatting" idea this narrows
- hum2song itself: Apache-2.0
