# Honest limits

What this tool cannot do, written down so nobody has to discover it by debugging
a bad song at 2 a.m.

## The melody condition is a condition

`cot="melody"` gives YuE2 your ABC as a symbolic plan to *condition* generation
on. It is not a constraint the sampler is forced to satisfy, and there is no
adherence score to check afterwards. Expect: the contour, the phrase structure,
roughly the rhythm. Do not expect: note-for-note fidelity, exact octaves, or your
exact tempo surviving the style prompt. If you need the generated song to *be* the
score, that is a different tool (a MIDI renderer), not a singing model.

## Monophonic, or it is garbage

pyin tracks one fundamental frequency per frame. Humming with words, two people,
a hum over a backing track, or a chord on a keyboard all produce a f0 that is
some average or some harmonic of what you meant, and the output ABC will be
confidently wrong. The voiced-probability diagnostic will often (not always) be
low when this happens — check it, and re-hum naked.

Related failure: **octave jumps**. If you hum low then suddenly an octave up, the
tracker can land on a harmonic (2× frequency) instead of the note. The ABC will
show a pitch 12 semitones off. This is a known limitation of autocorrelation
pitch trackers; the fix is to re-record with steadier register, or edit the note
in the ABC by hand — that is why the score is text.

## Rhythm is inferred, not measured

- **Tempo** = median inter-onset interval, folded into 60–180 BPM. Folding is
  necessary because "one note per beat" is an assumption: quarter-note humming at
  100 BPM and eighth-note humming at 200 BPM produce identical audio. If the
  song comes out at double or half the speed you felt, `--tempo` is the fix.
- **Meter** (3/4 vs 4/4) is estimated from note-duration periodicity — weak
  evidence on a hum with no accents. It defaults to 4/4 whenever the estimate is
  not clearly better, and `--meter` overrides it.
- **Quantisation** onto the grid (`--grid`, default 1/8) is lossy by design:
  rubato, swings and rushed onsets snap to the grid. Warnings are emitted when a
  note's quantised length differs from its measured length by >50%. If your tune
  breathes, transcribe with `--grid 1/16` and accept a busier score.
- Notes longer than a measure, or crossing a bar line, are **split into two
  attacks** rather than tied. The split is warned about. Ties are on the roadmap;
  until then a held note across a barline gets re-articulated.

## The transcription is not verified against your ear

Nothing in the pipeline listens back. `validate()` checks notation structure
(measure sums, chord-freeness, voice shape) — it cannot tell you the pitches are
what you sang. **Open the ABC in a viewer and read it** before spending 2.5 GPU
minutes on a song. This is a deliberate step in the workflow, not a gap in it.

## SheetSage2 is the heavier, better extractor — for songs, not hums

The official YuE2 cover workflow transcribes with SheetSage2 (MERT-v2 based). It
handles real recordings, full mixes and structure; it needs its own environment,
FFmpeg 6.1 and a ~4 GB model download. hum2song's pyin path is the zero-download
light path tuned for one clean voice close to a mic. For transcribing an existing
song to cover, use SheetSage2 (see the `yue2-music` skill); for capturing the
tune you just invented, humming + pyin is enough and much faster.

## YuE2 side

- Windows needs `--backend torch-eager` (no flash attention in official PyTorch
  Windows builds): ~12 tok/s on an RTX 3090, ~2.5 min per ~50 s song.
- 24 GB VRAM is the supported baseline. A GPU already holding a ComfyUI workflow
  will OOM or crawl — free it first.
- Generation is stochastic. Same ABC + same style, different seed = different
  song. The manifest records the seed; reuse it to reproduce.
- Weights are **CC BY-NC 4.0**: songs you make are non-commercial.

## Privacy

The recorder serves on `127.0.0.1` and writes to disk on your machine; the page
contains no analytics, no external resources, no fetch to anywhere else. The song
stage downloads YuE2 weights from Hugging Face on first use (after that, fully
offline is possible with `--offline` model caching). No audio you record ever
leaves the machine through this tool.
