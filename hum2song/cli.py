"""hum2song — hum a melody, get a song.

    hum2song record                       # browser recorder (localhost, no upload)
    hum2song melody audio.wav -o run/     # audio → notes + key/tempo/meter
    hum2song abc run/ -o run/ --style …   # notes → YuE2-native chord-free ABC
    hum2song song run/ --lyrics …         # ABC → song via a YuE2 install
    hum2song all hum.wav -o run/ …        # the whole chain

Every stage writes its artifacts plus a manifest.json into the run directory; a
later stage reads the earlier one's manifest, so `all` is just the four stages
with the plumbing done for you. Stages are independently runnable — that is the
point of keeping them separate.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, abc as abcmod, melody as melodmod, song as songmod
from .audio import load as load_audio, save_wav, duration as audio_duration

MANIFEST = "manifest.json"


def _read_manifest(run: Path) -> dict:
    path = run / MANIFEST
    if not path.exists():
        raise SystemExit(f"{run}/{MANIFEST} not found — run the previous stage first (see hum2song --help)")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(run: Path, updates: dict) -> None:
    manifest = _read_manifest(run) if (run / MANIFEST).exists() else {"tool": "hum2song", "version": __version__}
    manifest.update(updates)
    (run / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _vocal_band(text: str) -> tuple[int, int]:
    """argparse type for `--vocal-band LO-HI`, in note names or MIDI numbers."""
    parts = text.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"expected LO-HI, e.g. D4-G5 or 62-79; got {text!r}")
    try:
        lo, hi = (abcmod.parse_pitch(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if lo >= hi:
        raise argparse.ArgumentTypeError(
            f"band must ascend: LO={parts[0]} is not below HI={parts[1]}")
    return (lo, hi)


def cmd_record(args) -> int:
    """Serve the zero-dependency browser recorder."""
    from . import record
    return record.serve(host=args.host, port=args.port, out=Path(args.out))


def cmd_melody(args) -> int:
    run = Path(args.output)
    run.mkdir(parents=True, exist_ok=True)
    samples, rate = load_audio(args.audio, work_dir=run)
    wav = save_wav(samples, run / "input.24k.wav")

    t0 = time.time()
    mel = melodmod.track(samples, rate, grid=args.grid, voiced_threshold=args.voiced)
    elapsed = time.time() - t0

    (run / "melody.json").write_text(json.dumps(mel.to_dict(), indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    _write_manifest(run, {
        "audio": {"source": str(Path(args.audio).resolve()), "trimmed_wav": str(wav),
                  "seconds": round(audio_duration(samples, rate), 3)},
        "melody": {"file": "melody.json", "notes": len(mel.notes), "tempo": mel.tempo,
                   "key": mel.key, "meter": mel.meter, "grid": args.grid,
                   "warnings": mel.warnings, "diagnostics": mel.diagnostics,
                   "tracking_seconds": round(elapsed, 2)},
    })
    print(f"melody: {len(mel.notes)} notes, {mel.key}, {mel.meter}, {mel.tempo} BPM "
          f"({elapsed:.1f} s) → {run / 'melody.json'}")
    for w in mel.warnings:
        print(f"  warning: {w}")
    print(f"  range: {mel.range()}  mean voiced prob: {mel.diagnostics['mean_voiced_probability']}")
    return 0


def cmd_abc(args) -> int:
    run = Path(args.output)
    manifest = _read_manifest(run)
    mel = melodmod.Melody.from_dict(json.loads((run / "melody.json").read_text(encoding="utf-8")))

    sections = None
    if args.sections and args.sections > 1:
        sections = abcmod.split_sections(mel, args.sections)
    score = abcmod.from_melody(mel, title=args.title or "", sections=sections,
                               key=args.key, meter=args.meter, tempo=args.tempo,
                               transpose=args.transpose, vocal_band=args.vocal_band)
    problems = abcmod.validate(abcmod.render(score), meter=score.meter)
    if problems:
        print("the generated ABC failed its own validation:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    out = abcmod.write(score, run / "melody.abc")
    _write_manifest(run, {
        "abc": {"file": "melody.abc", "title": score.title, "key": score.key,
                "meter": score.meter, "tempo": score.tempo,
                "sections": [(s.name, len(s.measures)) for s in score.sections],
                "measures": sum(len(s.measures) for s in score.sections),
                "notes": manifest["melody"]["notes"],
                "warnings": score.warnings, "validated": not problems},
    })
    print(f"abc: {out}  key={score.key} meter={score.meter} tempo={int(score.tempo)} "
          f"sections={[s.name for s in score.sections]}")
    for w in score.warnings[:5]:
        print(f"  warning: {w}")
    return 0


def cmd_song(args) -> int:
    run = Path(args.output)
    abc_path = run / "melody.abc"
    if not abc_path.exists():
        raise SystemExit(f"{abc_path} not found — run `hum2song abc` first")
    style = args.style or _read_manifest(run).get("song", {}).get("style")
    if not style:
        raise SystemExit("--style is required the first time (genre, instruments, vocal character, tempo)")
    lyrics = args.lyrics
    if args.lyrics_file:
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")
    if not lyrics:
        raise SystemExit("lyrics are required: --lyrics TEXT or --lyrics-file PATH (YuE2 sings words)")

    request = songmod.write_request(run / "request.json", style=style, lyrics=lyrics,
                                    seed=args.seed, cot=args.cot,
                                    song_id=(run.name or "song"))
    out_dir = Path(args.song_output) if args.song_output else run / f"song-{int(time.time())}"
    try:
        result = songmod.generate(request, abc_path, out_dir, cot=args.cot,
                                  backend=args.backend, seed=args.seed)
    except songmod.YuE2NotConfigured as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except subprocess.TimeoutExpired:
        print("YuE2 generation timed out; the output directory holds partial artifacts", file=sys.stderr)
        return 3

    _write_manifest(run, {"song": {"style": style, "lyrics_chars": len(lyrics),
                                   "request": str(request), "cot": args.cot,
                                   "seed": args.seed, "result": result}})
    if result.get("ok"):
        print(f"song: {result['audio']}")
        trunc = (result.get("result") or {}).get("truncated")
        if trunc and any(trunc.values()):
            print(f"  warning: generation truncated: {trunc}")
        return 0
    print("song generation FAILED (see run manifest 'song.result.stderr_tail'):", file=sys.stderr)
    print(result.get("stderr_tail", "")[-600:], file=sys.stderr)
    return 1


def cmd_all(args) -> int:
    ns = argparse.Namespace(output=args.output, audio=args.audio, grid=args.grid,
                            voiced=args.voiced)
    if cmd_melody(ns):
        return 1
    ns = argparse.Namespace(output=args.output, title=args.title, sections=args.sections,
                            key=args.key, meter=args.meter, tempo=args.tempo,
                            transpose=args.transpose, vocal_band=args.vocal_band)
    if cmd_abc(ns):
        return 1
    ns = argparse.Namespace(output=args.output, style=args.style, lyrics=args.lyrics,
                            lyrics_file=args.lyrics_file, seed=args.seed, cot=args.cot,
                            backend=args.backend, song_output=args.song_output)
    return cmd_song(ns)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="hum2song", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"hum2song {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="serve the browser recorder (records stay on this machine)")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8388)
    r.add_argument("--out", default="recordings")
    r.set_defaults(fn=cmd_record)

    m = sub.add_parser("melody", help="audio → quantised notes, key, tempo, meter")
    m.add_argument("audio", type=Path)
    m.add_argument("-o", "--output", type=Path, required=True)
    m.add_argument("--grid", choices=("1/4", "1/8", "1/16"), default="1/8")
    m.add_argument("--voiced", type=float, default=0.35, help="pyin voiced-probability threshold")
    m.set_defaults(fn=cmd_melody)

    a = sub.add_parser("abc", help="melody.json → YuE2-native chord-free ABC")
    a.add_argument("-o", "--output", type=Path, required=True, help="run directory holding melody.json")
    a.add_argument("--title", default="")
    a.add_argument("--key", help="override the estimated key (e.g. Eb)")
    a.add_argument("--meter", choices=("2/4", "3/4", "4/4", "6/8"), help="override the estimated meter")
    a.add_argument("--tempo", type=float, help="override the estimated tempo")
    a.add_argument("--sections", type=int, default=1, help="split into N equal named sections")
    a.add_argument("--transpose", type=int, default=0,
                   help="semitones to shift the notated melody; keeps its contour "
                        "and moves the key signature with it")
    a.add_argument("--vocal-band", type=_vocal_band, default=None, metavar="LO-HI",
                   help="fold the melody into this register, e.g. D4-G5 or 62-79 "
                        "(at least an octave wide); alters the contour where it folds")
    a.set_defaults(fn=cmd_abc)

    s = sub.add_parser("song", help="ABC → song via a YuE2 install (separate environment)")
    s.add_argument("-o", "--output", type=Path, required=True, help="run directory holding melody.abc")
    s.add_argument("--style", help="YuE2 style text: genre, instruments, vocal, tempo")
    s.add_argument("--lyrics", help="lyrics with [Verse]/[Chorus] section tags")
    s.add_argument("--lyrics-file", type=Path)
    s.add_argument("--cot", choices=("full", "melody", "off"), default="melody")
    s.add_argument("--backend", choices=("torch", "torch-eager", "vllm"), default="torch-eager")
    s.add_argument("--seed", type=int)
    s.add_argument("--song-output", type=Path, help="fresh directory for YuE2 artifacts")
    s.set_defaults(fn=cmd_song)

    al = sub.add_parser("all", help="melody + abc + song in one run directory")
    al.add_argument("audio", type=Path)
    al.add_argument("-o", "--output", type=Path, required=True)
    al.add_argument("--grid", choices=("1/4", "1/8", "1/16"), default="1/8")
    al.add_argument("--voiced", type=float, default=0.35)
    al.add_argument("--title", default="")
    al.add_argument("--key")
    al.add_argument("--meter", choices=("2/4", "3/4", "4/4", "6/8"))
    al.add_argument("--tempo", type=float)
    al.add_argument("--sections", type=int, default=1)
    al.add_argument("--transpose", type=int, default=0)
    al.add_argument("--vocal-band", type=_vocal_band, default=None, metavar="LO-HI")
    al.add_argument("--style", required=True)
    al.add_argument("--lyrics")
    al.add_argument("--lyrics-file", type=Path)
    al.add_argument("--cot", choices=("full", "melody", "off"), default="melody")
    al.add_argument("--backend", choices=("torch", "torch-eager", "vllm"), default="torch-eager")
    al.add_argument("--seed", type=int)
    al.add_argument("--song-output", type=Path)
    al.set_defaults(fn=cmd_all)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
