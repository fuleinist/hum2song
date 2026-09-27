"""Stage 4: abc → song, by delegating to a YuE2 install.

hum2song does no synthesis of its own. YuE2 is a separate model with its own
pinned runtime (torch 2.10, transformers 4.57) that does not coexist with the
librosa/soundfile environment used for transcription, so the two stages run in
two interpreters and exchange files — the same boundary the `yue2-music` skill
draws between SheetSage2 and YuE2.

This module shells out to `yue2.cli generate` in the YuE2 environment. It never
imports torch. When no YuE2 install is configured it says so plainly and stops,
with the exact command to run by hand, rather than pretending to synthesize.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


class YuE2NotConfigured(RuntimeError):
    """Raised when there is no usable YuE2 interpreter to delegate to."""


# Well-known local checkouts, last in the search order. A module constant so a
# test can empty it and exercise the "no YuE2 installed" path on a machine that
# does have one.
LOCAL_CANDIDATES = (
    Path("G:/dev/AI/YuE/.venv/Scripts/python.exe"),
    Path("G:/dev/AI/YuE/.venv/bin/python"),
)


def _yue2_python() -> tuple[str, list[str]]:
    """Resolve the interpreter that can run `python -m yue2.cli`.

    Looked for, in order:
      1. `HUM2SONG_YUE2_PYTHON` — an explicit interpreter, the supported path.
      2. a `.venv` inside the YuE checkout named by `HUM2SONG_YUE2_HOME`.
      3. the well-known local checkout in `LOCAL_CANDIDATES`.

    Returns `(python_exe, module_prefix)`. Raises `YuE2NotConfigured` if none is
    found — which is the normal state on a fresh clone and is not an error to fix
    by guessing.
    """
    explicit = os.environ.get("HUM2SONG_YUE2_PYTHON")
    if explicit:
        exe = shutil.which(explicit) or (Path(explicit).as_posix() if Path(explicit).exists() else None)
        if not exe:
            raise YuE2NotConfigured(f"HUM2SONG_YUE2_PYTHON={explicit!r} does not exist")
        return exe, []

    candidates = []
    home = os.environ.get("HUM2SONG_YUE2_HOME")
    if home:
        candidates.append(Path(home) / ".venv" / "Scripts" / "python.exe")
        candidates.append(Path(home) / ".venv" / "bin" / "python")
    candidates.extend(LOCAL_CANDIDATES)

    for exe in candidates:
        if exe.exists():
            return exe.as_posix(), []
    raise YuE2NotConfigured(
        "no YuE2 interpreter found. Set HUM2SONG_YUE2_PYTHON to a python that can "
        "`import yue2` (the YuE checkout's own .venv), or run the song stage by hand:\n"
        "  <yue2-python> -m yue2.cli generate --request request.json "
        "--abc-file melody.abc --cot melody --backend torch-eager --output song/"
    )


def write_request(path: Path, *, style: str, lyrics: str, seed: int | None = None,
                  abc: str | None = None, cot: str = "melody") -> Path:
    """Write a YuE2 request JSON. `style`/`lyrics` are free text; no impl notes."""
    req = {"id": Path(path).stem, "style": style, "lyrics": lyrics, "cot": cot}
    if abc is not None:
        req["abc"] = abc
    if seed is not None:
        req["seed"] = int(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(req, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def generate(request: Path, abc_file: Path, output: Path, *, cot: str = "melody",
             backend: str = "torch-eager", model: str = "m-a-p/YuE2-3B",
             vae: str = "m-a-p/YuE2-Vae", seed: int | None = None,
             timeout: int = 1800) -> dict:
    """Run YuE2 generation in its own interpreter. Returns a result manifest.

    `backend="torch-eager"` is the Windows default: official PyTorch Windows builds
    lack flash attention, and the eager backend runs at ~12 tok/s on an RTX 3090
    (~2.5 min for a ~50 s song). Output must be a fresh directory — YuE2 refuses to
    overwrite, so each attempt is retained.
    """
    exe, prefix = _yue2_python()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output {output} is not empty; choose a fresh directory to retain versions")

    cmd = [exe, *prefix, "-m", "yue2.cli", "generate",
           "--request", str(request), "--abc-file", str(abc_file), "--cot", cot,
           "--backend", backend, "--model", model, "--vae", vae,
           "--output", str(output), "--device", "cuda"]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    audio = output / "audio.flac"
    manifest = {
        "ok": proc.returncode == 0 and audio.exists(),
        "interpreter": exe,
        "command": cmd,
        "returncode": proc.returncode,
        "audio": str(audio) if audio.exists() else None,
        "output_dir": str(output),
        "stderr_tail": proc.stderr.strip()[-1200:],
    }
    result_json = output / "result.json"
    if result_json.exists():
        try:
            manifest["result"] = json.loads(result_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            manifest["result_error"] = str(exc)
    if not manifest["ok"]:
        manifest["error"] = (
            "YuE2 generation failed. This is the synthesis stage, in its own environment; "
            "the ABC and melody artifacts from earlier stages are unaffected. Read stderr_tail."
        )
    return manifest
