"""CLI tests: the melody→abc→manifest chain on synthetic audio, no models.

The song stage is not run here — it needs a YuE2 environment and a GPU. What is
tested is that the stages chain, that manifests are honest, and that the song
stage fails with an actionable message when YuE2 is absent rather than pretending.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hum2song import cli
from tests import fixtures


@pytest.fixture(scope="module")
def twinkle_wav(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "twinkle.wav"
    sf.write(str(path), fixtures.twinkle_audio(bpm=100.0), fixtures.SAMPLE_RATE, subtype="PCM_16")
    return path


def test_melody_stage_writes_artifacts_and_manifest(twinkle_wav, tmp_path):
    run = tmp_path / "run1"
    assert cli.main(["melody", str(twinkle_wav), "-o", str(run)]) == 0

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["melody"]["notes"] == len(fixtures.TWINKLE)
    assert manifest["melody"]["key"] == "C major"
    melody = json.loads((run / "melody.json").read_text(encoding="utf-8"))
    assert [n["midi"] for n in melody["notes"]] == fixtures.TWINKLE
    assert (run / "input.24k.wav").exists()


def test_abc_stage_chains_from_melody(twinkle_wav, tmp_path):
    run = tmp_path / "run2"
    assert cli.main(["melody", str(twinkle_wav), "-o", str(run)]) == 0
    assert cli.main(["abc", "-o", str(run), "--title", "Twinkle"]) == 0

    text = (run / "melody.abc").read_text(encoding="utf-8")
    assert "T:Twinkle" in text and "K:C" in text
    assert '"G7"' not in text  # chord-free, always
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["abc"]["validated"] is True
    assert manifest["abc"]["notes"] == len(fixtures.TWINKLE)


def test_abc_stage_without_melody_fails_with_guidance(tmp_path, capsys):
    run = tmp_path / "empty"
    run.mkdir()
    with pytest.raises(SystemExit, match="previous stage"):
        cli.main(["abc", "-o", str(run)])


def test_song_stage_without_yue2_is_actionable(tmp_path, capsys, monkeypatch):
    # No YuE2 anywhere: the stage must say what to set, not crash with a traceback.
    from hum2song import song as songmod
    monkeypatch.delenv("HUM2SONG_YUE2_PYTHON", raising=False)
    monkeypatch.delenv("HUM2SONG_YUE2_HOME", raising=False)
    monkeypatch.setattr(songmod, "LOCAL_CANDIDATES", ())
    run = tmp_path / "run3"
    run.mkdir()
    (run / "melody.abc").write_text("X:1\n", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"tool": "hum2song"}), encoding="utf-8")
    rc = cli.main(["song", "-o", str(run), "--style", "warm pop", "--lyrics", "[Verse]\nla la"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "HUM2SONG_YUE2_PYTHON" in err and "yue2.cli generate" in err


def test_song_stage_requires_style_and_lyrics(tmp_path, capsys):
    run = tmp_path / "run4"
    run.mkdir()
    (run / "melody.abc").write_text("X:1\n", encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({"tool": "hum2song"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="--style"):
        cli.main(["song", "-o", str(run), "--lyrics", "x"])
    with pytest.raises(SystemExit, match="lyrics are required"):
        cli.main(["song", "-o", str(run), "--style", "pop"])


def test_request_json_shape(tmp_path):
    from hum2song import song
    req = song.write_request(tmp_path / "request.json", style="dream pop, female vocal",
                             lyrics="[Verse]\ntest", seed=42, cot="melody")
    data = json.loads(req.read_text(encoding="utf-8"))
    assert data["cot"] == "melody" and data["seed"] == 42
    assert data["style"].startswith("dream pop")
    assert data["id"] == "song"  # YuE2 nests artifacts under <output>/<id>/
    assert "abc" not in data  # ABC travels via --abc-file, not duplicated in the request


def test_generate_finds_audio_nested_under_the_request_id(tmp_path, monkeypatch):
    # Regression: YuE2's save_artifacts writes <output>/<id>/audio.flac, and an
    # earlier version looked only at <output>/audio.flac — reporting ok=False for
    # a generation that had actually succeeded. The fake subprocess writes the
    # nested layout; generate must find it.
    from hum2song import song as songmod

    out = tmp_path / "song-out"
    nested = out / "run1"

    class _Proc:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kwargs):
        nested.mkdir(parents=True)
        (nested / "audio.flac").write_bytes(b"fake-flac")
        (nested / "result.json").write_text(json.dumps({"truncated": {}}), encoding="utf-8")
        return _Proc()

    monkeypatch.setattr(songmod.subprocess, "run", fake_run)
    monkeypatch.setattr(songmod, "_yue2_python", lambda: ("python", []))
    result = songmod.generate(tmp_path / "request.json", tmp_path / "melody.abc", out)
    assert result["ok"] is True
    assert result["audio"].endswith(str(nested / "audio.flac").replace("\\", "/")) or \
           result["audio"] == str(nested / "audio.flac")
    assert result["artifacts_dir"] == str(nested)
    assert result["result"] == {"truncated": {}}


def test_generate_reports_failure_without_audio(tmp_path, monkeypatch):
    from hum2song import song as songmod

    class _Proc:
        returncode = 1
        stderr = "CUDA out of memory"

    monkeypatch.setattr(songmod.subprocess, "run", lambda *a, **k: _Proc())
    monkeypatch.setattr(songmod, "_yue2_python", lambda: ("python", []))
    result = songmod.generate(tmp_path / "request.json", tmp_path / "melody.abc",
                              tmp_path / "out2")
    assert result["ok"] is False and result["audio"] is None
    assert "CUDA out of memory" in result["stderr_tail"]
    assert "earlier stages are unaffected" in result["error"]
