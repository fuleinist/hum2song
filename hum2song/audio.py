"""Audio ingest: whatever the user recorded, as mono 24 kHz float32 samples.

`soundfile` reads WAV/FLAC/OGG natively. Browser recorders (the `record` page) and
phone voice memos produce webm/m4a/aac, which soundfile cannot open, so those go
through ffmpeg — the only external binary this package needs.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24_000

# What a microphone actually produces on each platform. MediaRecorder gives webm
# (Chrome/Firefox) or mp4/aac (Safari); phone exports add m4a and amr.
_DECODE_WITH_FFMPEG = {".webm", ".m4a", ".aac", ".amr", ".mp4", ".mov", ".opus", ".mp3"}


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_to_wav(source: Path, target: Path) -> Path:
    """Transcode anything ffmpeg understands into 24 kHz mono WAV."""
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is not on PATH, so browser recordings (webm/m4a) cannot be read. "
            "Install ffmpeg, or re-export the recording as WAV/FLAC."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(target)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {source.name}: {proc.stderr.strip()[:400]}")
    if not target.exists() or target.stat().st_size < 1024:
        raise RuntimeError(f"ffmpeg produced no usable audio from {source.name}")
    return target


def load(source: Path | str, *, work_dir: Path | None = None) -> tuple[np.ndarray, int]:
    """Return `(samples, sample_rate)`: mono, float32, 24 kHz, finite.

    Non-decodable containers are transcoded first; when `work_dir` is given the
    intermediate WAV is kept there rather than in a temp directory, so a run's
    artifacts stay together.
    """
    import soundfile as sf

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"no such audio file: {path}")
    if path.suffix.lower() in _DECODE_WITH_FFMPEG:
        decoded = (work_dir or path.parent) / f"{path.stem}.24k.wav"
        if not decoded.exists():
            decode_to_wav(path, decoded)
        path = decoded

    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError(f"{path.name} decoded to zero samples")
    mono = samples.mean(axis=1)  # soundfile gives [frames, channels]
    if not np.isfinite(mono).all():
        raise ValueError(f"{path.name} contains NaN/Inf samples; re-record it")

    if rate != SAMPLE_RATE:
        import librosa
        mono = librosa.resample(mono, orig_sr=rate, target_sr=SAMPLE_RATE)
        rate = SAMPLE_RATE

    # Trim leading/trailing silence: a hum recorded by hand starts with a breath
    # and ends with a click, and both waste transcription time.
    mono = _trim_silence(mono, rate)
    if mono.size < 1025:
        raise ValueError(
            f"{path.name} is {mono.size / rate:.2f} s of sound after trimming; "
            "at least ~0.05 s is needed, and a useful melody is usually 5-30 s"
        )
    return mono, rate


def _trim_silence(samples: np.ndarray, rate: int, *, top_db: float = 40.0) -> np.ndarray:
    """Cut silence off both ends, leaving a 50 ms pad so no note is clipped."""
    import librosa

    pad = int(rate * 0.05)
    intervals = librosa.effects.split(samples, top_db=top_db)
    if intervals.size == 0:
        return samples
    start = max(0, int(intervals[0][0]) - pad)
    end = min(samples.size, int(intervals[-1][1]) + pad)
    return samples[start:end]


def duration(samples: np.ndarray, rate: int = SAMPLE_RATE) -> float:
    return float(samples.size) / float(rate)


def save_wav(samples: np.ndarray, target: Path, rate: int = SAMPLE_RATE) -> Path:
    """Write normalized mono audio, so downstream stages see a known scale."""
    import soundfile as sf

    peak = float(np.max(np.abs(samples))) or 1.0
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), (samples / peak).astype("float32"), rate, subtype="PCM_16")
    return target
