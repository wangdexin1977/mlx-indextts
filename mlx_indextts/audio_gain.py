"""Constant output gain with measured true-peak headroom; no compression or EQ."""

from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import soundfile as sf


def boost_wav(path: Path, requested_db: float = 6.0, ceiling_db: float = -1.0) -> float:
    """Boost a complete WAV uniformly and atomically, leaving silent/hot audio alone.

    A small extra margin covers the meter's 0.1 dB reporting precision. MP3
    encoding can introduce new peaks; WAV/FLAC preserve the measured waveform.
    """
    if requested_db <= 0:
        return 0.0
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 FFmpeg，无法测量 Fish 输出真实峰值。")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
         "-af", "ebur128=peak=true:framelog=verbose", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    peaks = re.findall(r"Peak:\s+(-?\d+(?:\.\d+)?|-inf)\s+dBFS", result.stderr)
    if result.returncode or not peaks:
        raise RuntimeError("Fish 输出真实峰值测量失败；原始 WAV 已保留。")
    peak_db = float(peaks[-1])
    if not np.isfinite(peak_db):
        return 0.0
    gain_db = max(0.0, min(float(requested_db), float(ceiling_db) - 0.1 - peak_db))
    if gain_db <= 0:
        return 0.0
    temporary = path.with_name(path.stem + ".gain.wav")
    try:
        with sf.SoundFile(path) as source, sf.SoundFile(
            temporary, "w", samplerate=source.samplerate,
            channels=source.channels, subtype=source.subtype,
        ) as destination:
            for block in source.blocks(blocksize=262144, dtype="float64", always_2d=True):
                destination.write(block * (10 ** (gain_db / 20)))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return gain_db
