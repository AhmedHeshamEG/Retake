"""Isolated WhisperX forced-alignment worker for Retake.

The main application launches this file with the dedicated alignment Python so
CUDA Torch and its large dependency set cannot destabilize the editor runtime.
One invocation handles one device; the parent owns GPU-first/CPU-fallback policy.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import whisperx


def load_audio(ffmpeg: str, source: str, sample_rate: int = 16000) -> np.ndarray:
    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-nostdin", "-threads", "0",
            "-i", source, "-f", "s16le", "-ac", "1",
            "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-",
        ],
        capture_output=True,
        check=True,
    )
    return (
        np.frombuffer(result.stdout, np.int16)
        .flatten()
        .astype(np.float32)
        / 32768.0
    )


def main() -> int:
    request = json.load(sys.stdin)
    device = str(request["device"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Torch is unavailable in the alignment runtime")

    started = time.monotonic()
    audio = load_audio(str(request["ffmpeg"]), str(request["source_path"]))
    model, metadata = whisperx.load_align_model(
        language_code=str(request["language"]),
        device=device,
        model_dir=str(Path(request["model_dir"]).resolve()),
    )
    result = whisperx.align(
        request["segments"], model, metadata, audio, device,
        return_char_alignments=False,
    )
    output = {
        "device": device,
        "model": (
            "WAV2VEC2_ASR_BASE_960H"
            if str(request["language"]) == "en"
            else "whisperx-default"
        ),
        "elapsed_s": round(time.monotonic() - started, 3),
        "words": result.get("word_segments", []),
    }
    json.dump(output, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
