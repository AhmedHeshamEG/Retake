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
    groups = None
    if request.get("separate"):
        # One small window per run of words the first pass could not place.
        # Aligned one at a time, each keeps its own words: whisperx may split a
        # segment into sentences, which would lose track of which run is which.
        # Each window is handed only its own slice of the recording; handing
        # whisperx the whole file costs seconds per window on a long take.
        groups = []
        sample_rate = 16000
        for segment in request["segments"]:
            try:
                first = max(0, int(float(segment["start"]) * sample_rate))
                last = min(len(audio), int(float(segment["end"]) * sample_rate) + 1)
                offset = first / sample_rate
                local = {
                    "start": 0.0, "end": (last - first) / sample_rate,
                    "text": segment["text"],
                }
                aligned = whisperx.align(
                    [local], model, metadata, audio[first:last], device,
                    return_char_alignments=False,
                )
                words = []
                for word in aligned.get("word_segments", []):
                    word = dict(word)
                    for key in ("start", "end"):
                        if word.get(key) is not None:
                            word[key] = float(word[key]) + offset
                    words.append(word)
                groups.append(words)
            except Exception:
                groups.append([])
        result = {"word_segments": [word for group in groups for word in group]}
    else:
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
    if groups is not None:
        output["groups"] = groups
    json.dump(output, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
