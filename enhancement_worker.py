"""Isolated DeepFilterNet3 worker for conservative Retake noise cleanup.

The parent process decides whether noise is meaningful and owns all fallbacks.
This worker processes one lossless edited-audio stem, preserves its channels,
compensates model delay through the official API, and returns one aligned stem.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torchaudio
from df.enhance import enhance, init_df


def main() -> int:
    request = json.load(sys.stdin)
    source = Path(str(request["source_path"])).resolve()
    output = Path(str(request["output_path"])).resolve()
    model_dir = Path(str(request["model_dir"])).resolve()
    cleanup = max(0.0, min(100.0, float(request["noise_cleanup"]))) / 100.0
    detail = max(0.0, min(100.0, float(request["original_detail"]))) / 100.0

    started = time.monotonic()
    model, state, _ = init_df(model_base_dir=str(model_dir))
    target_rate = int(state.sr())
    audio, source_rate = torchaudio.load(str(source))
    if source_rate != target_rate:
        audio = torchaudio.functional.resample(audio, source_rate, target_rate)

    attenuation_limit = 3.0 + 15.0 * cleanup
    with torch.no_grad():
        cleaned = enhance(model, state, audio, atten_lim_db=attenuation_limit, pad=True)
    if cleaned.shape[-1] != audio.shape[-1]:
        length = min(cleaned.shape[-1], audio.shape[-1])
        cleaned = cleaned[..., :length]
        audio = audio[..., :length]

    # Cleanup strength and detail are independent macro controls. High detail
    # deliberately favors the dry microphone signal to protect consonants.
    wet = min(1.0, cleanup * (1.25 - 0.75 * detail))
    mixed = audio * (1.0 - wet) + cleaned * wet
    peak = float(mixed.abs().max().item()) if mixed.numel() else 0.0
    if peak > 1.0:
        mixed = mixed / peak
    output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output), mixed.cpu(), target_rate, encoding="PCM_F", bits_per_sample=32)
    json.dump(
        {
            "sample_rate": target_rate,
            "channels": int(mixed.shape[0]),
            "samples": int(mixed.shape[-1]),
            "wet": round(wet, 4),
            "attenuation_limit_db": round(attenuation_limit, 2),
            "elapsed_s": round(time.monotonic() - started, 3),
        },
        sys.stdout,
        separators=(",", ":"),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
