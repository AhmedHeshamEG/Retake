# RETAKE (Simple Edition)

Transcript-based rough-cut tool. 100% local. One Python file, one HTML file.

## Run

```
pip install -r requirements.txt
python retake.py
```

The browser opens at `http://localhost:8710`. The command window also prints the private LAN link to open on a phone. Choose or drop a video/audio file, select its spoken language (or leave Auto-detect), and Retake uploads it with visible resumable progress.

Core transcription, embedding, and optional GGUF weights load from `./models/`.
Accurate word alignment downloads its language model once into `models/alignment/`
when that optional runtime is first prepared; later calibration is local.

## Docker

Docker is optional. With Docker Desktop or Docker Engine installed, start Retake with:

```
docker compose up --build
```

Then open `http://localhost:8710`. The local `models/` and `projects/` folders
are mounted into the container, so model weights stay outside the image and
projects survive container replacement. A phone or tablet on the same private
network can use `http://COMPUTER-LAN-IP:8710` when the host firewall allows it.

The standard container uses the CPU fallback. To grant an NVIDIA GPU to the
same image (NVIDIA Container Toolkit required), use:

```
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

The backend still retries on CPU if an accelerated job fails after startup.

## Accurate word timing

Whisper's normal word timestamps can drift or end before the audible word.
`Recalibrate Timing` uses WhisperX phoneme alignment against the original audio,
then protects the retained side of every transcript cut. It runs once per
project and is cached; normal preview and export stay fast. Existing word IDs,
text, cut selections, markers, and source media are preserved, and a timestamped
project backup is written before publication.

On Windows, prepare the isolated GPU-first runtime with Python 3.11:

```
py -3.11 -m venv models\alignment-runtime
models\alignment-runtime\Scripts\python.exe -m pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
models\alignment-runtime\Scripts\python.exe -m pip install -r requirements-alignment.txt
```

The CUDA download is large. The runtime and alignment models remain under the
Git-ignored `models/` folder. Retake always tries CUDA first; if CUDA is absent,
out of memory, or errors, the isolated worker exits and Retake retries once on
CPU without partially updating the project.

## Optional AI cut

Drop any instruct `.gguf` (recommended: Qwen3-4B-Instruct Q4_K_M) into `models/llm/` and install `pip install llama-cpp-python==0.3.33`. "✦ Cut with AI" accepts short guidance or detailed mixed Arabic/English edit lists. Quoted phrases, keep commands, timestamps, and gaps resolve to exact word/gap token IDs and stay highlighted for review; ambiguous or unmatched instructions are shown instead of guessed. Plain-text reference scripts (`.txt`, `.md`, `.srt`, and similar) remain supported. No GGUF → the AI card explains what is missing; everything else works.

Tests: `python test_retake.py`. Every project is self-contained under `projects/Project N/`, including its original media, `project.json`, and `exports/`. Reopening a project never retranscribes it.

## Real-audio gaps and export quality

`Show gaps` is off by default. Turning it on analyzes the actual audio with FFmpeg instead of trusting Whisper's padded word timestamps. Detected gaps are a separate editable layer with a compact waveform, draggable edges, exact timestamp fields, configurable sensitivity/minimum duration, and a natural-pause setting. Bulk removal is reviewable and undoable; it never rewrites words or cuts automatically.

When gaps are visible, `Remove gaps in kept sentences` offers a separate scoped batch: sentences with at least one kept word remain eligible, while fully deleted sentences and gaps already covered by consecutive deleted content are skipped. The original `Remove all gaps` action remains available and unchanged.

`Reliable Quality` is the default media export. It creates a high-quality
H.264/AAC MP4 with continuous frame timing and uses NVIDIA hardware encoding
when available. Retake verifies duration, displayed dimensions, FPS, audio
properties, and every video packet's timing before making the export available.
If hardware encoding is unavailable or fails, Retake retries with a reliable
CPU encoder. `Compatibility` uses the same stable timeline with a smaller-file
quality setting.

`Download Latest Export` transfers the newest completed media export for the
open project directly through the browser. It works after restarting Retake and
shows a prompt to export first when no media export exists. Text exports such
as EDL, CSV, TXT, and SRT are not selected.

Edited preview uses native range-aware file delivery and seeks across cuts
without pausing first. For high-bitrate 4K phone media, `Smooth Preview` can
prepare a reusable 720p H.264/AAC proxy with NVIDIA decoding, scaling, and
encoding, with a speed-oriented CPU fallback. This is an explicit one-time job,
remains valid when edits change, and
automatically falls back to the original if proxy playback fails. The source and
final export never use or depend on this disposable proxy. Cut jumps retry
silently when decoding is briefly delayed. Safe local reads/autosaves retry
transient connection failures; job-starting actions such as AI, preview
preparation, and export are never duplicated automatically.

Consecutive deleted words are composed as one continuous backend cut from the first deleted word's start to the last deleted word's end, even across sentence boundaries. This removes breaths, noise, and unreported timestamp holes inside deleted passages while a kept spoken word always splits the cut.

## Deviations

- **Word click vs. seek**: the spec binds plain click to both "toggle strike" and "seek there". Plain click toggles the cut (the core editing loop); **Alt+click or double-click seeks**. Listed in the `?` shortcut overlay.
- **ffprobe fallback**: `imageio-ffmpeg` ships only ffmpeg. If ffprobe isn't on PATH (or next to ffmpeg), probing falls back to parsing `ffmpeg -i` output instead of failing at startup — strictly more robust, same results.
- **llama-cpp-python** is commented out in `requirements.txt` (it may need a compiler to build). This keeps `pip install -r requirements.txt` failure-proof; the AI feature documents its own one-line install above.
- **Whisper gaps vs. real-audio gaps**: legacy Whisper-derived gap tokens remain compatible with old projects and AI instructions. The finishing-stage `Show gaps` layer is detected independently from the audio waveform, can overlap inaccurate word timestamps safely, and is stored under stable string IDs.
