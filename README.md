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

## Voice enhancement and export quality

`Voice Enhancement` is enabled by default with the conservative **Great**
profile. It targets comfortable -16 LUFS speech, gently levels quiet/loud
moments, protects true peaks, and preserves the source sample rate and channels.
`Fine tune` exposes friendly percentages for output loudness, voice leveling,
noise cleanup, and original detail. Cleanup strength never changes the selected
final loudness.

Clean recordings bypass neural denoising. When meaningful background noise is
detected and the optional local DeepFilterNet3 runtime is ready, Retake applies
limited cleanup before leveling and measured two-pass loudness normalization.
If the model is absent or fails, export safely continues with the untouched
voice plus leveling/loudness and shows a warning.

On Windows, prepare the isolated optional runtime with Python 3.11 and place the
official DeepFilterNet3 model directory at `models/deepfilternet/`:

```
py -3.11 -m venv models\enhancement-runtime
models\enhancement-runtime\Scripts\python.exe -m pip install torch torchaudio
models\enhancement-runtime\Scripts\python.exe -m pip install -r requirements-enhancement.txt
```

The runtime and checkpoints remain under the Git-ignored `models/` folder.

During media export, Retake first builds the existing authoritative consecutive
cut/keep intervals. It then inspects only each kept interval's edge and snaps a
nearby trustworthy join to real waveform silence. The two directions carry
different budgets because they carry different risk: moving an edge inward,
toward the speech the interval exists for, only discards audio measured as
silence and may travel up to 2.0 s, while moving an edge outward restores
excluded audio and stays capped at 0.4 s so it can recover a quiet word
attack/release and nothing more. Neither changes saved word timestamps, edit
selections, text exports, or preview. If no safe nearby silence exists, the
established speech-safe boundary is used unchanged.

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

Consecutive deleted words are composed as one continuous backend cut, even across
sentence boundaries, and that cut extends outward into the non-speech on either
side: it begins where the previous kept word ended and stops where the next kept
word begins, each held back by a small safety handle. Deleting a sentence
therefore also removes the breath before it and the pause after it, instead of
leaving them audible between the sentences you kept. A run with no kept word
before or after it reaches the start or the end of the recording. A kept spoken
word always splits the cut, and kept speech is never entered.

## Deviations

- **Word click vs. seek**: the spec binds plain click to both "toggle strike" and "seek there". Plain click toggles the cut (the core editing loop); **Alt+click or double-click seeks**. Listed in the `?` shortcut overlay.
- **ffprobe fallback**: `imageio-ffmpeg` ships only ffmpeg. If ffprobe isn't on PATH (or next to ffmpeg), probing falls back to parsing `ffmpeg -i` output instead of failing at startup — strictly more robust, same results.
- **llama-cpp-python** is commented out in `requirements.txt` (it may need a compiler to build). This keeps `pip install -r requirements.txt` failure-proof; the AI feature documents its own one-line install above.
- **Legacy gap data**: Whisper-derived gap tokens remain compatible with old
  projects and AI instructions. The removed manual real-audio gap editor's saved
  records are left untouched for compatibility but are intentionally inert, so
  an invisible cut can never affect preview or export.
