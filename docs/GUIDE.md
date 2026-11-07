# How Retake works

The parts of Retake that aren't visible from the editor, and the reasons behind them.

## How a cut is made

Deleted words next to each other become one cut, even across sentence boundaries. The cut then reaches outward into
the silence on both sides: it starts where the previous kept word ends and stops where the next kept word begins,
minus a small safety handle. So deleting a sentence also deletes the breath before it and the pause after it, instead
of leaving them hanging between the sentences you kept. A cut with no kept word on one side runs to the start or the
end of the recording. A kept word always splits a cut, and kept speech is never entered.

At export, each kept stretch has its edges snapped to real silence in the waveform where a safe join exists. The two
directions get different budgets because they carry different risks. Moving an edge inward, toward the speech, only
throws away audio measured as silence, so it may travel up to 2.0 s. Moving an edge outward brings back audio you cut,
so it's capped at 0.4 s: enough to recover the quiet start or tail of a word and nothing more. Neither changes the
saved timestamps, your selections, the text exports or the preview. If there's no safe silence nearby, the
speech-safe boundary is used as it is.

## Controls

Click a word to cut it or bring it back. Alt+click or double-click a word to jump the player there. Press `?` for the
full shortcut list, and `M` during playback to drop a marker.

## Sending video from a phone

Retake serves plain HTTP on port 8710 and HTTPS on port 8443, both at once. Phones should use the HTTPS one: iOS
Safari only allows a wake lock over HTTPS, and without it the screen sleeps halfway through a large upload, Safari
suspends the tab, and the transfer stalls.

The certificate is generated on first run, covers this computer's LAN addresses, and never leaves the machine. The
phone warns once that it isn't trusted; choose Advanced and continue. Retake prints the address to open.

Uploads are resumable and don't give up on their own. If the connection drops or the tab goes to the background, the
page waits and continues from the last byte the laptop confirmed. Chunk size adapts to the link, up to 8 MB. Reopening
Retake after an interruption asks the laptop how much arrived and offers to pick up from there.

## Docker

```bash
docker compose up --build
```

`models/` and `projects/` are mounted into the container, so the weights stay out of the image and projects survive a
rebuild. The default container runs on the CPU. For an NVIDIA GPU (needs the NVIDIA Container Toolkit):

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

If an accelerated job fails after startup, the backend retries it on the CPU.

## Export

**Reliable Quality** is the default. It writes an H.264/AAC MP4 with continuous frame timing, using NVIDIA hardware
encoding when it can. Before the file is offered, Retake checks its duration, dimensions, frame rate, audio
properties and the timing of every video packet. If hardware encoding is missing or fails, it retries on a CPU
encoder. **Compatibility** uses the same timeline at a smaller file size.

**Download Latest Export** hands you the newest finished video for the open project, even after a restart. Text
exports (EDL, CSV, TXT, SRT) are separate.

## Preview

The edited preview seeks across cuts without pausing. For heavy 4K phone footage, **Smooth Preview** builds a 720p
proxy once, using NVIDIA decode, scale and encode where it can. The proxy stays valid as you keep editing, playback
falls back to the original if the proxy fails, and the export never touches it.

## Voice enhancement

On by default, with the conservative **Great** profile: speech is brought to a comfortable −16 LUFS, quiet and loud
moments are levelled gently, true peaks are protected, and the sample rate and channels stay as they were. **Fine
tune** exposes loudness, levelling, noise cleanup and original detail as percentages. Changing cleanup strength never
changes the loudness you picked.

Clean recordings skip neural denoising. When there's real background noise and the optional DeepFilterNet3 runtime is
installed, Retake applies limited cleanup before levelling and a two-pass loudness normalisation. If the model is
missing or fails, the export goes ahead with levelling and loudness only, and tells you.

## Optional runtimes

Both live in their own Python 3.11 environments under the git-ignored `models/` folder, so they never touch the main
install.

### Precise word timing (WhisperX)

Whisper's word timestamps can drift or end before the word you can hear. **Recalibrate Timing** re-aligns every word
against the audio with WhisperX phoneme alignment, then protects the kept side of every cut. It runs once per project
and is cached. Word IDs, text, cuts, markers and media are kept, and a timestamped backup of the project is written
first. It tries CUDA first and, if that's missing, out of memory or fails, retries once on the CPU without
half-updating the project.

```bat
py -3.11 -m venv models\alignment-runtime
models\alignment-runtime\Scripts\python.exe -m pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
models\alignment-runtime\Scripts\python.exe -m pip install -r requirements-alignment.txt
```

The alignment model for each language downloads once into `models/alignment/` the first time it's needed.

### Noise cleanup (DeepFilterNet3)

```bat
py -3.11 -m venv models\enhancement-runtime
models\enhancement-runtime\Scripts\python.exe -m pip install torch torchaudio
models\enhancement-runtime\Scripts\python.exe -m pip install -r requirements-enhancement.txt
```

Then put the official DeepFilterNet3 model folder at `models/deepfilternet/`.

## Notes

- If `ffprobe` isn't on `PATH` or next to ffmpeg (`imageio-ffmpeg` only ships ffmpeg), probing reads `ffmpeg -i`
  output instead.
- Old projects can carry gap records from a manual gap editor that has since been removed. They're kept for
  compatibility and ignored, so an invisible cut can never reach the preview or the export.
- If you use a virtual environment, install into the same interpreter that runs Retake. Otherwise it starts without
  HTTPS or the MCP endpoint, and prints the exact command to fix it.
