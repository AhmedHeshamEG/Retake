# How Retake works

The parts of Retake that aren't visible from the editor, and the reasons behind them.

## How a cut is made

Deleted words next to each other become one cut, even across sentence boundaries. The cut then reaches outward into
the silence on both sides: it starts where the previous kept word ends and stops where the next kept word begins,
minus a small safety handle. So deleting a sentence also deletes the breath before it and the pause after it, instead
of leaving them hanging between the sentences you kept. A cut with no kept word on one side runs to the start or the
end of the recording. A kept word always splits a cut, and kept speech is never entered.

Word timestamps are estimates, so they never decide where a cut lands. They only decide which pause it belongs in:
between this kept word and that deleted one. Inside that pause, `splice.py` reads the waveform the way you would after
zooming in:

- a kept stretch that ends stops a short handle after its last word has died away;
- a kept stretch that starts begins a short handle before its first word, at the end of the last quiet stretch, so a
  breath is never cut in half;
- in a very short dip the join sits in the middle of it, away from both words;
- when two words were said in one breath and there is no quiet at all, the join goes to the quietest instant between
  them.

Pause trims get the same treatment, walled in by the words either side of the silence. A pause too short to trim is
left alone, and a pause whose edges you dragged by hand is used exactly as drawn. A word whose timing the aligner
couldn't confirm is trusted as a wall but searched past, by up to 150 ms, when its pause holds no quiet.

Preview, media export and the text exports (EDL, SRT, CSV) all play this one plan, so what you hear while editing is
what you get. Every join fades out and back in over 6 ms, which is inaudible in silence and removes the click from a
join between words said in one breath.

The plan checks itself. Every join is measured, and one that still sits on sound is marked amber on the strip under
the player; **Check next join** plays from just before it. On the recordings this was tuned against, about 4% of joins
are like that, and all of them are words run together with no pause to find.

## How a recording is transcribed

Whisper isn't handed the whole file. An improvised recording is mostly retakes, and a window holding the same line
twice gets written down once: the other take gets no words at all, and Whisper hides the hole by stretching one word
of the take it kept across the one it dropped. So `transcription.py` does three things instead:

1. **Measures the recording.** Every stretch above the recording's own noise floor is an island of sound. Islands are
   packed into windows that end on real pauses, and pauses of two seconds or more are left out, which stops words
   being invented for dead air.
2. **Decodes each window on its own**, with the same `large-v3` model. With the language on Auto-detect, several
   windows spread across the recording are asked, not just the first thirty seconds, so an accented opening can't
   get the whole thing written out in another language.
3. **Checks the words against the measurement.** Audible sound with no word over it, or with one word lasting far
   longer than it could take to say, is decoded again on its own, where there's no neighbouring take to confuse it
   with. Whatever it finds is added in place.

Whatever is still wordless after that isn't speech. It shows in the transcript as a **sound** chip (`♪ 0.31s`): a
breath, a click, a hum. Removing gaps leaves sounds alone; click a chip to cut that one, or use **Cut sounds** in the
toolbar for all of them. Both are ordinary cuts, so undo, preview and export treat them like any other.

`retake.log` records every stretch that was re-heard and what came of it. Projects transcribed before this keep their
old transcript; open the recording again as a new project to get a new one.

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

The edited preview seeks across cuts without pausing. For audio it checks for a cut every animation frame and one
frame ahead, so a deleted word is skipped before it's heard. An iPhone voice memo saved as lossless is Apple Lossless
(ALAC) in an `.m4a`, which only Safari plays, so Retake makes a FLAC copy once, in the background when the project
opens, and plays that instead; FLAC has no encoder delay, so preview stays sample-aligned with the transcript. For heavy 4K phone footage, **Smooth Preview** builds a 720p
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

Alignment runs in two passes. About one word in twenty comes back from the first without a confident time, and those
words cluster in retakes and false starts, which is exactly where cuts go. The second pass re-aligns each run of them
on its own, inside the gap between its aligned neighbours, and keeps a time only if it falls in order inside that gap.
On real recordings it places roughly half of the missed words.

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
