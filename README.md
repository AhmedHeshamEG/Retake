# RETAKE (Simple Edition)

Transcript-based rough-cut tool. 100% local. One Python file, one HTML file.

## Run

```
pip install -r requirements.txt
python retake.py
```

On Windows, `Start Retake.cmd` launches the bundled `.venv`. If you use a
virtualenv, install into that same interpreter -- a bare `pip install` lands in
whichever Python is on PATH, and Retake will then start without HTTPS or the MCP
endpoint. It prints the exact command to fix that on startup, naming the
interpreter it is actually running under.

The browser opens at `http://localhost:8710`. The command window also prints a
private LAN link to open on a phone. Choose or drop a video/audio file, select
its spoken language (or leave Auto-detect), and Retake uploads it with visible
resumable progress.

Core transcription and embedding weights load from `./models/`. Accurate word
alignment downloads its language model once into `models/alignment/` when that
optional runtime is first prepared; later calibration is local.

## Sending video from a phone

Retake serves two addresses: plain HTTP on port 8710, which the desktop has
always used, and HTTPS on port 8443. Both run at once, so nothing about the
desktop workflow changes, and **the plain HTTP address is the one to use from a
phone**. It needs no certificate and no warning screen.

A locked phone suspends the browser tab, and a suspended tab cannot upload. The
obvious fix, `navigator.wakeLock`, needs a secure context, and a LAN address
cannot have one without a certificate the phone has to be talked into trusting
-- a warning screen in exchange for a screen lock is a bad trade. So the page
does what a web audio player does instead: while a transfer is running it loops
one second of inaudible tone, which keeps the tab alive with the screen off. It
starts from the same tap that picks the file, and stops when the transfer ends.

HTTPS on 8443 is still served for anything that genuinely wants a secure
context. Its certificate is generated on first run, covers this computer's own
LAN addresses, and never leaves the machine, so a phone will warn once that it
is not trusted. Nothing about phone transfers requires it.

A transfer runs over four connections at once. Sending one chunk at a time left
the link idle for a full round trip after every chunk -- laptop writes the bytes
down, answers, and only then does the phone start sending again -- which is what
held phone uploads to a few hundred KB/s no matter how fast the Wi-Fi was. Each
chunk now carries the offset it belongs at, so the laptop writes it wherever it
lands and the four lanes never wait for one another. Chunk size adapts to the
link, up to 8 MB.

Transfers are resumable and no longer give up on their own. If the connection
drops or the tab is backgrounded, the page waits for it to come back and
continues from the last confirmed byte rather than failing. A resume starts at
the end of the unbroken run from byte zero, so a lane that finished ahead of a
lane that did not costs a few re-sent megabytes and never a corrupt file.

The one thing a browser cannot do by itself is re-open a file it was handed. If
a phone drops the file entirely -- which is the only failure left -- Retake says
how much already arrived, and tapping **Resume** and picking the same file
continues from that byte. Nothing already on the laptop is ever re-sent.

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

## Cut by instruction

"✦ Cut by instruction" accepts short guidance or detailed mixed Arabic/English
edit lists. Quoted phrases, keep commands, timestamps, and gaps resolve to exact
word/gap token IDs and stay highlighted for review; ambiguous or unmatched
instructions are shown instead of guessed. Plain-text reference scripts (`.txt`,
`.md`, `.srt`, and similar) are supported as context.

Nothing needs installing for this and no model runs locally. Retake resolves
what it can understand on its own; language it cannot parse is reported as
unresolved rather than guessed at. For conversational editing, connect an MCP
client -- see below.

## Editing with Claude, over MCP

`retake_mcp.py` exposes Retake to any MCP client. The division of labour is the
point: the client supplies the thing Retake cannot do, understanding what you
meant, and Retake supplies the thing a language model must not be trusted with,
deciding which real audio a phrase refers to. Ask for "the bit where I fumbled
the sponsor read" and Retake answers with exact word tokens and their
timestamps, or says it could not find them. A client can never name a raw time
span to delete or invent a token ID.

Every mutating tool previews first. `apply_cuts` reports the token IDs that
would change and the resulting duration, and writes nothing until you call it
again with `dry_run=false`. Applied edits push onto an undo stack of project
snapshots that `undo` unwinds one at a time, and edits are deltas, so cutting
five words never restores the rest of your work.

Start Retake, then point a client at it. Claude Code picks the server up from
the `.mcp.json` in this repository as soon as a session is started here -- it
launches `retake_mcp.py` from the project's own `.venv`, so a global
`pip install mcp` is never needed. Approve it once when Claude Code asks, and
`/mcp` will list `retake`.

For Claude Desktop, or to make the server available outside this directory, add
it to that client's own MCP server config:

```json
{
  "mcpServers": {
    "retake": {
      "command": "E:/path/to/Retake/.venv/Scripts/python.exe",
      "args": ["E:/path/to/Retake/retake_mcp.py"]
    }
  }
}
```

Clients that speak streamable HTTP can use `http://localhost:8710/mcp`
directly, which the running editor serves itself. `retake_mcp.py --http --port
8711` runs the same tools as a standalone HTTP server. Set `RETAKE_URL` if
Retake listens somewhere other than `http://127.0.0.1:8710`.

The 19 tools cover reading (`get_status`, `list_projects`, `open_project`,
`get_edit_summary`, `read_transcript`, `find_phrase`, `list_retakes`), editing
(`plan_edit`, `get_proposals`, `apply_cuts`, `apply_proposals`, `undo`,
`set_markers`, `set_voice_enhancement`), and jobs (`start_export`,
`get_job_status`, `export_text`, `get_latest_export`, `recalibrate_timing`).
`mcp` is an optional dependency: without it the editor runs unchanged and simply
has no `/mcp` endpoint.

## The retake-brain skill

`retake-brain.skill` is an Agent Skill for Claude that turns a raw recording
into a decisive cut-list: it reconstructs what the video is trying to say,
groups every re-attempt of each beat, and picks one winner per cluster ("last
complete attempt wins"), flagging only the calls that genuinely need ears --
mid-sentence splices, delivery choices, and facts the speaker contradicted.

With the MCP server connected it reads the transcript from the open project and
applies the list itself, through a dry run you approve. Without it, it works
from an uploaded SRT and prints the list for you to paste into "Cut by
instruction" -- the command grammar it emits is the one Retake's parser reads.

Install the `.skill` file into Claude. To change it, edit
`skills/retake-brain/SKILL.md` and rebuild:

```
python skills/build_skill.py
```

The test suite checks the skill's own worked example against the real parser, so
the two cannot drift apart.

Tests: `python test_retake.py`. Every project is self-contained under
`projects/Project N/`, including its original media, `project.json`, and
`exports/`. Reopening a project never retranscribes it.

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
- **Gap data**: Whisper-derived gap tokens remain compatible with old projects
  and AI instructions. Measured real-audio silences are stored separately in
  `audio_gaps`, each keeping the detector's own boundaries beside any manual
  adjustment, so filtering or re-detecting always judges against the evidence.

## Licence

MIT. See `LICENSE`.
