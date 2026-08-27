"""
RETAKE (Simple Edition) — single-file backend.

Run:  python retake.py   →  http://localhost:8710 opens in the browser.

Everything lives here: HTTP server, transcription, retake clustering,
cut-interval math, optional local-LLM cut proposals, and export.
See context.md for the binding spec.
"""
from __future__ import annotations

import asyncio
import copy
import csv
import difflib
import hashlib
import importlib.util
import ipaddress
import itertools
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from array import array
from fractions import Fraction
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import uvicorn
from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response

# The application ships with its model weights. Never contact a model hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# --------------------------------------------------------------------------
# Paths, logging
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
PROJECTS_DIR = ROOT / "projects"
INCOMING_DIR = PROJECTS_DIR / ".incoming"
TLS_DIR = MODELS_DIR / "tls"
INDEX_HTML = ROOT / "index.html"

for d in (MODELS_DIR, PROJECTS_DIR, INCOMING_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "retake.log", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),
    ],
)
log = logging.getLogger("retake")

PORT = 8710
HTTPS_PORT = 8443
TLS_CERT_DAYS = 397
TLS_CERT_VERSION = 1
GAP_THRESHOLD_DEFAULT = 0.35
ACTIVE_GAP_MIN_DEFAULT = 0.35
ACTIVE_GAP_DB_DEFAULT = -40.0
ACTIVE_GAP_KEEP_DEFAULT = 0.12
MERGE_EPS = 0.001
AI_ATTACHMENT_MAX_FILES = 5
AI_ATTACHMENT_MAX_BYTES = 512 * 1024
AI_ATTACHMENT_MAX_CHARS = 12_000
AI_INSTRUCTION_MAX_CHARS = 50_000
ASSISTANT_MAX_OPERATIONS = 400
ASSISTANT_MAX_TOKEN_IDS = 20_000
ASSISTANT_TRANSCRIPT_PAGE = 200
ASSISTANT_BACKUP_KEEP = 20
MCP_HTTP_PATH = "/mcp"
UPLOAD_CHUNK_MAX_BYTES = 8 * 1024 * 1024
RETAKE_MAX_SPAN_S = 90.0
RETAKE_MAX_MEMBERS = 8
PREVIEW_PROXY_VERSION = 2
ALIGNMENT_VERSION = 1
ALIGNMENT_MIN_COVERAGE = 0.85
ALIGNMENT_MIN_SCORE = 0.45
ALIGNMENT_MAX_WORD_SECONDS = 2.5
WORD_CUT_SAFETY_S = 0.06
# Unaligned Whisper word boundaries drift more than forced-aligned ones, so a
# cut that absorbs the pause around deleted speech keeps a wider handle there.
WORD_CUT_SAFETY_UNALIGNED_S = 0.12
VOICE_ENHANCEMENT_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "profile": "great",
    "output_loudness": 80.0,
    "voice_leveling": 25.0,
    "noise_cleanup": 25.0,
    "original_detail": 90.0,
}
EXPORT_SILENCE_MIN_S = 0.05
EXPORT_SILENCE_DB_DEFAULT = -45.0
# Export silence snapping is asymmetric. Moving a kept edge inward (toward
# speech) only ever discards material ffmpeg measured as silence, so it is
# allowed a generous budget. Moving an edge outward restores audio that cut
# composition excluded, so it stays tightly capped.
EXPORT_EDGE_MAX_SHIFT_S = 0.40
EXPORT_EDGE_TRIM_MAX_S = 2.00
EXPORT_TRUE_PEAK_DB = -1.5

# --------------------------------------------------------------------------
# ffmpeg / ffprobe resolution
# --------------------------------------------------------------------------

FFMPEG: str = ""
FFPROBE: Optional[str] = None


def resolve_ffmpeg() -> None:
    """Locate ffmpeg (imageio-ffmpeg first, then PATH) and ffprobe (PATH,
    then next to ffmpeg). Exits with a clear message if ffmpeg is absent."""
    global FFMPEG, FFPROBE
    try:
        import imageio_ffmpeg  # type: ignore

        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG = shutil.which("ffmpeg") or ""
    if not FFMPEG:
        FFMPEG = shutil.which("ffmpeg") or ""
    if not FFMPEG:
        print(
            "FATAL: ffmpeg not found. Install the pinned requirements "
            "(imageio-ffmpeg ships a binary) or put ffmpeg on your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    FFPROBE = shutil.which("ffprobe")
    if not FFPROBE:
        sibling = Path(FFMPEG).with_name(
            "ffprobe.exe" if os.name == "nt" else "ffprobe"
        )
        if sibling.exists():
            FFPROBE = str(sibling)
    log.info("ffmpeg: %s | ffprobe: %s", FFMPEG, FFPROBE or "(fallback via ffmpeg -i)")


def _run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
    """Run a subprocess with captured output; never raises on non-zero exit."""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kw
    )


def probe_media(path: str) -> dict[str, Any]:
    """Return media properties needed by preview and export verification.

    Prefers ffprobe JSON; falls back to parsing `ffmpeg -i` stderr when
    ffprobe is unavailable (imageio-ffmpeg ships only ffmpeg).
    """
    container = Path(path).suffix.lstrip(".").lower()
    if FFPROBE:
        p = _run(
            [
                FFPROBE, "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", path,
            ]
        )
        if p.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {p.stderr.strip()[:300]}")
        data = json.loads(p.stdout)
        duration = float(data.get("format", {}).get("duration") or 0.0)
        vcodec = acodec = None
        width = height = sample_rate = channels = None
        rotation = 0
        video_streams = audio_streams = 0
        fps = 25.0
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                # skip attached cover art
                if s.get("disposition", {}).get("attached_pic"):
                    continue
                video_streams += 1
                if vcodec is None:
                    vcodec = s.get("codec_name")
                    width = s.get("width")
                    height = s.get("height")
                    try:
                        rotation = int(s.get("tags", {}).get("rotate") or 0)
                        for side_data in s.get("side_data_list", []):
                            if side_data.get("rotation") is not None:
                                rotation = int(side_data["rotation"])
                                break
                    except (TypeError, ValueError):
                        rotation = 0
                    rate = s.get("r_frame_rate") or "25/1"
                    try:
                        fps = float(Fraction(rate)) or 25.0
                    except (ValueError, ZeroDivisionError):
                        fps = 25.0
            elif s.get("codec_type") == "audio":
                audio_streams += 1
                if acodec is None:
                    acodec = s.get("codec_name")
                    try:
                        sample_rate = int(s.get("sample_rate")) if s.get("sample_rate") else None
                    except (TypeError, ValueError):
                        sample_rate = None
                    channels = s.get("channels")
            if duration <= 0 and s.get("duration"):
                duration = max(duration, float(s["duration"]))
        return {
            "duration_s": duration, "container": container, "vcodec": vcodec,
            "acodec": acodec, "has_video": vcodec is not None, "fps": fps,
            "width": width, "height": height, "sample_rate": sample_rate,
            "channels": channels, "rotation": rotation,
            "video_streams": video_streams,
            "audio_streams": audio_streams,
        }

    # Fallback: parse `ffmpeg -i` stderr.
    p = _run([FFMPEG, "-hide_banner", "-i", path])
    err = p.stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", err)
    if not m:
        raise RuntimeError(f"could not probe media: {err.strip()[:300]}")
    h, mi, s, cs = (int(g) for g in m.groups())
    duration = h * 3600 + mi * 60 + s + cs / 100.0
    vm = re.search(r"Stream .*?: Video: (\w+)", err)
    am = re.search(r"Stream .*?: Audio: (\w+)", err)
    fm = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
    return {
        "duration_s": duration, "container": container,
        "vcodec": vm.group(1) if vm else None,
        "acodec": am.group(1) if am else None,
        "has_video": vm is not None,
        "fps": float(fm.group(1)) if fm else 25.0,
        "width": None, "height": None, "sample_rate": None, "channels": None,
        "rotation": 0,
        "video_streams": 1 if vm else 0, "audio_streams": 1 if am else 0,
    }


# --------------------------------------------------------------------------
# Status + app state (single user, single project)
# --------------------------------------------------------------------------

_STATE_LOCK = threading.Lock()
JOB_LOCK = threading.Lock()  # one heavy job at a time
PROJECT_ALLOC_LOCK = threading.Lock()

STATUS: dict[str, Any] = {
    # idle|probe|load_model|transcribe|cluster|ready|ai|align|export|preview|error
    "phase": "idle",
    "pct": 0.0,
    "msg": "",
    "error": None,
    "err_seq": 0,  # bumps on every new error so the UI toasts it exactly once
}
CURRENT: dict[str, Any] = {
    "project": None, "media_path": None, "project_dir": None, "source_filename": None,
}
AI_STATE: dict[str, Any] = {
    "running": False, "proposals": [], "warnings": [], "done": False, "mode": None,
}
LAST_EXPORT: dict[str, Any] = {"path": None, "warning": None, "enhancement": None}
PREVIEW_STATE: dict[str, Any] = {
    "running": False, "identity": None, "error": None,
    "device": None, "fallback": False,
}
ALIGNMENT_STATE: dict[str, Any] = {
    "running": False, "identity": None, "error": None,
    "device": None, "fallback": False, "coverage": None,
}
_PREVIEW_GPU_USABLE: Optional[bool] = None


def set_status(phase: str, pct: float = 0.0, msg: str = "", error: Optional[str] = None) -> None:
    """Update the global status atomically; bumps err_seq when a new error is set."""
    with _STATE_LOCK:
        STATUS["phase"] = phase
        STATUS["pct"] = round(float(pct), 1)
        STATUS["msg"] = msg
        if error is not None:
            STATUS["error"] = error
            STATUS["err_seq"] += 1


def fail(msg: str, back_to: str = "ready") -> None:
    """Log an error, surface it via /status, and return to a usable phase."""
    log.exception(msg)
    phase = back_to if CURRENT["project"] else "idle"
    set_status(phase, 0, "", error=msg)


# --------------------------------------------------------------------------
# Project persistence
# --------------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def project_directories() -> list[Path]:
    rows: list[tuple[int, Path]] = []
    for path in PROJECTS_DIR.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"Project (\d+)", path.name)
        if match:
            rows.append((int(match.group(1)), path))
    return [path for _, path in sorted(rows)]


def next_project_directory() -> Path:
    with PROJECT_ALLOC_LOCK:
        numbers = [int(path.name.split()[-1]) for path in project_directories()]
        path = PROJECTS_DIR / f"Project {max(numbers, default=0) + 1}"
        path.mkdir()
        (path / "media").mkdir()
        (path / "exports").mkdir()
        return path


def project_directory_for_media(media_path: str) -> Optional[Path]:
    media = Path(media_path).resolve()
    for project_dir in project_directories():
        try:
            media.relative_to((project_dir / "media").resolve())
            return project_dir
        except ValueError:
            continue
    return None


def project_path_for(media_path: str) -> Path:
    project_dir = project_directory_for_media(media_path)
    if project_dir is None:
        raise ValueError("media is not inside a Retake project folder")
    return project_dir / "project.json"


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON via tmp file + os.replace so a crash never corrupts a project."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


def save_current_project() -> None:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return
        proj["updated_at"] = utc_now()
        project_dir = CURRENT.get("project_dir") or project_directory_for_media(proj["source_path"])
        if project_dir is None:
            raise RuntimeError("current project folder is unavailable")
        atomic_write_json(Path(project_dir) / "project.json", proj)


def validate_voice_enhancement(raw: Any) -> dict[str, Any]:
    """Return one strict, stable export-enhancement settings object."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("voice enhancement settings must be an object")
    enabled = raw.get("enabled", VOICE_ENHANCEMENT_DEFAULTS["enabled"])
    if not isinstance(enabled, bool):
        raise ValueError("voice enhancement enabled must be true or false")
    profile = str(raw.get("profile", VOICE_ENHANCEMENT_DEFAULTS["profile"]))
    if profile not in {"great", "custom"}:
        raise ValueError("voice enhancement profile must be great or custom")

    settings: dict[str, Any] = {"enabled": enabled, "profile": profile}
    for key in (
        "output_loudness", "voice_leveling", "noise_cleanup", "original_detail",
    ):
        value = raw.get(key, VOICE_ENHANCEMENT_DEFAULTS[key])
        if isinstance(value, bool):
            raise ValueError(f"{key.replace('_', ' ')} must be a percentage")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key.replace('_', ' ')} must be a percentage") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 100.0:
            raise ValueError(f"{key.replace('_', ' ')} must be between 0 and 100")
        settings[key] = round(value, 1)
    return settings


def ensure_voice_enhancement_state(proj: dict[str, Any]) -> bool:
    """Lazily add validated enhancement defaults without touching edit data."""
    existing = proj.get("voice_enhancement")
    try:
        validated = validate_voice_enhancement(existing)
    except ValueError:
        validated = dict(VOICE_ENHANCEMENT_DEFAULTS)
    if existing != validated:
        proj["voice_enhancement"] = validated
        return True
    return False


def ensure_audio_gap_state(proj: dict[str, Any]) -> bool:
    """Lazily add the export-only real-audio gap layer to old projects."""
    changed = False
    defaults = {
        "min_duration_s": ACTIVE_GAP_MIN_DEFAULT,
        "silence_db": ACTIVE_GAP_DB_DEFAULT,
        "keep_pause_s": ACTIVE_GAP_KEEP_DEFAULT,
    }
    settings = proj.get("audio_gap_settings")
    if not isinstance(settings, dict):
        proj["audio_gap_settings"] = dict(defaults)
        changed = True
    else:
        for key, value in defaults.items():
            if key not in settings:
                settings[key] = value
                changed = True
    if not isinstance(proj.get("audio_gaps"), list):
        proj["audio_gaps"] = []
        changed = True
    if "audio_gaps_analyzed" not in proj:
        proj["audio_gaps_analyzed"] = False
        changed = True
    return changed


def loudness_target_lufs(percent: float) -> float:
    """Map the friendly 0-100 control onto the -24 to -14 LUFS range."""
    bounded = max(0.0, min(100.0, float(percent)))
    return round(-24.0 + bounded * 0.10, 1)


# --------------------------------------------------------------------------
# Cut-interval math (pure, unit-tested in test_retake.py)
# --------------------------------------------------------------------------

def merge_intervals(
    intervals: list[tuple[float, float]], eps: float = MERGE_EPS
) -> list[tuple[float, float]]:
    """Sort intervals by start and merge overlapping/touching ones.

    Two intervals merge when next.start <= cur.end + eps. Empty/negative
    intervals are dropped. This single function feeds preview and export.
    """
    cleaned = [(float(a), float(b)) for a, b in intervals if b > a]
    if not cleaned:
        return []
    cleaned.sort(key=lambda iv: iv[0])
    out: list[tuple[float, float]] = [cleaned[0]]
    for start, end in cleaned[1:]:
        cur_start, cur_end = out[-1]
        if start <= cur_end + eps:
            out[-1] = (cur_start, max(cur_end, end))
        else:
            out.append((start, end))
    return out


def keep_list(
    cut_intervals: list[tuple[float, float]], duration: float, eps: float = MERGE_EPS
) -> list[tuple[float, float]]:
    """Complement of merged cut intervals over [0, duration]."""
    merged = merge_intervals(cut_intervals, eps)
    keeps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if start - cursor > eps:
            keeps.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor > eps:
        keeps.append((cursor, duration))
    return keeps


def _timed_interval(item: dict[str, Any]) -> Optional[tuple[float, float]]:
    """Return one finite, positive interval without trusting saved project data."""
    try:
        start, end = float(item["start"]), float(item["end"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return start, end


def consecutive_word_cut_intervals(
    tokens: list[dict[str, Any]],
    safety_handle: float = 0.0,
    absorb_pauses: bool = False,
    duration: Optional[float] = None,
) -> list[tuple[float, float]]:
    """Collapse every consecutive run of cut spoken words into one interval.

    Only a kept spoken word ends a run. Sentence boundaries and represented or
    unrepresented gaps are deliberately ignored, so breaths/noise between two
    neighboring deleted words cannot leak into preview or export.

    With ``absorb_pauses`` the run also claims the non-speech on both of its
    outer sides: it begins where the previous kept word ended and stops where
    the next kept word begins, each held back by ``safety_handle``. Whisper
    reports a deleted sentence as first-word-start .. last-word-end, so without
    this the breath before it and the pause after it belong to no cut and
    survive the edit. Runs at the head or tail of the recording extend to 0.0
    and ``duration`` respectively. Kept speech is never entered: the handle is
    applied against the neighboring kept word, not the deleted one.
    """
    words: list[tuple[float, float, str, bool]] = []
    for token in tokens:
        if token.get("kind") != "word":
            continue
        interval = _timed_interval(token)
        if interval is None:
            continue
        start, end = interval
        words.append((start, end, str(token.get("id", "")), bool(token.get("cut"))))
    words.sort(key=lambda word: (word[0], word[1], word[2]))

    intervals: list[tuple[float, float]] = []
    run: Optional[tuple[float, float]] = None
    previous_kept_end: Optional[float] = None
    for start, end, _token_id, is_cut in words:
        if is_cut:
            if run is not None:
                run = (run[0], max(run[1], end))
            else:
                protected_start = start
                if absorb_pauses:
                    protected_start = (
                        max(0.0, previous_kept_end + safety_handle)
                        if previous_kept_end is not None
                        else 0.0
                    )
                elif safety_handle > 0 and previous_kept_end is not None:
                    protected_start = max(
                        protected_start, previous_kept_end + safety_handle
                    )
                run = (protected_start, max(end, protected_start))
        elif run is not None:
            protected_end = start - safety_handle if absorb_pauses else run[1]
            if safety_handle > 0:
                protected_end = min(protected_end, start - safety_handle)
            if protected_end > run[0] + MERGE_EPS:
                intervals.append((run[0], protected_end))
            run = None
            previous_kept_end = end
        else:
            previous_kept_end = end
    if run is not None:
        run_end = run[1]
        if absorb_pauses and duration is not None and math.isfinite(duration):
            run_end = max(run_end, float(duration))
        if run_end > run[0] + MERGE_EPS:
            intervals.append((run[0], run_end))
    return intervals


def transcript_cut_intervals(
    tokens: list[dict[str, Any]],
    safety_handle: float = 0.0,
    absorb_pauses: bool = False,
    duration: Optional[float] = None,
) -> list[tuple[float, float]]:
    """Continuous spoken-word runs plus explicitly cut legacy gap tokens."""
    legacy_gap_cuts = [
        interval
        for token in tokens
        if token.get("kind") != "word" and token.get("cut")
        if (interval := _timed_interval(token)) is not None
    ]
    return merge_intervals(
        consecutive_word_cut_intervals(
            tokens, safety_handle, absorb_pauses=absorb_pauses, duration=duration
        )
        + legacy_gap_cuts
    )


def project_uses_aligned_timing(proj: dict[str, Any]) -> bool:
    alignment = proj.get("alignment")
    return (
        isinstance(alignment, dict)
        and alignment.get("status") == "aligned"
        and alignment.get("version") == ALIGNMENT_VERSION
        and float(alignment.get("coverage") or 0.0) >= ALIGNMENT_MIN_COVERAGE
    )


def cut_composition_params(proj: dict[str, Any]) -> dict[str, Any]:
    """The one place preview, saving, and export agree on how cuts compose.

    Absorbing the pause around a deleted run is unconditional, so the handle can
    no longer be zero: it is the only thing standing between a cut boundary and
    the neighboring kept word.
    """
    aligned = project_uses_aligned_timing(proj)
    try:
        duration = float(proj.get("duration_s") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return {
        "safety_handle": WORD_CUT_SAFETY_S if aligned else WORD_CUT_SAFETY_UNALIGNED_S,
        "absorb_pauses": True,
        "duration": duration if math.isfinite(duration) and duration > 0 else None,
    }


def cut_intervals_from_tokens(proj: dict[str, Any]) -> list[tuple[float, float]]:
    """All accepted edits, with consecutive cut words composed continuously."""
    transcript_cuts = transcript_cut_intervals(proj.get("tokens", []), **cut_composition_params(proj))
    gap_cuts = [
        interval
        for gap in proj.get("audio_gaps", [])
        if gap.get("cut")
        if (interval := _timed_interval(gap)) is not None
    ]
    return merge_intervals(transcript_cuts + gap_cuts)


def scoped_gap_candidate_ids(proj: dict[str, Any]) -> list[str]:
    """Return gaps near kept speech, without changing transcript timestamps."""
    segment_words: dict[Any, list[tuple[float, float, bool]]] = {}
    for token in proj.get("tokens", []):
        if token.get("kind") != "word" or "seg" not in token:
            continue
        interval = _timed_interval(token)
        if interval is None:
            continue
        segment_words.setdefault(token["seg"], []).append(
            (interval[0], interval[1], bool(token.get("cut")))
        )
    sentence_ranges = [
        (
            min(word[0] for word in words),
            max(word[1] for word in words),
            any(not word[2] for word in words),
        )
        for words in segment_words.values()
        if words
    ]
    word_cuts = consecutive_word_cut_intervals(
        proj.get("tokens", []), **cut_composition_params(proj)
    )

    candidates: list[tuple[float, str]] = []
    for gap in proj.get("audio_gaps", []):
        try:
            start = float(gap.get("detected_start", gap.get("start")))
            end = float(gap.get("detected_end", gap.get("end")))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        midpoint = (start + end) / 2.0
        owners = [
            has_kept_word
            for sentence_start, sentence_end, has_kept_word in sentence_ranges
            if sentence_start - MERGE_EPS <= midpoint <= sentence_end + MERGE_EPS
        ]
        if owners:
            eligible = any(owners)
        else:
            eligible = not any(
                start >= cut_start - MERGE_EPS and end <= cut_end + MERGE_EPS
                for cut_start, cut_end in word_cuts
            )
        gap_id = str(gap.get("id", ""))
        if eligible and gap_id:
            candidates.append((start, gap_id))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [gap_id for _start, gap_id in candidates]


def project_response_payload(proj: dict[str, Any]) -> dict[str, Any]:
    """Project JSON plus authoritative derived intervals, without mutating it."""
    payload = dict(proj)
    params = cut_composition_params(proj)
    payload["word_cut_intervals"] = consecutive_word_cut_intervals(
        proj.get("tokens", []), **params
    )
    payload["transcript_cut_intervals"] = transcript_cut_intervals(
        proj.get("tokens", []), **params
    )
    payload["cut_intervals"] = cut_intervals_from_tokens(proj)
    payload["scoped_gap_candidate_ids"] = scoped_gap_candidate_ids(proj)
    payload["voice_enhancement"] = validate_voice_enhancement(
        proj.get("voice_enhancement")
    )
    return payload


def parse_silencedetect_output(text: str, duration: float) -> list[tuple[float, float]]:
    """Parse FFmpeg silencedetect lines, including leading/trailing silence."""
    events: list[tuple[str, float]] = []
    for kind, value in re.findall(r"silence_(start|end):\s*(-?\d+(?:\.\d+)?)", text):
        events.append((kind, float(value)))
    gaps: list[tuple[float, float]] = []
    pending: Optional[float] = None
    for kind, value in events:
        value = max(0.0, min(float(duration), value))
        if kind == "start":
            pending = value
        else:
            start = 0.0 if pending is None else pending
            if value > start:
                gaps.append((start, value))
            pending = None
    if pending is not None and duration > pending:
        gaps.append((pending, float(duration)))
    return merge_intervals(gaps)


def detect_audio_gaps(
    media_path: str, duration: float, min_duration: float, silence_db: float,
) -> list[dict[str, Any]]:
    """Detect actual low-energy audio intervals without touching Whisper data."""
    if not FFMPEG:
        resolve_ffmpeg()
    filt = f"silencedetect=noise={silence_db:.1f}dB:d={min_duration:.3f}"
    result = _run([
        FFMPEG, "-hide_banner", "-nostats", "-i", media_path,
        "-vn", "-af", filt, "-f", "null", "-",
    ])
    if result.returncode != 0:
        raise RuntimeError(f"audio gap analysis failed: {result.stderr.strip()[-400:]}")
    gaps = parse_silencedetect_output(result.stderr, duration)
    return [
        {
            "id": f"agap-{index + 1}",
            "detected_start": round(start, 3), "detected_end": round(end, 3),
            "start": round(start, 3), "end": round(end, 3),
            "cut": False, "manual": False,
        }
        for index, (start, end) in enumerate(gaps)
        if end - start >= min_duration - MERGE_EPS
    ]


def validate_audio_gap_settings(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        raw = {}
    try:
        min_duration = float(raw.get("min_duration_s", ACTIVE_GAP_MIN_DEFAULT))
        silence_db = float(raw.get("silence_db", ACTIVE_GAP_DB_DEFAULT))
        keep_pause = float(raw.get("keep_pause_s", ACTIVE_GAP_KEEP_DEFAULT))
    except (TypeError, ValueError) as exc:
        raise ValueError("gap settings must be numbers") from exc
    if not 0.05 <= min_duration <= 30:
        raise ValueError("minimum gap must be between 0.05 and 30 seconds")
    if not -80 <= silence_db <= -10:
        raise ValueError("silence sensitivity must be between -80 and -10 dB")
    if not 0 <= keep_pause <= 5:
        raise ValueError("retained pause must be between 0 and 5 seconds")
    return {
        "min_duration_s": round(min_duration, 3),
        "silence_db": round(silence_db, 1),
        "keep_pause_s": round(keep_pause, 3),
    }


def validate_audio_gaps(
    raw: Any, existing: list[dict[str, Any]], duration: float,
) -> list[dict[str, Any]]:
    """Accept edits only for detector-created stable IDs; preserve evidence."""
    if not isinstance(raw, list):
        raise ValueError("audio gaps must be a list")
    by_id = {str(gap.get("id")): gap for gap in existing}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        gap_id = str(item.get("id", ""))
        original = by_id.get(gap_id)
        if original is None or gap_id in seen:
            continue
        seen.add(gap_id)
        try:
            start = max(0.0, min(duration, float(item.get("start"))))
            end = max(0.0, min(duration, float(item.get("end"))))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid boundaries for {gap_id}") from exc
        if end <= start:
            raise ValueError(f"gap {gap_id} must end after it starts")
        row = dict(original)
        row.update(
            start=round(start, 3), end=round(end, 3),
            cut=bool(item.get("cut", False)),
            manual=bool(item.get("manual", original.get("manual", False))),
        )
        out.append(row)
    for gap in existing:
        if str(gap.get("id")) not in seen:
            out.append(dict(gap))
    out.sort(key=lambda gap: (float(gap["detected_start"]), str(gap["id"])))
    return out


def gap_bulk_cut_bounds(
    gap: dict[str, Any], keep_pause: float,
) -> Optional[tuple[float, float]]:
    start = float(gap["detected_start"])
    end = float(gap["detected_end"])
    if end - start <= keep_pause + MERGE_EPS:
        return None
    side = keep_pause / 2.0
    return (round(start + side, 3), round(end - side, 3))


def audio_waveform_peaks(
    media_path: str, start: float, end: float, points: int,
) -> list[float]:
    """Return normalized mono peak buckets for a small gap-boundary editor."""
    if not FFMPEG:
        resolve_ffmpeg()
    span = max(0.01, end - start)
    cmd = [
        FFMPEG, "-v", "error", "-ss", f"{start:.6f}", "-t", f"{span:.6f}",
        "-i", media_path, "-vn", "-ac", "1", "-ar", "8000",
        "-f", "f32le", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"waveform decode failed: {error[-300:]}")
    samples = array("f")
    usable = len(result.stdout) - (len(result.stdout) % samples.itemsize)
    samples.frombytes(result.stdout[:usable])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return [0.0] * points
    bucket = max(1, len(samples) // points)
    peaks = [
        max((abs(value) for value in samples[i:i + bucket]), default=0.0)
        for i in range(0, len(samples), bucket)
    ][:points]
    if len(peaks) < points:
        peaks.extend([0.0] * (points - len(peaks)))
    maximum = max(peaks) or 1.0
    return [round(min(1.0, peak / maximum), 4) for peak in peaks]


def detect_silence_intervals(
    media_path: str,
    duration: float,
    min_duration: float = EXPORT_SILENCE_MIN_S,
    silence_db: float = EXPORT_SILENCE_DB_DEFAULT,
) -> list[tuple[float, float]]:
    """Detect ephemeral source silences for export; never mutate project state."""
    if not FFMPEG:
        resolve_ffmpeg()
    filt = f"silencedetect=noise={silence_db:.1f}dB:d={min_duration:.3f}"
    result = _run([
        FFMPEG, "-hide_banner", "-nostats", "-i", media_path,
        "-vn", "-af", filt, "-f", "null", "-",
    ])
    if result.returncode != 0:
        raise RuntimeError(f"audio silence analysis failed: {result.stderr.strip()[-400:]}")
    return parse_silencedetect_output(result.stderr, duration)


def refine_export_keep_edges(
    keeps: list[tuple[float, float]],
    silences: list[tuple[float, float]],
    duration: float,
    max_shift: float = EXPORT_EDGE_MAX_SHIFT_S,
    max_trim: float = EXPORT_EDGE_TRIM_MAX_S,
) -> list[tuple[float, float]]:
    """Snap established keep edges to nearby real silence, or change nothing.

    This pure export-only operation intentionally runs after cut composition and
    keep-list construction. A kept start maps to the speech-side silence end; a
    kept end maps to the speech-side silence start. Distant/invalid candidates
    are rejected rather than guessed.

    The two directions carry different risk, so they carry different budgets.
    Moving an edge *inward*, toward the speech the keep exists for, only ever
    drops audio ffmpeg measured as silence, so it gets ``max_trim``. Moving an
    edge *outward* restores audio that cut composition deliberately excluded and
    is capped at the much smaller ``max_shift``, which exists only to recover a
    quiet word attack or release.
    """
    original = [(float(start), float(end)) for start, end in keeps]
    window = max(float(max_shift), float(max_trim))
    if not original or not silences or window <= 0:
        return original
    cleaned_silences: list[tuple[float, float]] = []
    for raw_start, raw_end in silences:
        try:
            silence_start, silence_end = float(raw_start), float(raw_end)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(silence_start) or not math.isfinite(silence_end):
            continue
        silence_start = max(0.0, silence_start)
        silence_end = min(float(duration), silence_end)
        if silence_end - silence_start >= EXPORT_SILENCE_MIN_S - MERGE_EPS:
            cleaned_silences.append((silence_start, silence_end))
    valid_silences = merge_intervals(cleaned_silences)
    if not valid_silences:
        return original

    def nearest_target(boundary: float, edge: str) -> float:
        candidates: list[tuple[float, float, float]] = []
        for silence_start, silence_end in valid_silences:
            if silence_end < boundary - window or silence_start > boundary + window:
                continue
            if silence_start <= boundary <= silence_end:
                distance = 0.0
            else:
                distance = min(abs(boundary - silence_start), abs(boundary - silence_end))
            target = silence_end if edge == "start" else silence_start
            shift = abs(target - boundary)
            # Inward for a keep start is later; inward for a keep end is earlier.
            trimming = target > boundary if edge == "start" else target < boundary
            budget = max_trim if trimming else max_shift
            if shift <= budget + MERGE_EPS:
                candidates.append((distance, shift, target))
        if not candidates:
            return boundary
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][2]

    refined: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(original):
        new_start = start if start <= MERGE_EPS else nearest_target(start, "start")
        new_end = end if duration - end <= MERGE_EPS else nearest_target(end, "end")
        new_start = max(0.0, min(float(duration), new_start))
        new_end = max(0.0, min(float(duration), new_end))
        # A restored edge must never reach back across the keep that precedes it.
        if refined and new_start < refined[-1][1]:
            new_start = max(new_start, refined[-1][1], start)
        if new_end <= new_start + MERGE_EPS:
            new_start, new_end = start, end
        refined.append((new_start, new_end))

    for index, (start, end) in enumerate(refined):
        if end <= start + MERGE_EPS:
            return original
        if index and start < refined[index - 1][1] - MERGE_EPS:
            return original
    return [(round(start, 6), round(end, 6)) for start, end in refined]


# --------------------------------------------------------------------------
# Tokens from faster-whisper segments
# --------------------------------------------------------------------------

def build_tokens(
    fw_segments: list[dict[str, Any]], duration: float, gap_threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Alternating word/gap tokens (chronological) + segment records.

    A gap token is emitted for any inter-word silence >= gap_threshold,
    including leading/trailing silence, so dead air at the edges is cuttable.
    """
    tokens: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    tid = 0
    prev_end = 0.0

    def push_gap(start: float, end: float) -> None:
        nonlocal tid
        if end - start >= gap_threshold:
            tokens.append(
                {"id": tid, "kind": "gap",
                 "start": round(start, 3), "end": round(end, 3), "cut": False}
            )
            tid += 1

    for seg in fw_segments:
        words = seg["words"]
        if not words:
            continue
        seg_id = len(segments)
        segments.append(
            {"id": seg_id, "start": round(words[0]["start"], 3),
             "end": round(words[-1]["end"], 3),
             "text": " ".join(w["text"] for w in words)}
        )
        for w in words:
            push_gap(prev_end, w["start"])
            tokens.append(
                {"id": tid, "kind": "word", "text": w["text"],
                 "start": round(w["start"], 3), "end": round(w["end"], 3),
                 "seg": seg_id, "cut": False}
            )
            tid += 1
            prev_end = w["end"]
    push_gap(prev_end, duration)
    return tokens, segments


# --------------------------------------------------------------------------
# Forced word alignment (isolated WhisperX worker)
# --------------------------------------------------------------------------

def alignment_segments_for_project(
    proj: dict[str, Any], max_span: float = 30.0, padding: float = 2.5,
) -> list[dict[str, Any]]:
    """Group transcript sections into bounded, padded forced-alignment windows."""
    duration = float(proj.get("duration_s") or 0.0)
    source: list[dict[str, Any]] = []
    for segment in proj.get("segments", []):
        text = str(segment.get("text", "")).strip()
        interval = _timed_interval(segment)
        if text and interval is not None:
            source.append({
                "start": interval[0], "end": interval[1], "text": text,
            })
    source.sort(key=lambda item: (item["start"], item["end"]))
    if not source:
        words = [
            token for token in proj.get("tokens", [])
            if token.get("kind") == "word" and _timed_interval(token) is not None
        ]
        if words:
            source = [{
                "start": float(words[0]["start"]),
                "end": float(words[-1]["end"]),
                "text": " ".join(str(word.get("text", "")) for word in words),
            }]

    grouped: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for segment in source:
        if current and segment["end"] - current[0]["start"] > max_span:
            grouped.append({
                "start": max(0.0, current[0]["start"] - padding),
                "end": min(duration, current[-1]["end"] + padding),
                "text": " ".join(item["text"] for item in current),
            })
            current = []
        current.append(segment)
    if current:
        grouped.append({
            "start": max(0.0, current[0]["start"] - padding),
            "end": min(duration, current[-1]["end"] + padding),
            "text": " ".join(item["text"] for item in current),
        })
    return grouped


def alignment_python_path() -> Optional[Path]:
    override = os.environ.get("RETAKE_ALIGN_PYTHON")
    candidates = [Path(override)] if override else []
    candidates.extend([
        MODELS_DIR / "alignment-runtime" / "Scripts" / "python.exe",
        MODELS_DIR / "alignment-runtime" / "bin" / "python",
    ])
    if importlib.util.find_spec("whisperx") is not None:
        candidates.append(Path(sys.executable))
    return next((path.resolve() for path in candidates if path.is_file()), None)


def alignment_worker_call(proj: dict[str, Any], device: str) -> dict[str, Any]:
    """Run one device attempt out of process so failures release all GPU state."""
    python = alignment_python_path()
    if python is None:
        raise RuntimeError(
            "alignment runtime is not installed under models/alignment-runtime"
        )
    payload = {
        "device": device,
        "source_path": str(Path(proj["source_path"]).resolve()),
        "language": str(proj.get("language") or "en"),
        "segments": alignment_segments_for_project(proj),
        "model_dir": str((MODELS_DIR / "alignment").resolve()),
        "ffmpeg": FFMPEG,
    }
    if not payload["segments"]:
        raise RuntimeError("project transcript contains no alignable sections")
    environment = os.environ.copy()
    environment.pop("HF_HUB_OFFLINE", None)
    environment.pop("TRANSFORMERS_OFFLINE", None)
    environment["HF_HOME"] = str((MODELS_DIR / "alignment" / "huggingface").resolve())
    nltk_data = (MODELS_DIR / "alignment" / "nltk").resolve()
    nltk_data.mkdir(parents=True, exist_ok=True)
    environment["NLTK_DATA"] = str(nltk_data)
    result = subprocess.run(
        [str(python), str(ROOT / "alignment_worker.py")],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=environment, timeout=7200,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-1200:] or "alignment worker failed"
        raise RuntimeError(detail)
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("alignment worker returned invalid output") from exc
    if not isinstance(output.get("words"), list):
        raise RuntimeError("alignment worker returned no word timings")
    return output


def run_forced_alignment(proj: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Attempt CUDA first and retry exactly once on CPU."""
    failures: list[str] = []
    for device in ("cuda", "cpu"):
        with _STATE_LOCK:
            ALIGNMENT_STATE.update(
                device=device, fallback=device == "cpu", error=None,
            )
        try:
            return alignment_worker_call(proj, device), device == "cpu"
        except Exception as exc:
            failures.append(f"{device}: {exc}")
            if device == "cuda":
                log.warning("GPU alignment failed; retrying on CPU: %s", exc)
                continue
            raise RuntimeError(" | ".join(failures)) from exc
    raise RuntimeError(" | ".join(failures) or "alignment failed")


def _aligned_word_is_trustworthy(word: dict[str, Any]) -> bool:
    try:
        start, end = float(word["start"]), float(word["end"])
        score = float(word.get("score", 0.0))
    except (KeyError, TypeError, ValueError):
        return False
    return (
        math.isfinite(start) and math.isfinite(end)
        and 0.0 <= start < end
        and score >= ALIGNMENT_MIN_SCORE
        and end - start <= ALIGNMENT_MAX_WORD_SECONDS
    )


def recalibrate_legacy_gap_timings(
    tokens: list[dict[str, Any]], duration: float,
) -> None:
    """Constrain saved transcript gaps between their newly aligned neighbors."""
    next_words: list[Optional[dict[str, Any]]] = [None] * len(tokens)
    following: Optional[dict[str, Any]] = None
    for index in range(len(tokens) - 1, -1, -1):
        next_words[index] = following
        if tokens[index].get("kind") == "word":
            following = tokens[index]

    previous: Optional[dict[str, Any]] = None
    for index, token in enumerate(tokens):
        if token.get("kind") == "word":
            previous = token
            continue
        following = next_words[index]
        previous_interval = _timed_interval(previous) if previous is not None else None
        following_interval = _timed_interval(following) if following is not None else None
        start = previous_interval[1] if previous_interval is not None else 0.0
        end = following_interval[0] if following_interval is not None else duration
        if previous is not None and not previous.get("cut"):
            start += WORD_CUT_SAFETY_S
        if following is not None and not following.get("cut"):
            end -= WORD_CUT_SAFETY_S
        start = max(0.0, min(duration, start))
        end = max(0.0, min(duration, end))
        if end <= start + MERGE_EPS:
            end = start
        token["start"] = round(start, 3)
        token["end"] = round(end, 3)


def apply_aligned_word_timings(
    proj: dict[str, Any], result: dict[str, Any], fallback: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a validated copy with only confidently mapped word times changed."""
    candidate = json.loads(json.dumps(proj))
    original_words = [
        token for token in proj.get("tokens", []) if token.get("kind") == "word"
    ]
    candidate_words = [
        token for token in candidate.get("tokens", []) if token.get("kind") == "word"
    ]
    aligned_words = [
        word for word in result.get("words", [])
        if isinstance(word, dict) and str(word.get("word", "")).strip()
    ]
    original_norm = [norm_word(str(word.get("text", ""))) for word in original_words]
    aligned_norm = [norm_word(str(word.get("word", ""))) for word in aligned_words]
    matcher = difflib.SequenceMatcher(
        None, original_norm, aligned_norm, autojunk=False
    )
    accepted_indices: set[int] = set()
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            old_index = block.a + offset
            aligned = aligned_words[block.b + offset]
            if not _aligned_word_is_trustworthy(aligned):
                continue
            candidate_words[old_index]["start"] = round(float(aligned["start"]), 3)
            candidate_words[old_index]["end"] = round(float(aligned["end"]), 3)
            candidate_words[old_index]["alignment_score"] = round(
                float(aligned.get("score", 0.0)), 4
            )
            accepted_indices.add(old_index)

    # Independently padded windows can occasionally align a boundary word into
    # the previous window. Revert only the conflicting proposal; never reorder
    # transcript tokens or guess a replacement time.
    for _pass in range(2):
        changed = False
        for index in range(1, len(candidate_words)):
            previous = _timed_interval(candidate_words[index - 1])
            current = _timed_interval(candidate_words[index])
            if previous is None or current is None or current[0] + MERGE_EPS >= previous[0]:
                continue
            revert_index = (
                index if index in accepted_indices
                else index - 1 if index - 1 in accepted_indices
                else None
            )
            if revert_index is None:
                continue
            original = original_words[revert_index]
            candidate_words[revert_index]["start"] = original.get("start")
            candidate_words[revert_index]["end"] = original.get("end")
            candidate_words[revert_index].pop("alignment_score", None)
            accepted_indices.remove(revert_index)
            changed = True
        if not changed:
            break

    total = len(original_words)
    accepted = len(accepted_indices)
    coverage = accepted / total if total else 0.0
    if coverage < ALIGNMENT_MIN_COVERAGE:
        raise RuntimeError(
            f"alignment coverage {coverage:.1%} is below "
            f"the required {ALIGNMENT_MIN_COVERAGE:.0%}"
        )

    old_signature = [
        (token.get("id"), token.get("kind"), token.get("text"), bool(token.get("cut")))
        for token in proj.get("tokens", [])
    ]
    new_signature = [
        (token.get("id"), token.get("kind"), token.get("text"), bool(token.get("cut")))
        for token in candidate.get("tokens", [])
    ]
    if old_signature != new_signature:
        raise RuntimeError("alignment changed token identity or edit decisions")

    duration = float(candidate.get("duration_s") or 0.0)
    recalibrate_legacy_gap_timings(candidate.get("tokens", []), duration)
    previous_start = -1.0
    for index, word in enumerate(candidate_words):
        interval = _timed_interval(word)
        original_interval = _timed_interval(original_words[index])
        if interval is None:
            if original_interval is None and index not in accepted_indices:
                continue
            raise RuntimeError("alignment produced an invalid word interval")
        if interval[0] < 0 or interval[1] > duration + 0.05:
            raise RuntimeError("alignment produced an invalid word interval")
        if interval[0] + MERGE_EPS < previous_start:
            if index not in accepted_indices and original_interval == interval:
                continue
            raise RuntimeError("alignment produced non-chronological word timings")
        previous_start = interval[0]

    by_segment: dict[Any, list[dict[str, Any]]] = {}
    for word in candidate_words:
        if "seg" in word:
            by_segment.setdefault(word["seg"], []).append(word)
    for segment in candidate.get("segments", []):
        words = by_segment.get(segment.get("id"), [])
        if words:
            segment["start"] = round(min(float(word["start"]) for word in words), 3)
            segment["end"] = round(max(float(word["end"]) for word in words), 3)

    stats = {
        "version": ALIGNMENT_VERSION,
        "status": "aligned",
        "device": str(result.get("device") or ("cpu" if fallback else "cuda")),
        "fallback": bool(fallback),
        "model": str(result.get("model") or "whisperx-default"),
        "matched_words": accepted,
        "total_words": total,
        "coverage": round(coverage, 6),
        "elapsed_s": float(result.get("elapsed_s") or 0.0),
        "completed_at": utc_now(),
    }
    candidate["alignment"] = stats
    return candidate, stats


def alignment_source_identity(proj: dict[str, Any]) -> dict[str, Any]:
    source = Path(str(proj["source_path"])).resolve()
    stat = source.stat()
    transcript = [
        (token.get("id"), token.get("text"))
        for token in proj.get("tokens", [])
        if token.get("kind") == "word"
    ]
    digest = hashlib.sha256(
        json.dumps(transcript, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "version": ALIGNMENT_VERSION,
        "source_name": source.name,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "transcript_sha256": digest,
    }


def alignment_status_payload() -> dict[str, Any]:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        state = dict(ALIGNMENT_STATE)
    if proj is None:
        return {
            "state": "no_project", "ready": False, "running": False,
            "runtime_ready": alignment_python_path() is not None,
        }
    try:
        identity = alignment_source_identity(proj)
    except (KeyError, FileNotFoundError, OSError):
        return {
            "state": "unavailable", "ready": False, "running": False,
            "runtime_ready": alignment_python_path() is not None,
            "error": "Source media is unavailable.",
        }
    saved = proj.get("alignment") if isinstance(proj.get("alignment"), dict) else {}
    if (
        saved.get("status") == "aligned"
        and saved.get("version") == ALIGNMENT_VERSION
        and saved.get("source") == identity
    ):
        return {
            "state": "ready", "ready": True, "running": False,
            "runtime_ready": True, "error": None,
            "device": saved.get("device"), "fallback": bool(saved.get("fallback")),
            "coverage": saved.get("coverage"),
        }
    if state.get("running") and state.get("identity") == identity:
        return {
            "state": "preparing", "ready": False, "running": True,
            "runtime_ready": alignment_python_path() is not None, "error": None,
            "device": state.get("device"), "fallback": bool(state.get("fallback")),
            "coverage": None,
        }
    if state.get("error") and state.get("identity") == identity:
        return {
            "state": "failed", "ready": False, "running": False,
            "runtime_ready": alignment_python_path() is not None,
            "error": str(state["error"]), "device": state.get("device"),
            "fallback": bool(state.get("fallback")), "coverage": None,
        }
    return {
        "state": "available" if alignment_python_path() is not None else "unavailable",
        "ready": False, "running": False,
        "runtime_ready": alignment_python_path() is not None,
        "error": (
            None if alignment_python_path() is not None
            else "Accurate timing runtime is not installed."
        ),
        "device": None, "fallback": False, "coverage": None,
    }


def recalibrate_alignment_job(
    proj: dict[str, Any], identity: dict[str, Any],
) -> None:
    try:
        project_dir = project_directory_for_media(proj["source_path"])
        if project_dir is None:
            raise RuntimeError("project folder is unavailable")
        backup = project_dir / (
            "project.alignment-backup-"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + ".json"
        )
        atomic_write_json(backup, proj)
        set_status("align", 4, "aligning transcript on GPU")
        result, fallback = run_forced_alignment(proj)
        set_status("align", 92, "validating aligned word boundaries")

        with _STATE_LOCK:
            latest = CURRENT["project"]
            if latest is None or alignment_source_identity(latest) != identity:
                raise RuntimeError("project changed while timing was being aligned")
            candidate, stats = apply_aligned_word_timings(latest, result, fallback)
            stats["source"] = identity
            candidate["alignment"] = stats
            candidate["updated_at"] = utc_now()
            atomic_write_json(project_dir / "project.json", candidate)
            CURRENT["project"] = candidate
            ALIGNMENT_STATE.update(
                running=False, identity=identity, error=None,
                device=stats["device"], fallback=stats["fallback"],
                coverage=stats["coverage"],
            )
        set_status(
            "ready", 100,
            (
                f"timing calibrated on {stats['device'].upper()} "
                f"({stats['coverage']:.1%} words)"
            ),
        )
        log.info(
            "alignment ready device=%s fallback=%s coverage=%.1f%% backup=%s",
            stats["device"], stats["fallback"], stats["coverage"] * 100, backup,
        )
    except Exception as exc:
        with _STATE_LOCK:
            ALIGNMENT_STATE.update(
                running=False, identity=identity, error=str(exc), coverage=None,
            )
        fail(f"timing recalibration failed: {exc}")
    finally:
        JOB_LOCK.release()


# --------------------------------------------------------------------------
# Retake coloring: word repeats + sentence clusters (deterministic, no LLM)
# --------------------------------------------------------------------------

STOPWORDS = set(
    """a an and are as at be but by for from had has have he her his i if in is it
    its me my no not of on or our she so that the their them then there they this
    to up us was we were what when who will with you your don don't do does did
    just about into out over under again more can could would should than too very
    s t re ve ll d m o im it's i'm""".split()
)

_norm_re = re.compile(r"[^\w']+", re.UNICODE)


def norm_word(w: str) -> str:
    return _norm_re.sub("", w.lower()).strip("'_")


def norm_sentence(s: str) -> str:
    return " ".join(norm_word(w) for w in s.split() if norm_word(w))


def find_word_repeats(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Any non-stopword appearing >= 3 times gets a chip entry."""
    buckets: dict[str, list[int]] = {}
    for t in tokens:
        if t["kind"] != "word":
            continue
        w = norm_word(t["text"])
        if len(w) < 3 or w in STOPWORDS:
            continue
        buckets.setdefault(w, []).append(t["id"])
    return [
        {"word": w, "token_ids": ids}
        for w, ids in sorted(buckets.items())
        if len(ids) >= 3
    ]


def clusters_from_similarity(
    segments: list[dict[str, Any]], similarity: Any,
) -> list[dict[str, Any]]:
    """Build conservative complete-link retake groups from a similarity matrix.

    A new member must match every existing member directly. This deliberately
    prevents the transitive chain failure where weak A↔B↔C links merged most of
    a recording into one cluster.
    """
    texts = [norm_sentence(s["text"]) for s in segments]

    def matches(i: int, j: int) -> bool:
        return retake_text_match(texts[i], texts[j], float(similarity[i][j]))

    groups: list[list[int]] = []
    for j, seg in enumerate(segments):
        candidates: list[tuple[float, int]] = []
        for gi, members in enumerate(groups):
            if len(members) >= RETAKE_MAX_MEMBERS:
                continue
            if float(seg["start"]) - float(segments[members[0]]["start"]) > RETAKE_MAX_SPAN_S:
                continue
            if all(matches(i, j) for i in members):
                mean_sim = sum(float(similarity[i][j]) for i in members) / len(members)
                candidates.append((mean_sim, gi))
        if candidates:
            _, best = max(candidates)
            groups[best].append(j)
        else:
            groups.append([j])

    kept = sorted((g for g in groups if len(g) >= 2), key=lambda g: g[0])
    return [{"id": cid, "members": members} for cid, members in enumerate(kept)]


def retake_text_match(a: str, b: str, semantic_similarity: Optional[float] = None) -> bool:
    """True only for a close full repeat or a short repeated opening prefix."""
    from rapidfuzz import fuzz

    if not a or not b:
        return False
    if (
        (semantic_similarity is None or semantic_similarity >= 0.82)
        and fuzz.token_set_ratio(a, b) >= 75
        and fuzz.ratio(a, b) >= 65
    ):
        return True
    short, long_ = (a, b) if len(a.split()) <= len(b.split()) else (b, a)
    short_words, long_words = short.split(), long_.split()
    if not (1 < len(short_words) < 8 and len(short_words) < len(long_words)):
        return False
    opening = " ".join(long_words[:len(short_words)])
    return (
        (semantic_similarity is None or semantic_similarity >= 0.70)
        and fuzz.ratio(short, opening) >= 88
    )


def sanitize_clusters(
    segments: list[dict[str, Any]], clusters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop unsafe legacy clusters when loading projects created by older builds."""
    safe: list[list[int]] = []
    for cluster in clusters:
        try:
            members = sorted({int(i) for i in cluster.get("members", [])})
        except (TypeError, ValueError):
            continue
        if len(members) < 2 or len(members) > RETAKE_MAX_MEMBERS:
            continue
        if members[0] < 0 or members[-1] >= len(segments):
            continue
        span = float(segments[members[-1]]["end"]) - float(segments[members[0]]["start"])
        if span > RETAKE_MAX_SPAN_S:
            continue
        final_text = norm_sentence(segments[members[-1]]["text"])
        if not all(
            retake_text_match(norm_sentence(segments[index]["text"]), final_text)
            for index in members[:-1]
        ):
            continue
        safe.append(members)
    return [{"id": cid, "members": members} for cid, members in enumerate(safe)]


def find_clusters(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Embed transcript segments and form conservative, cohesive retake groups."""
    if len(segments) < 2:
        return []
    set_status("cluster", 5, "loading local sentence embedder")
    import numpy as np
    from sentence_transformers import SentenceTransformer

    texts = [norm_sentence(s["text"]) for s in segments]
    model = SentenceTransformer(
        "all-MiniLM-L6-v2", device="cpu", cache_folder=str(MODELS_DIR),
        local_files_only=True,
    )
    set_status("cluster", 40, "embedding transcript sections")
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    similarity = np.asarray(emb) @ np.asarray(emb).T
    set_status("cluster", 70, "matching cohesive retakes")
    return clusters_from_similarity(segments, similarity)


# --------------------------------------------------------------------------
# Transcription job (background thread)
# --------------------------------------------------------------------------

def transcribe_job(media_path: str, requested_language: Optional[str] = None) -> None:
    """Probe → whisper (GPU turbo → CPU small fallback) → tokens → colors → save."""
    try:
        set_status("probe", 2, "probing media")
        probe = probe_media(media_path)
        duration = probe["duration_s"]
        if duration <= 0:
            raise RuntimeError("media has zero duration")

        from faster_whisper import WhisperModel  # heavy: import inside the job

        attempts = [
            ("large-v3-turbo", "cuda", "int8_float16"),
            ("large-v3-turbo", "cpu", "int8"),
        ]
        fw_segments: list[dict[str, Any]] = []
        language = "en"
        last_err: Optional[Exception] = None
        for model_name, device, compute in attempts:
            try:
                set_status(
                    "load_model", 3,
                    f"loading local whisper {model_name} on {device}",
                )
                model = WhisperModel(
                    model_name, device=device, compute_type=compute,
                    download_root=str(MODELS_DIR), local_files_only=True,
                )
                set_status("transcribe", 4, f"transcribing with {model_name} ({device})")
                seg_iter, info = model.transcribe(
                    media_path, word_timestamps=True, vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 200},
                    condition_on_previous_text=False,
                    language=requested_language or None,
                )
                language = info.language or "en"
                fw_segments = []
                for seg in seg_iter:
                    words = [
                        {"text": w.word.strip(), "start": float(w.start), "end": float(w.end)}
                        for w in (seg.words or [])
                        if w.word.strip()
                    ]
                    if words:
                        fw_segments.append({"words": words})
                    pct = 4 + min(1.0, float(seg.end) / duration) * 90
                    set_status("transcribe", pct,
                               f"transcribing… {seg.end:.0f}s / {duration:.0f}s")
                last_err = None
                break
            except Exception as e:  # fall through to CPU small
                last_err = e
                log.warning("whisper attempt %s/%s failed: %s", model_name, device, e)
        if last_err is not None:
            raise last_err

        tokens, segments = build_tokens(fw_segments, duration, GAP_THRESHOLD_DEFAULT)
        # The CTranslate2 model can occupy most of a small GPU. Release it
        # before the isolated forced-aligner gets its GPU-first attempt.
        if "model" in locals():
            del model
            import gc
            gc.collect()

        alignment_meta: dict[str, Any]
        alignment_draft = {
            "source_path": str(Path(media_path).resolve()),
            "duration_s": round(duration, 3),
            "language": language,
            "tokens": tokens,
            "segments": segments,
        }
        if alignment_python_path() is not None:
            try:
                set_status("align", 92, "calibrating word timing on GPU")
                aligned_result, alignment_fallback = run_forced_alignment(
                    alignment_draft
                )
                aligned_draft, alignment_meta = apply_aligned_word_timings(
                    alignment_draft, aligned_result, alignment_fallback
                )
                tokens = aligned_draft["tokens"]
                segments = aligned_draft["segments"]
                alignment_meta["source"] = alignment_source_identity(aligned_draft)
            except Exception as alignment_error:
                log.warning(
                    "new transcript alignment failed; saving unaligned timing: %s",
                    alignment_error,
                )
                alignment_meta = {
                    "version": ALIGNMENT_VERSION, "status": "unaligned",
                    "error": str(alignment_error), "completed_at": utc_now(),
                }
        else:
            alignment_meta = {
                "version": ALIGNMENT_VERSION, "status": "unaligned",
                "error": "Accurate timing runtime is not installed.",
                "completed_at": utc_now(),
            }
        set_status("cluster", 95, "coloring retakes")
        clusters = find_clusters(segments)
        word_repeats = find_word_repeats(tokens)

        with _STATE_LOCK:
            project_dir = CURRENT.get("project_dir")
        if not project_dir:
            raise RuntimeError("project folder was not prepared")
        project_dir = Path(project_dir)
        proj = {
            "schema_version": 1,
            "source_path": str(Path(media_path).resolve()),
            "duration_s": round(duration, 3),
            "probe": {
                "container": probe["container"], "vcodec": probe["vcodec"],
                "acodec": probe["acodec"], "has_video": probe["has_video"],
                "fps": probe["fps"], "width": probe.get("width"),
                "height": probe.get("height"), "sample_rate": probe.get("sample_rate"),
                "channels": probe.get("channels"),
                "video_streams": probe.get("video_streams", 0),
                "audio_streams": probe.get("audio_streams", 0),
            },
            "language": language,
            "requested_language": requested_language or "auto",
            "alignment": alignment_meta,
            "tokens": tokens,
            "segments": segments,
            "clusters": clusters,
            "word_repeats": word_repeats,
            "markers": [],
            "gap_threshold_s": GAP_THRESHOLD_DEFAULT,
            "audio_gap_settings": {
                "min_duration_s": ACTIVE_GAP_MIN_DEFAULT,
                "silence_db": ACTIVE_GAP_DB_DEFAULT,
                "keep_pause_s": ACTIVE_GAP_KEEP_DEFAULT,
            },
            "audio_gaps": [],
            "audio_gaps_analyzed": False,
            "voice_enhancement": dict(VOICE_ENHANCEMENT_DEFAULTS),
            "project_name": project_dir.name,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "last_opened_at": utc_now(),
            "source_filename": CURRENT.get("source_filename") or Path(media_path).name,
        }
        with _STATE_LOCK:
            CURRENT["project"] = proj
            CURRENT["media_path"] = proj["source_path"]
            CURRENT["project_dir"] = str(project_dir)
        save_current_project()
        set_status("ready", 100, "transcription complete")
        log.info("project ready: %d tokens, %d segments, %d clusters",
                 len(tokens), len(segments), len(clusters))
    except Exception as e:
        fail(f"transcription failed: {e}", back_to="idle")
    finally:
        JOB_LOCK.release()


def audio_gap_detection_job(settings: dict[str, float]) -> None:
    """Analyze source audio and persist export-only gap candidates."""
    try:
        with _STATE_LOCK:
            proj = CURRENT["project"]
            if proj is None:
                raise RuntimeError("no project loaded")
            source = str(proj["source_path"])
            duration = float(proj["duration_s"])
            has_audio = proj.get("probe", {}).get("acodec") is not None
        if not has_audio:
            raise RuntimeError("this media has no audio track")
        set_status("gaps", 5, "detecting real audio gaps")
        gaps = detect_audio_gaps(
            source, duration, settings["min_duration_s"], settings["silence_db"]
        )
        with _STATE_LOCK:
            current = CURRENT["project"]
            if current is None or str(current.get("source_path")) != source:
                raise RuntimeError("project changed during gap analysis")
            current["audio_gap_settings"] = settings
            current["audio_gaps"] = gaps
            current["audio_gaps_analyzed"] = True
        save_current_project()
        set_status("ready", 100, f"detected {len(gaps)} real audio gaps")
    except Exception as exc:
        fail(f"audio gap detection failed: {exc}")
    finally:
        JOB_LOCK.release()





def validate_ai_attachments(raw: Any) -> tuple[list[dict[str, str]], list[str]]:
    """Validate browser-decoded plain-text references and enforce context limits."""
    if not isinstance(raw, list):
        return [], []
    if len(raw) > AI_ATTACHMENT_MAX_FILES:
        raise ValueError(f"attach at most {AI_ATTACHMENT_MAX_FILES} text files")
    accepted: list[dict[str, str]] = []
    warnings: list[str] = []
    remaining = AI_ATTACHMENT_MAX_CHARS
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            warnings.append(f"attachment {i}: ignored invalid entry")
            continue
        name = Path(str(item.get("name", f"reference-{i}.txt"))).name[:160]
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            warnings.append(f"{name}: ignored empty or unreadable text")
            continue
        encoded = content.encode("utf-8", errors="replace")
        controls = sum(ord(ch) < 32 and ch not in "\n\r\t" for ch in content)
        if b"\x00" in encoded or controls > max(3, len(content) // 100):
            warnings.append(f"{name}: ignored because it looks binary")
            continue
        if len(encoded) > AI_ATTACHMENT_MAX_BYTES:
            warnings.append(f"{name}: ignored (larger than 512 KiB)")
            continue
        if remaining <= 0:
            warnings.append(f"{name}: omitted because the reference limit was reached")
            continue
        clipped = content[:remaining]
        if len(clipped) < len(content):
            warnings.append(f"{name}: truncated to fit the 12,000-character reference limit")
        accepted.append({"name": name, "content": clipped})
        remaining -= len(clipped)
    return accepted, warnings


def _proposal_for_segment(
    segment: dict[str, Any], reason: str, source: str,
) -> dict[str, Any]:
    text = " ".join(str(segment.get("text", "")).split())
    return {
        "sentence_ids": [int(segment["id"])],
        "start": float(segment["start"]),
        "end": float(segment["end"]),
        "excerpt": text[:157] + ("…" if len(text) > 157 else ""),
        "reason": reason[:120],
        "source": source,
    }


def deterministic_retake_proposals(
    segments: list[dict[str, Any]], clusters: list[dict[str, Any]],
    excluded_ids: Optional[set[int]] = None,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Propose every earlier member of a trusted retake group, never the last."""
    excluded = excluded_ids or set()
    proposals: list[dict[str, Any]] = []
    proposed_ids: set[int] = set()
    for cluster in sanitize_clusters(segments, clusters):
        members = cluster["members"]
        for take_no, index in enumerate(members[:-1], 1):
            segment = segments[index]
            sid = int(segment["id"])
            if sid in excluded:
                continue
            proposals.append(_proposal_for_segment(
                segment,
                f"Earlier retake ({take_no}/{len(members)}); keeping the final take",
                "retake rule",
            ))
            proposed_ids.add(sid)
    return proposals, proposed_ids


_GENERIC_TARGET_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "from", "in", "on", "part",
    "parts", "section", "sections", "video", "take", "takes", "bad", "weaker",
    "repeated", "attempts", "false", "starts", "filler", "fillers", "pause", "pauses",
    "dead", "air", "awkward", "obvious", "remove", "delete", "cut", "discard", "trim",
}


def explicit_target_terms(instructions: str) -> set[str]:
    """Extract terms only from specific user removal commands, not default guidance."""
    terms: set[str] = set()
    for sentence in re.split(r"[.\n;]+", instructions.lower()):
        match = re.search(r"\b(?:remove|delete|cut|drop)\b\s+(.+)", sentence)
        if not match:
            continue
        phrase = match.group(1)
        if any(generic in phrase for generic in (
            "bad take", "repeated attempt", "false start", "obvious filler",
            "awkward dead air", "awkward pause",
        )):
            continue
        terms.update(
            word for word in re.findall(r"\w+", phrase, flags=re.UNICODE)
            if len(word) >= 3 and word not in _GENERIC_TARGET_WORDS
        )
    return terms


def enforce_llm_budget(
    proposals: list[dict[str, Any]], segments: list[dict[str, Any]], duration_s: float,
) -> bool:
    """Refuse an implausibly broad automated edit.

    Originally a guard on the local GGUF's output; it now guards operations an
    MCP client supplies, which carry the same risk of one confident mistake
    deleting most of a recording.
    """
    if not proposals:
        return True
    max_count = max(1, min(12, int(len(segments) * 0.15)))
    proposed_duration = sum(float(p["end"]) - float(p["start"]) for p in proposals)
    return len(proposals) <= max_count and proposed_duration <= max(1.0, duration_s * 0.15)


_TIME_RE = re.compile(
    r"(?<!\d)(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:[.,]\d+)?)"
)
_DETAIL_ACTION_RE = re.compile(
    r"(?:شيل|احذف|خل[ّ]?ي|خلي|remove|delete|cut|keep|retain|gap|silence|"
    r"لازم\s+تختار|choose|←)", re.IGNORECASE,
)


def _ascii_digits(text: str) -> str:
    """Normalize Arabic-Indic digits without changing line positions materially."""
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return text.translate(table)


def timestamps_in_text(text: str) -> list[float]:
    values: list[float] = []
    for match in _TIME_RE.finditer(_ascii_digits(text)):
        hours = int(match.group("h") or 0)
        minutes = int(match.group("m"))
        seconds = float(match.group("s").replace(",", "."))
        if minutes < 60 and seconds < 60:
            values.append(hours * 3600 + minutes * 60 + seconds)
    return values


def is_detailed_edit_request(instructions: str) -> bool:
    """Concrete quotes/times/clusters select exact-edit mode, not advisory cleanup."""
    text = instructions.strip()
    if not text:
        return False
    has_action = bool(_DETAIL_ACTION_RE.search(text))
    has_quote = bool(re.search(r'["“”«»].+?["“”«»]', text, re.DOTALL))
    has_time = bool(_TIME_RE.search(_ascii_digits(text)))
    has_cluster = bool(re.search(r"\bcluster\s*\d+", text, re.IGNORECASE))
    return has_cluster or (has_action and (has_quote or has_time))


def _instruction_context(
    instructions: str, duration_s: float,
) -> tuple[list[str], dict[int, tuple[float, float]]]:
    """Return original lines and deterministic inherited time bounds per line."""
    lines = instructions.splitlines()
    inherited = (0.0, duration_s)
    bounds: dict[int, tuple[float, float]] = {}
    for line_no, line in enumerate(lines, 1):
        times = timestamps_in_text(line)
        is_action = bool(_DETAIL_ACTION_RE.search(line))
        if len(times) >= 2 and not is_action:
            inherited = (max(0.0, times[0]), min(duration_s, times[1]))
        bounds[line_no] = inherited
    return lines, bounds


def _planner_operation_bounds(
    operation: dict[str, Any], source_line: str,
    inherited: tuple[float, float], duration_s: float,
) -> tuple[float, float]:
    """Derive hard bounds from source text; model times are accepted only if quoted there."""
    source_times = timestamps_in_text(source_line)
    start_raw, end_raw = operation.get("start"), operation.get("end")

    def supported(value: Any) -> Optional[float]:
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if any(abs(value - item) <= 0.11 for item in source_times) else None

    start, end = supported(start_raw), supported(end_raw)
    if start is not None or end is not None:
        lo = inherited[0] if start is None else start
        hi = inherited[1] if end is None else end
    elif len(source_times) >= 2:
        lo, hi = source_times[0], source_times[1]
    elif len(source_times) == 1:
        marker = source_times[0]
        normalized_source = _ascii_digits(source_line)
        from_to_end = bool(re.search(
            r"(?:\bfrom\b|من).*(?:\bend\b|للآ?خر|للاخر|النهاية|وطالع|onwards?)",
            normalized_source, re.IGNORECASE,
        ))
        if from_to_end:
            return max(0.0, marker), inherited[1]
        arrow = "→" in source_line or "->" in source_line
        if arrow:
            # Numeric formatting may differ; direction is determined from text around the arrow.
            normalized = _ascii_digits(source_line)
            arrow_at = normalized.find("→")
            if arrow_at < 0:
                arrow_at = normalized.find("->")
            time_at = _TIME_RE.search(normalized)
            if time_at and time_at.start() < arrow_at:
                lo, hi = marker, inherited[1]
            else:
                lo, hi = inherited[0], marker
        else:
            lo, hi = marker - 3.0, marker + 3.0
    else:
        lo, hi = inherited
    lo, hi = max(0.0, lo), min(duration_s, hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _actionable_instruction_lines(lines: list[str]) -> set[int]:
    return {
        i for i, line in enumerate(lines, 1)
        if _DETAIL_ACTION_RE.search(line) and not re.search(r"^\s*cluster\b", line, re.IGNORECASE)
    }


def validate_planned_operations(
    parsed: Any, instructions: str, duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate planner authority and return operations plus visible unparsed rows."""
    lines, inherited_bounds = _instruction_context(instructions, duration_s)
    raw_operations = parsed.get("operations", []) if isinstance(parsed, dict) else []
    operations: list[dict[str, Any]] = []
    covered: set[int] = set()
    allowed = {"cut_phrase", "keep_phrase", "cut_gap", "needs_decision"}
    for raw in raw_operations if isinstance(raw_operations, list) else []:
        if not isinstance(raw, dict) or not isinstance(raw.get("line"), (int, float)):
            continue
        line_no = int(raw["line"])
        if not (1 <= line_no <= len(lines)) or str(raw.get("action", "")) not in allowed:
            continue
        action = str(raw["action"])
        phrase = " ".join(str(raw.get("phrase", "")).split()).strip(' "“”«»')
        source_line = lines[line_no - 1].strip()
        if action in {"cut_phrase", "keep_phrase"}:
            needle = norm_sentence(phrase.replace("…", " ").replace("...", " "))
            haystack = norm_sentence(source_line)
            block_reference = bool(re.search(r"(?:\bblock\b|بلوك)\s*\d+", source_line, re.IGNORECASE))
            document_text = norm_sentence(instructions)
            if not needle or (needle not in haystack and not (block_reference and needle in document_text)):
                continue
        lo, hi = _planner_operation_bounds(
            raw, source_line, inherited_bounds[line_no], duration_s
        )
        occurrence_raw = str(raw.get("occurrence"))
        occurrence: Optional[str] = None
        if occurrence_raw == "first" and re.search(
            r"(?:\bfirst\b|أول|الاول|الأول)", source_line, re.IGNORECASE
        ):
            occurrence = "first"
        elif occurrence_raw == "last" and re.search(
            r"(?:\blast\b|\bsecond\b|التاني[ةه]?|الثاني[ةه]?)", source_line, re.IGNORECASE
        ):
            occurrence = "last"
        operations.append({
            "operation_id": len(operations) + 1,
            "line": line_no,
            "action": action,
            "phrase": phrase,
            "start": lo,
            "end": hi,
            "instruction": source_line,
            "reason": str(raw.get("reason", action)).strip()[:160] or action,
            "all_matches": bool(raw.get("all_matches")) and bool(
                re.search(r"(?:\ball\b|\bevery\b|كل|كله|كلها|كاملة|بالكامل)", source_line, re.IGNORECASE)
            ),
            "occurrence": occurrence,
        })
        covered.add(line_no)

    unresolved: list[dict[str, Any]] = []
    for line_no in sorted(_actionable_instruction_lines(lines) - covered):
        source = lines[line_no - 1].strip()
        unresolved.append({
            "operation_id": len(operations) + len(unresolved) + 1,
            "line": line_no,
            "action": "unparsed",
            "instruction": source,
            "status": "not_found",
            "reason": "The planner could not safely parse this instruction",
            "token_ids": [],
            "start": inherited_bounds[line_no][0],
            "end": inherited_bounds[line_no][1],
            "excerpt": source[:180],
            "source": "AI instruction",
        })
    return operations, unresolved


def _normalized_phrase_words(phrase: str) -> list[str]:
    clean = phrase.replace("…", " ").replace("...", " ")
    return [word for word in (norm_word(part) for part in clean.split()) if word]


def _candidate_words(
    tokens: list[dict[str, Any]], start: float, end: float,
) -> list[dict[str, Any]]:
    return [
        token for token in tokens
        if token.get("kind") == "word"
        and float(token["start"]) >= start - 0.15
        and float(token["end"]) <= end + 0.15
    ]


def _word_windows(
    words: list[dict[str, Any]], phrase_len: int,
) -> Iterator[list[dict[str, Any]]]:
    for size in range(max(1, phrase_len - 2), phrase_len + 3):
        for index in range(0, len(words) - size + 1):
            window = words[index:index + size]
            if any(
                float(window[j + 1]["start"]) - float(window[j]["end"]) > 0.75
                for j in range(len(window) - 1)
            ):
                continue
            yield window


def resolve_phrase_tokens(
    tokens: list[dict[str, Any]], phrase: str, start: float, end: float,
    all_matches: bool = False, occurrence: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve a quote to a unique exact/strong-approximate word-token span."""
    target = _normalized_phrase_words(phrase)
    words = _candidate_words(tokens, start, end)
    if not target:
        return {"status": "not_found", "token_ids": [], "reason": "Empty phrase"}
    exact: list[list[dict[str, Any]]] = []
    for index in range(0, len(words) - len(target) + 1):
        window = words[index:index + len(target)]
        if [norm_word(item.get("text", "")) for item in window] == target:
            if all(
                float(window[j + 1]["start"]) - float(window[j]["end"]) <= 0.75
                for j in range(len(window) - 1)
            ):
                exact.append(window)
    if len(exact) == 1:
        return {"status": "exact", "token_ids": [int(t["id"]) for t in exact[0]], "score": 1.0}
    if len(exact) > 1 and all_matches:
        ids = sorted({int(token["id"]) for window in exact for token in window})
        return {"status": "exact", "token_ids": ids, "score": 1.0}
    if len(exact) > 1 and occurrence in {"first", "last"}:
        chosen = exact[0] if occurrence == "first" else exact[-1]
        return {"status": "exact", "token_ids": [int(t["id"]) for t in chosen], "score": 1.0}
    if len(exact) > 1:
        return {"status": "ambiguous", "token_ids": [], "reason": f"{len(exact)} exact matches inside the allowed time range"}

    target_text = " ".join(target)
    scored: list[tuple[float, list[dict[str, Any]]]] = []
    for window in _word_windows(words, len(target)):
        candidate = " ".join(norm_word(item.get("text", "")) for item in window)
        score = difflib.SequenceMatcher(None, target_text, candidate).ratio()
        if score >= 0.90:
            scored.append((score, window))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return {"status": "not_found", "token_ids": [], "reason": "No phrase match inside the allowed time range"}
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.08:
        return {"status": "ambiguous", "token_ids": [], "reason": "Multiple approximate matches are too similar"}
    return {
        "status": "approximate",
        "token_ids": [int(t["id"]) for t in scored[0][1]],
        "score": round(scored[0][0], 3),
    }


def resolve_detailed_operations(
    operations: list[dict[str, Any]], tokens: list[dict[str, Any]],
    unresolved: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Resolve keeps first, then exact cut proposals, preserving source order."""
    token_by_id = {int(token["id"]): token for token in tokens}
    results: list[dict[str, Any]] = []
    protected: set[int] = set()

    def base_result(op: dict[str, Any]) -> dict[str, Any]:
        return {
            "operation_id": int(op["operation_id"]),
            "line": int(op["line"]),
            "instruction": op["instruction"],
            "action": op["action"],
            "reason": op["reason"],
            "source": "AI instruction",
            "selected": False,
        }

    # Protection is independent of the user's presentation order.
    keep_resolution: dict[int, dict[str, Any]] = {}
    for op in operations:
        if op["action"] != "keep_phrase":
            continue
        match = resolve_phrase_tokens(
            tokens, op["phrase"], op["start"], op["end"],
            bool(op.get("all_matches")), op.get("occurrence"),
        )
        keep_resolution[int(op["operation_id"])] = match
        if match["status"] in {"exact", "approximate"}:
            protected.update(match["token_ids"])

    for op in operations:
        result = base_result(op)
        if op["action"] == "needs_decision":
            result.update(status="needs_decision", token_ids=[], start=op["start"], end=op["end"], excerpt=op["instruction"][:180])
        elif op["action"] == "cut_gap":
            ids = [
                int(token["id"]) for token in tokens
                if token.get("kind") == "gap"
                and float(token["end"]) > op["start"]
                and float(token["start"]) < op["end"]
            ]
            status = "exact" if ids else "not_found"
            excerpt = " + ".join(
                f"gap {token_by_id[token_id]['start']:.3f}–{token_by_id[token_id]['end']:.3f}"
                for token_id in ids
            ) or "No matching gap"
            result.update(status=status, token_ids=ids, start=op["start"], end=op["end"], excerpt=excerpt, selected=bool(ids))
        else:
            occurrence = op.get("occurrence")
            if occurrence is None and op["action"] == "cut_phrase":
                current_words = _normalized_phrase_words(op["phrase"])
                for later in operations:
                    if int(later["operation_id"]) <= int(op["operation_id"]):
                        continue
                    if later["action"] not in {"cut_phrase", "keep_phrase"}:
                        continue
                    if abs(float(later["start"]) - float(op["start"])) > 0.15 or abs(float(later["end"]) - float(op["end"])) > 0.15:
                        continue
                    later_words = _normalized_phrase_words(later["phrase"])
                    if len(later_words) > len(current_words) and later_words[:len(current_words)] == current_words:
                        occurrence = "first"
                        break
            match = keep_resolution.get(int(op["operation_id"])) or resolve_phrase_tokens(
                tokens, op["phrase"], op["start"], op["end"],
                bool(op.get("all_matches")), occurrence,
            )
            ids = list(match.get("token_ids", []))
            status = str(match["status"])
            if op["action"] == "cut_phrase" and set(ids) & protected:
                status, ids = "conflict", []
                match["reason"] = "This cut overlaps words explicitly protected by a keep instruction"
            matched = [token_by_id[token_id] for token_id in ids]
            excerpt = " ".join(str(token.get("text", "")) for token in matched) or op["phrase"]
            result.update(
                status=status, token_ids=ids,
                start=float(matched[0]["start"]) if matched else op["start"],
                end=float(matched[-1]["end"]) if matched else op["end"],
                excerpt=excerpt[:220], score=match.get("score"),
                selected=op["action"] == "cut_phrase" and status == "exact",
            )
            if match.get("reason"):
                result["reason"] = str(match["reason"])
        results.append(result)
        log.info(
            "AI operation line=%s action=%s bounds=%.3f..%.3f status=%s tokens=%s",
            op["line"], op["action"], op["start"], op["end"],
            result["status"], result.get("token_ids", []),
        )
    results.extend(unresolved or [])
    results.sort(key=lambda item: (int(item.get("line", 0)), int(item.get("operation_id", 0))))
    word_ids = {int(token["id"]) for token in tokens if token.get("kind") == "word"}
    already_cut_words = {
        int(token["id"]) for token in tokens
        if token.get("kind") == "word" and token.get("cut")
    }
    proposed_cut_words = {
        token_id for result in results
        if result.get("action") == "cut_phrase"
        and result.get("status") in {"exact", "approximate"}
        for token_id in result.get("token_ids", [])
        if token_id in word_ids
    }
    if proposed_cut_words and len(word_ids - already_cut_words - proposed_cut_words) < 3:
        for result in results:
            if result.get("action") in {"cut_phrase", "cut_gap"} and result.get("token_ids"):
                result.update(
                    status="conflict", token_ids=[], selected=False,
                    reason="Refused because accepting all resolved instructions would remove all meaningful speech",
                )
    return results


def _compact_transcript_index(proj: dict[str, Any]) -> str:
    return "\n".join(
        f'SEGMENT {segment["id"]} [{segment["start"]:.3f} -> {segment["end"]:.3f}]: {segment["text"]}'
        for segment in proj.get("segments", [])
    )


_QUOTED_RE = re.compile(r'["“«](.+?)["”»]')
_CUT_MARKER_RE = re.compile(r"(?:شيل|احذف|remove|delete|cut|drop)", re.IGNORECASE)
_KEEP_MARKER_RE = re.compile(r"(?:خل[ّ]?ي|خلي|keep|retain)", re.IGNORECASE)


def _nearest_action(text_before: str, text_after: str) -> Optional[str]:
    found: list[tuple[int, str]] = []
    found.extend((match.start(), "cut_phrase") for match in _CUT_MARKER_RE.finditer(text_before))
    found.extend((match.start(), "keep_phrase") for match in _KEEP_MARKER_RE.finditer(text_before))
    if found:
        return max(found, key=lambda item: item[0])[1]
    after: list[tuple[int, str]] = []
    after.extend((match.start(), "cut_phrase") for match in _CUT_MARKER_RE.finditer(text_after))
    after.extend((match.start(), "keep_phrase") for match in _KEEP_MARKER_RE.finditer(text_after))
    return min(after, key=lambda item: item[0])[1] if after else None


def deterministic_instruction_plan(
    instructions: str,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Parse the common quoted Arabic/English edit-list syntax without inference."""
    raw: list[dict[str, Any]] = []
    covered: set[int] = set()
    for line_no, line in enumerate(instructions.splitlines(), 1):
        if not _DETAIL_ACTION_RE.search(line):
            continue
        times = timestamps_in_text(line)
        all_matches = bool(re.search(
            r"(?:\ball\b|\bevery\b|كل|كله|كلها|كاملة|بالكامل)", line, re.IGNORECASE
        ))
        quoted = list(_QUOTED_RE.finditer(line))
        for quote in quoted:
            action = _nearest_action(line[:quote.start()], line[quote.end():])
            if not action:
                continue
            occurrence = None
            if re.search(r"(?:\bfirst\b|أول|الاول|الأول)", line, re.IGNORECASE):
                occurrence = "first"
            if action == "keep_phrase" and re.search(r"(?:\blast\b|\bsecond\b|التاني[ةه]?|الثاني[ةه]?)", line, re.IGNORECASE):
                occurrence = "last"
            raw.append({
                "line": line_no, "action": action, "phrase": quote.group(1),
                "start": None,
                "end": None, "all_matches": all_matches,
                "occurrence": occurrence,
                "reason": "Exact quoted instruction",
            })
            covered.add(line_no)
            if action == "cut_phrase" and re.search(
                r"(?:خل[ّ]?ي|خلي|keep)\s+(?:التاني[ةه]?|الثاني[ةه]?|the\s+second|the\s+last)",
                line[quote.end():], re.IGNORECASE,
            ):
                raw.append({
                    "line": line_no, "action": "keep_phrase", "phrase": quote.group(1),
                    "start": None, "end": None, "all_matches": False,
                    "occurrence": "last", "reason": "Explicitly keep the later occurrence",
                })

        is_gap = bool(re.search(r"(?:\bgaps?\b|\bsilence\b|سكوت)", line, re.IGNORECASE))
        if is_gap and _CUT_MARKER_RE.search(line) and times:
            if len(times) == 1:
                pairs = [(None, None)]
            else:
                pairs = [
                    (times[index], times[index + 1])
                    for index in range(0, len(times) - 1, 2)
                ]
            for start, end in pairs:
                raw.append({
                    "line": line_no, "action": "cut_gap", "phrase": "",
                    "start": start, "end": end, "all_matches": True,
                    "reason": "Explicit gap instruction",
                })
            covered.add(line_no)

        if re.search(
            r"(?:لازم\s+تختار|اختار\s+الرقم|needs?\s+(?:a\s+)?decision|"
            r"choose\s+the\s+correct|contradict)", line, re.IGNORECASE,
        ):
            raw.append({
                "line": line_no, "action": "needs_decision", "phrase": "",
                "start": None, "end": None, "all_matches": False,
                "reason": "The instruction requires an editorial or factual choice",
            })
            covered.add(line_no)
    return raw, covered


def _compact_transcript_for_lines(
    proj: dict[str, Any], line_numbers: set[int], instructions: str,
) -> str:
    lines, bounds = _instruction_context(instructions, float(proj["duration_s"]))
    relevant = [bounds[line] for line in line_numbers if 1 <= line <= len(lines)]
    if not relevant:
        return _compact_transcript_index(proj)
    return "\n".join(
        f'SEGMENT {segment["id"]} [{segment["start"]:.3f} -> {segment["end"]:.3f}]: {segment["text"]}'
        for segment in proj.get("segments", [])
        if any(float(segment["end"]) >= lo and float(segment["start"]) <= hi for lo, hi in relevant)
    )


def plan_detailed_instructions(
    proj: dict[str, Any],
    instructions: str,
    references: str = "none",
    client_operations: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn edit instructions into exact token selections, deterministically.

    Language understanding is the only part Retake does not do itself. It used
    to come from a local GGUF; it now comes from whatever MCP client is driving
    the session, which submits ``client_operations`` in the same shape. Either
    way the operations are only ever *parsed* language: `validate_planned_operations`
    and `resolve_detailed_operations` below decide which real tokens are touched,
    so a caller can never name a token ID or invent a span.

    Lines this function's own deterministic planner already understood are
    resolved without any client involvement; the rest are reported as unresolved
    when no client supplied an operation for them, rather than guessed at.
    """
    deterministic_raw, deterministic_lines = deterministic_instruction_plan(instructions)
    all_lines = instructions.splitlines()
    target_lines = _actionable_instruction_lines(all_lines) - deterministic_lines
    supplied = [op for op in (client_operations or []) if isinstance(op, dict)]
    if not target_lines and not supplied:
        operations, unresolved = validate_planned_operations(
            {"operations": deterministic_raw}, instructions, float(proj["duration_s"])
        )
        return resolve_detailed_operations(operations, proj["tokens"], unresolved), []

    warnings: list[str] = []
    unaddressed = sorted(
        line for line in target_lines
        if not any(op.get("line") == line for op in supplied)
    )
    if unaddressed:
        warnings.append(
            "no operation was supplied for instruction line(s) "
            + ", ".join(str(line) for line in unaddressed)
        )
    model_raw = supplied
    operations, unresolved = validate_planned_operations(
        {"operations": deterministic_raw + (model_raw if isinstance(model_raw, list) else [])},
        instructions, float(proj["duration_s"])
    )
    proposals = resolve_detailed_operations(operations, proj["tokens"], unresolved)
    return proposals, warnings


def assistant_job(
    instructions: str,
    attachments: Optional[list[dict[str, str]]] = None,
    initial_warnings: Optional[list[str]] = None,
    client_operations: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Resolve edit instructions into reviewable proposals, without any model.

    Two paths, both deterministic. Concrete instructions -- quoted phrases,
    timestamps, cluster references -- go through the instruction planner and
    resolve to exact token IDs. Anything else falls back to the conservative
    retake detector, which proposes only repeated takes the app already grouped.
    Proposals are never applied here; they are staged for review.
    """
    try:
        proj = CURRENT["project"]
        assert proj is not None
        attachments = attachments or []
        references = (
            "\n\n".join(
                f'--- Reference file: {a["name"]} ---\n{a["content"]}'
                for a in attachments
            )
            or "none"
        )
        warnings: list[str] = list(initial_warnings or [])

        if client_operations or is_detailed_edit_request(instructions):
            set_status("assistant", 20, "matching exact words and gaps")
            detailed, detailed_warnings = plan_detailed_instructions(
                proj, instructions, references, client_operations
            )
            warnings.extend(detailed_warnings)
            with _STATE_LOCK:
                AI_STATE.update(
                    running=False, proposals=detailed, warnings=warnings,
                    done=True, mode="detailed",
                )
            executable = sum(
                item.get("action") in {"cut_phrase", "cut_gap"}
                and item.get("status") in {"exact", "approximate"}
                for item in detailed
            )
            unresolved_count = sum(
                item.get("status") not in {"exact", "approximate"} for item in detailed
            )
            set_status(
                "ready", 100,
                f"Matched {executable} exact edits; {unresolved_count} need review",
            )
            return

        set_status("assistant", 20, "grouping repeated takes")
        segments = proj["segments"]
        clusters = sanitize_clusters(segments, proj.get("clusters", []))
        already_cut = fully_cut_segments(proj)
        proposals, _deterministic_ids = deterministic_retake_proposals(
            segments, clusters, already_cut
        )
        if not enforce_llm_budget(proposals, segments, float(proj["duration_s"])):
            warnings.append(
                "the retake detector selected an unusually broad edit; it was refused"
            )
            proposals = []

        unique: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for proposal in proposals:
            sid = int(proposal["sentence_ids"][0])
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            proposal["token_ids"] = [
                int(token["id"]) for token in proj["tokens"]
                if token.get("kind") == "word" and int(token.get("seg", -1)) == sid
            ]
            proposal["action"] = "cut_phrase"
            proposal["status"] = "exact"
            proposal["instruction"] = proposal.get("reason", "Repeated take")
            unique.append(proposal)

        with _STATE_LOCK:
            AI_STATE.update(
                running=False, proposals=unique, warnings=warnings,
                done=True, mode="advisory",
            )
        set_status("ready", 100, f"Found {len(unique)} repeated takes to review")
    except Exception as e:
        with _STATE_LOCK:
            AI_STATE.update(running=False, done=True)
        fail(f"assistant pass failed: {e}")
    finally:
        JOB_LOCK.release()


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def _fmt_srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_tc(t: float, fps: float) -> str:
    frames_total = int(round(t * fps))
    fpsr = max(1, int(round(fps)))
    f = frames_total % fpsr
    s_total = frames_total // fpsr
    h, rem = divmod(s_total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def export_output_path(proj: dict[str, Any], suffix: str, ext: Optional[str] = None) -> Path:
    src = Path(proj["source_path"])
    project_dir = project_directory_for_media(str(src))
    if project_dir is None:
        raise RuntimeError("project export folder is unavailable")
    exports_dir = project_dir / "exports"
    exports_dir.mkdir(exist_ok=True)
    ext = ext if ext is not None else src.suffix
    out = exports_dir / f"{src.stem}.{suffix}{ext}"
    i = 1
    while out.exists():
        out = exports_dir / f"{src.stem}.{suffix}.{i}{ext}"
        i += 1
    return out


def latest_media_export() -> Optional[Path]:
    """Newest completed Retake media export for the currently open project."""
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return None
        source_path = str(proj.get("source_path", ""))

    project_dir = project_directory_for_media(source_path)
    if project_dir is None:
        return None
    exports_dir = (project_dir / "exports").resolve()
    if not exports_dir.is_dir():
        return None

    source = Path(source_path)
    pattern = re.compile(
        rf"^{re.escape(source.stem)}\.retake_cut(?:\.\d+)?\.[^.]+$",
        re.IGNORECASE,
    )
    media_suffixes = {
        source.suffix.lower(), ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
        ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    }
    candidates: list[tuple[int, str, Path]] = []
    for candidate in exports_dir.iterdir():
        try:
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.suffix.lower() not in media_suffixes
                or not pattern.fullmatch(candidate.name)
            ):
                continue
            resolved = candidate.resolve()
            resolved.relative_to(exports_dir)
            candidates.append((resolved.stat().st_mtime_ns, resolved.name, resolved))
        except (OSError, ValueError):
            continue
    if not candidates:
        return None
    return max(candidates)[2]


AUDIO_CODEC_MAP: dict[str, list[str]] = {
    "aac": ["-c:a", "aac", "-b:a", "192k"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
    "opus": ["-c:a", "libopus", "-b:a", "128k"],
    "vorbis": ["-c:a", "libvorbis", "-q:a", "5"],
    "flac": ["-c:a", "flac"],
}
def display_dimensions(properties: dict[str, Any]) -> tuple[Optional[int], Optional[int]]:
    """Dimensions as a player displays them after applying rotation metadata."""
    width, height = properties.get("width"), properties.get("height")
    rotation = int(properties.get("rotation") or 0) % 360
    if rotation in (90, 270):
        return height, width
    return width, height


def _orientation_normalization_filters(properties: dict[str, Any]) -> list[str]:
    """Bake source display rotation into pixels, leaving no rotation metadata."""
    rotation = int(properties.get("rotation") or 0) % 360
    if rotation == 90:
        # FFprobe's display-matrix sign is opposite FFmpeg's transpose filter.
        return ["transpose=cclock"]
    if rotation == 270:
        return ["transpose=clock"]
    if rotation == 180:
        return ["hflip", "vflip"]
    return []


def _audio_codec_args(acodec: Optional[str]) -> list[str]:
    if acodec and acodec.startswith("pcm"):
        return ["-c:a", "pcm_s16le"]
    return AUDIO_CODEC_MAP.get(acodec or "", ["-c:a", "aac", "-b:a", "192k"])


def _run_ffmpeg_with_progress(
    cmd: list[str],
    edited_duration: float,
    label: str,
    phase: str = "export",
) -> None:
    """Run ffmpeg, parsing -progress pipe:1 into /status. Raises on failure."""
    proc = subprocess.Popen(
        cmd + ["-progress", "pipe:1", "-nostats"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
            try:
                us = int(line.split("=", 1)[1])
                pct = 5 + min(1.0, (us / 1e6) / max(0.01, edited_duration)) * 90
                set_status(phase, pct, label)
            except ValueError:
                pass
    proc.wait()
    if proc.returncode != 0:
        err = (proc.stderr.read() if proc.stderr else "").strip()
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {err[-400:]}")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return -120.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def audio_energy_profile(media_path: str, sample_rate: int = 16000) -> dict[str, float | bool]:
    """Estimate solo-speech noise/speech levels from 20 ms mono RMS windows."""
    if not FFMPEG:
        resolve_ffmpeg()
    result = subprocess.run(
        [
            FFMPEG, "-v", "error", "-nostdin", "-i", media_path, "-vn",
            "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"audio level analysis failed: {error[-300:]}")
    samples = array("h")
    usable = len(result.stdout) - len(result.stdout) % samples.itemsize
    samples.frombytes(result.stdout[:usable])
    if sys.byteorder != "little":
        samples.byteswap()
    frame_samples = max(1, int(sample_rate * 0.020))
    levels: list[float] = []
    for offset in range(0, len(samples), frame_samples):
        frame = samples[offset:offset + frame_samples]
        if len(frame) < frame_samples // 2:
            continue
        square_mean = sum(float(value) * float(value) for value in frame) / len(frame)
        rms = math.sqrt(square_mean) / 32768.0
        levels.append(20.0 * math.log10(max(rms, 1e-6)))
    if not levels:
        return {
            "noise_floor_db": -120.0, "speech_level_db": -120.0,
            "snr_db": 0.0, "quiet_fraction": 1.0,
            "meaningful_noise": False, "silence_db": EXPORT_SILENCE_DB_DEFAULT,
        }
    noise_floor = _percentile(levels, 0.20)
    speech_level = _percentile(levels, 0.80)
    snr = speech_level - noise_floor
    quiet_cutoff = noise_floor + 2.5
    quiet_fraction = sum(level <= quiet_cutoff for level in levels) / len(levels)
    meaningful_noise = bool(
        quiet_fraction >= 0.25 and noise_floor > -60.0 and 3.0 <= snr < 32.0
    )
    silence_db = max(-55.0, min(-32.0, noise_floor + 3.0))
    return {
        "noise_floor_db": round(noise_floor, 2),
        "speech_level_db": round(speech_level, 2),
        "snr_db": round(snr, 2),
        "quiet_fraction": round(quiet_fraction, 4),
        "meaningful_noise": meaningful_noise,
        "silence_db": round(silence_db, 1),
    }


def _audio_concat_graph(keeps: list[tuple[float, float]], input_label: str = "0:a:0") -> str:
    parts = [
        f"[{input_label}]atrim=start={start:.6f}:end={end:.6f},"
        f"asetpts=PTS-STARTPTS[a{index}]"
        for index, (start, end) in enumerate(keeps)
    ]
    labels = "".join(f"[a{index}]" for index in range(len(keeps)))
    parts.append(f"{labels}concat=n={len(keeps)}:v=0:a=1[out]")
    return ";".join(parts)


def _render_edited_audio_stem(
    proj: dict[str, Any], keeps: list[tuple[float, float]], out: Path,
) -> None:
    sample_rate = int(proj.get("probe", {}).get("sample_rate") or 48000)
    channels = int(proj.get("probe", {}).get("channels") or 1)
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-i", proj["source_path"], "-filter_complex", _audio_concat_graph(keeps),
        "-map", "[out]", "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_f32le", str(out),
    ]
    edited = sum(end - start for start, end in keeps)
    _run_ffmpeg_with_progress(cmd, edited, "building lossless edited audio")


def enhancement_python_path() -> Optional[Path]:
    configured = os.environ.get("RETAKE_ENHANCEMENT_PYTHON", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    runtime = MODELS_DIR / "enhancement-runtime"
    candidate = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if candidate.is_file():
        return candidate
    if importlib.util.find_spec("df") is not None:
        return Path(sys.executable)
    return None


def enhancement_runtime_ready() -> bool:
    return bool(
        enhancement_python_path()
        and (ROOT / "enhancement_worker.py").is_file()
        and (MODELS_DIR / "deepfilternet").is_dir()
    )


def _run_deepfilter_worker(
    source: Path,
    output: Path,
    settings: dict[str, Any],
    expected_duration: float,
    expected_channels: int,
) -> None:
    python = enhancement_python_path()
    model_dir = MODELS_DIR / "deepfilternet"
    if python is None or not model_dir.is_dir():
        raise RuntimeError("the optional DeepFilterNet runtime/model is not installed")
    request = {
        "source_path": str(source), "output_path": str(output),
        "model_dir": str(model_dir),
        "noise_cleanup": float(settings["noise_cleanup"]),
        "original_detail": float(settings["original_detail"]),
    }
    result = subprocess.run(
        [str(python), str(ROOT / "enhancement_worker.py")],
        input=json.dumps(request), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"DeepFilterNet failed: {result.stderr.strip()[-400:]}")
    response = json.loads(result.stdout or "{}")
    properties = probe_media(str(output))
    if abs(float(properties["duration_s"]) - expected_duration) > 0.05:
        raise RuntimeError("DeepFilterNet changed audio duration")
    if int(properties.get("channels") or 0) != int(expected_channels):
        raise RuntimeError("DeepFilterNet changed the channel count")
    if int(response.get("samples") or 0) <= 0:
        raise RuntimeError("DeepFilterNet returned no audio samples")


def _leveling_filter(percent: float) -> str:
    if percent <= 0:
        return "anull"
    ratio = 1.0 + 2.0 * max(0.0, min(100.0, percent)) / 100.0
    return (
        "acompressor=threshold=0.125:"
        f"ratio={ratio:.3f}:attack=20:release=180:makeup=1:knee=2.828"
    )


def _parse_loudnorm_measurement(stderr: str) -> dict[str, float]:
    required = {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"}
    for match in reversed(re.findall(r"\{[^{}]*\}", stderr, flags=re.DOTALL)):
        try:
            data = json.loads(match)
            if required <= data.keys():
                return {key: float(data[key]) for key in required}
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    raise RuntimeError("FFmpeg did not return a readable loudness measurement")


def _normalize_audio_stem(
    source: Path,
    output: Path,
    settings: dict[str, Any],
    sample_rate: int,
    channels: int,
    edited_duration: float,
) -> None:
    target = loudness_target_lufs(float(settings["output_loudness"]))
    leveling = _leveling_filter(float(settings["voice_leveling"]))
    first_filter = (
        f"{leveling},loudnorm=I={target:.1f}:TP={EXPORT_TRUE_PEAK_DB:.1f}:"
        "LRA=11:print_format=json"
    )
    measured = _run([
        FFMPEG, "-hide_banner", "-nostats", "-i", str(source),
        "-af", first_filter, "-f", "null", "-",
    ])
    if measured.returncode != 0:
        raise RuntimeError(f"loudness measurement failed: {measured.stderr.strip()[-400:]}")
    values = _parse_loudnorm_measurement(measured.stderr)
    peak_limit = 10.0 ** (EXPORT_TRUE_PEAK_DB / 20.0)
    second_filter = (
        f"{leveling},loudnorm=I={target:.1f}:TP={EXPORT_TRUE_PEAK_DB:.1f}:LRA=11:"
        f"measured_I={values['input_i']:.3f}:measured_TP={values['input_tp']:.3f}:"
        f"measured_LRA={values['input_lra']:.3f}:"
        f"measured_thresh={values['input_thresh']:.3f}:"
        f"offset={values['target_offset']:.3f}:linear=true:print_format=summary,"
        f"alimiter=limit={peak_limit:.6f}:attack=5:release=50:level=false:latency=1,"
        f"aresample={sample_rate}:async=1:first_pts=0"
    )
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-af", second_filter, "-ar", str(sample_rate), "-ac", str(channels),
        "-c:a", "pcm_f32le", str(output),
    ]
    _run_ffmpeg_with_progress(cmd, edited_duration, "leveling voice and setting loudness")


def prepare_export_audio(
    proj: dict[str, Any],
    keeps: list[tuple[float, float]],
    settings: dict[str, Any],
    work_dir: Path,
) -> tuple[Path, list[str], str]:
    """Create one lossless edited/enhanced audio stem for final mux or encode."""
    settings = validate_voice_enhancement(settings)
    warnings: list[str] = []
    edited_duration = sum(end - start for start, end in keeps)
    sample_rate = int(proj.get("probe", {}).get("sample_rate") or 48000)
    channels = int(proj.get("probe", {}).get("channels") or 1)
    edited = work_dir / "edited.wav"
    _render_edited_audio_stem(proj, keeps, edited)
    if not settings["enabled"]:
        return edited, warnings, "disabled"

    source_for_leveling = edited
    cleanup_status = "unnecessary"
    if float(settings["noise_cleanup"]) > 0:
        profile = audio_energy_profile(str(edited))
        if bool(profile["meaningful_noise"]):
            if enhancement_runtime_ready():
                cleaned = work_dir / "cleaned.wav"
                try:
                    set_status("export", 20, "cleaning background noise conservatively")
                    _run_deepfilter_worker(
                        edited, cleaned, settings, edited_duration, channels
                    )
                    source_for_leveling = cleaned
                    cleanup_status = "applied"
                except Exception as exc:
                    cleaned.unlink(missing_ok=True)
                    log.warning("noise cleanup skipped: %s", exc)
                    warnings.append("noise cleanup was skipped; original voice detail was preserved")
                    cleanup_status = "skipped"
            else:
                warnings.append("noise cleanup model is not installed; loudness and leveling still applied")
                cleanup_status = "skipped"

    normalized = work_dir / "enhanced.wav"
    _normalize_audio_stem(
        source_for_leveling, normalized, settings, sample_rate, channels, edited_duration
    )
    return normalized, warnings, cleanup_status


def _ffmpeg_audio_cut(proj: dict[str, Any], keeps: list[tuple[float, float]], out: Path) -> None:
    """Sample-accurate audio cut+concat via one atrim/concat filtergraph."""
    parts = []
    for i, (a, b) in enumerate(keeps):
        parts.append(f"[0:a]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS[a{i}]")
    concat_in = "".join(f"[a{i}]" for i in range(len(keeps)))
    graph = ";".join(parts) + f";{concat_in}concat=n={len(keeps)}:v=0:a=1[out]"
    cmd = [FFMPEG, "-y", "-hide_banner", "-i", proj["source_path"],
           "-filter_complex", graph, "-map", "[out]",
           *_audio_codec_args(proj["probe"].get("acodec")), str(out)]
    edited = sum(b - a for a, b in keeps)
    _run_ffmpeg_with_progress(cmd, edited, "cutting audio (sample-accurate)")


def _encode_audio_stem(
    proj: dict[str, Any], stem: Path, out: Path, edited_duration: float,
) -> None:
    sample_rate = int(proj.get("probe", {}).get("sample_rate") or 48000)
    channels = int(proj.get("probe", {}).get("channels") or 1)
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(stem),
        "-ar", str(sample_rate), "-ac", str(channels),
        *_audio_codec_args(proj.get("probe", {}).get("acodec")), str(out),
    ]
    _run_ffmpeg_with_progress(cmd, edited_duration, "encoding enhanced audio")


def audio_export_integrity_errors(
    source_path: str, output_path: str, expected_duration: float,
) -> list[str]:
    source = probe_media(source_path)
    output = probe_media(output_path)
    errors: list[str] = []
    if abs(float(output["duration_s"]) - expected_duration) > 0.10:
        errors.append(
            f"duration {output['duration_s']:.3f}s vs expected {expected_duration:.3f}s"
        )
    for key, label in (("sample_rate", "sample rate"), ("channels", "channels")):
        before, after = source.get(key), output.get(key)
        if before is not None and after is not None and before != after:
            errors.append(f"{label} changed from {before} to {after}")
    if output.get("audio_streams") != 1:
        errors.append(f"audio stream count is {output.get('audio_streams')}, expected 1")
    return errors


def _ffmpeg_encoder_usable(encoder: str) -> bool:
    """Test the actual encoder and device, not only FFmpeg's encoder listing."""
    if not FFMPEG:
        resolve_ffmpeg()
    result = _run([
        FFMPEG, "-v", "error", "-f", "lavfi",
        "-i", "color=c=black:s=256x256:r=1", "-frames:v", "1",
        "-an", "-c:v", encoder, "-f", "null", "-",
    ])
    return result.returncode == 0


def preview_gpu_usable() -> bool:
    """Cache a real NVENC device probe for the optional preview job."""
    global _PREVIEW_GPU_USABLE
    with _STATE_LOCK:
        cached = _PREVIEW_GPU_USABLE
    if cached is not None:
        return cached
    usable = _ffmpeg_encoder_usable("h264_nvenc")
    with _STATE_LOCK:
        _PREVIEW_GPU_USABLE = usable
    return usable


def preview_source_identity(source_path: str) -> dict[str, Any]:
    source = Path(source_path).resolve()
    stat = source.stat()
    return {
        "version": PREVIEW_PROXY_VERSION,
        "source_name": source.name,
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def preview_proxy_paths(source_path: str) -> tuple[Path, Path, Path]:
    """Return project-confined final, temporary, and metadata proxy paths."""
    project_dir = project_directory_for_media(source_path)
    if project_dir is None:
        raise RuntimeError("preview folder is unavailable")
    preview_dir = (project_dir / "preview").resolve()
    preview_dir.relative_to(project_dir.resolve())
    return (
        preview_dir / "smooth-preview.mp4",
        preview_dir / "smooth-preview.partial.mp4",
        preview_dir / "smooth-preview.json",
    )


def preview_target_dimensions(properties: dict[str, Any]) -> tuple[int, int]:
    """Even display dimensions fitted inside a 1280x720 orientation box."""
    width, height = display_dimensions(properties)
    if not width or not height:
        raise RuntimeError("source dimensions are unavailable")
    max_width, max_height = ((720, 1280) if height > width else (1280, 720))
    scale = min(1.0, max_width / width, max_height / height)
    target_width = max(2, int(width * scale) // 2 * 2)
    target_height = max(2, int(height * scale) // 2 * 2)
    return target_width, target_height


def _preview_proxy_record(source_path: str) -> Optional[dict[str, Any]]:
    """Validated lightweight discovery without probing media on every UI poll."""
    try:
        final, _partial, metadata = preview_proxy_paths(source_path)
        if final.is_symlink() or metadata.is_symlink() or not final.is_file():
            return None
        stored = json.loads(metadata.read_text(encoding="utf-8"))
        if stored.get("source") != preview_source_identity(source_path):
            return None
        stat = final.stat()
        if (
            int(stored.get("proxy_size", -1)) != stat.st_size
            or int(stored.get("proxy_mtime_ns", -1)) != stat.st_mtime_ns
        ):
            return None
        return {"path": final, "metadata": stored}
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _preview_proxy_artifacts_exist(source_path: str) -> bool:
    try:
        final, partial, metadata = preview_proxy_paths(source_path)
        return final.exists() or partial.exists() or metadata.exists()
    except (OSError, RuntimeError):
        return False


def preview_status_payload() -> dict[str, Any]:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        running = bool(PREVIEW_STATE["running"])
        running_identity = PREVIEW_STATE["identity"]
        error = PREVIEW_STATE["error"]
        active_device = PREVIEW_STATE.get("device")
        fallback = bool(PREVIEW_STATE.get("fallback"))
    if proj is None:
        return {
            "state": "no_project", "ready": False, "running": False,
            "gpu_usable": None, "error": None,
        }

    source_path = str(proj.get("source_path", ""))
    try:
        identity = preview_source_identity(source_path)
    except (FileNotFoundError, OSError):
        return {
            "state": "unavailable", "ready": False, "running": False,
            "gpu_usable": False, "error": "Source media is unavailable.",
        }
    record = _preview_proxy_record(source_path)
    if record is not None:
        stat = record["path"].stat()
        return {
            "state": "ready", "ready": True, "running": False,
            "gpu_usable": preview_gpu_usable(), "error": None, "size": stat.st_size,
            "device": record["metadata"].get("device"),
            "fallback": bool(record["metadata"].get("fallback")),
        }
    if running and running_identity == identity:
        return {
            "state": "preparing", "ready": False, "running": True,
            "gpu_usable": preview_gpu_usable(), "error": None,
            "device": active_device, "fallback": fallback,
        }
    if not proj.get("probe", {}).get("has_video"):
        return {
            "state": "unavailable", "ready": False, "running": False,
            "gpu_usable": False, "error": "Smooth Preview requires video media.",
        }
    if error and running_identity == identity:
        return {
            "state": "failed", "ready": False, "running": False,
            "gpu_usable": preview_gpu_usable(), "error": str(error),
        }
    usable = preview_gpu_usable()
    return {
        "state": (
            "stale" if _preview_proxy_artifacts_exist(source_path) else "absent"
        ),
        "ready": False, "running": False, "gpu_usable": usable,
        "error": None, "device": None, "fallback": not usable,
    }


def _preview_video_filter(
    properties: dict[str, Any], target_width: int, target_height: int,
    device: str = "gpu",
) -> str:
    """Scale in stored orientation, then normalize FFmpeg display rotation."""
    rotation = int(properties.get("rotation") or 0) % 360
    stored_width, stored_height = (
        (target_height, target_width) if rotation in (90, 270)
        else (target_width, target_height)
    )
    if device == "gpu":
        filters = [f"scale_cuda=w={stored_width}:h={stored_height}:format=nv12"]
    elif device == "cpu":
        filters = [
            f"scale=w={stored_width}:h={stored_height}:flags=fast_bilinear",
            "format=nv12",
        ]
    else:
        raise ValueError(f"unknown preview device: {device}")
    orientation_filters = _orientation_normalization_filters(properties)
    if orientation_filters:
        if device == "gpu":
            filters.extend(["hwdownload", "format=nv12"])
        filters.extend(orientation_filters)
    elif device == "gpu":
        # NVENC reliably accepts these downloaded NV12 frames on every tested
        # FFmpeg build; direct CUDA-frame negotiation fails on some builds.
        filters.extend(["hwdownload", "format=nv12"])
    return ",".join(filters)


def preview_proxy_command(
    proj: dict[str, Any], out: Path, device: str = "gpu",
) -> list[str]:
    # Saved projects created by older builds may not include rotation metadata.
    # Probe the immutable source again so portrait media is never flattened wrong.
    properties = probe_media(proj["source_path"])
    target_width, target_height = preview_target_dimensions(properties)
    fps = min(30.0, max(1.0, float(properties.get("fps") or 25.0)))
    fps_fraction = Fraction(fps).limit_denominator(1001)
    fps_expr = (
        str(fps_fraction.numerator)
        if fps_fraction.denominator == 1
        else f"{fps_fraction.numerator}/{fps_fraction.denominator}"
    )
    keyframe_interval = max(1, round(fps * 0.5))
    has_audio = bool(properties.get("audio_streams") or properties.get("acodec"))
    audio_args = (
        ["-map", "0:a:0", "-c:a", "aac", "-b:a", "96k",
         "-af", "aresample=async=1:first_pts=0"]
        if has_audio else ["-an"]
    )
    if device == "gpu":
        input_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        video_args = [
            "-c:v", "h264_nvenc", "-preset", "p3", "-tune", "hq",
            "-rc", "vbr", "-cq", "24", "-b:v", "0",
        ]
    elif device == "cpu":
        input_args = []
        video_args = [
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        ]
    else:
        raise ValueError(f"unknown preview device: {device}")
    return [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        *input_args,
        "-noautorotate", "-display_rotation", "0",
        "-i", proj["source_path"],
        "-map", "0:v:0", *audio_args,
        "-vf", _preview_video_filter(
            properties, target_width, target_height, device=device
        ),
        "-map_metadata", "-1", "-map_chapters", "-1", "-sn", "-dn",
        *video_args,
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-g", str(keyframe_interval), "-keyint_min", str(keyframe_interval),
        "-sc_threshold", "0",
        "-r", fps_expr, "-fps_mode", "cfr",
        "-metadata:s:v:0", "rotate=0",
        "-max_muxing_queue_size", "4096", "-shortest",
        "-movflags", "+faststart", str(out),
    ]


def preview_proxy_integrity_errors(
    source_path: str, output_path: str,
) -> list[str]:
    source = probe_media(source_path)
    output = probe_media(output_path)
    expected_width, expected_height = preview_target_dimensions(source)
    expected_duration = float(source.get("duration_s") or 0.0)
    expected_fps = min(30.0, max(1.0, float(source.get("fps") or 25.0)))
    errors: list[str] = []
    if output.get("vcodec") != "h264" or output.get("video_streams") != 1:
        errors.append("proxy must contain exactly one H.264 video stream")
    expected_audio = 1 if source.get("audio_streams") else 0
    if output.get("audio_streams") != expected_audio:
        errors.append(
            f"audio stream count is {output.get('audio_streams')}, expected {expected_audio}"
        )
    if expected_audio and output.get("acodec") != "aac":
        errors.append("proxy audio codec must be AAC")
    if display_dimensions(output) != (expected_width, expected_height):
        errors.append(
            f"proxy display dimensions are {display_dimensions(output)}, "
            f"expected {(expected_width, expected_height)}"
        )
    if int(output.get("rotation") or 0) % 360:
        errors.append("proxy rotation metadata was not normalized")
    actual_duration = float(output.get("duration_s") or 0.0)
    if abs(actual_duration - expected_duration) > max(0.5, 3.0 / expected_fps):
        errors.append(
            f"proxy duration is {actual_duration:.3f}s, expected {expected_duration:.3f}s"
        )
    actual_fps = float(output.get("fps") or 0.0)
    if actual_fps <= 0 or actual_fps > 30.01 or abs(actual_fps - expected_fps) > 0.01:
        errors.append(f"proxy frame rate is {actual_fps:g}, expected {expected_fps:g}")
    errors.extend(validate_video_timeline(output_path, expected_fps, expected_duration))
    decode = _run([
        FFMPEG, "-v", "error", "-i", output_path, "-t", "2",
        "-map", "0:v:0", "-f", "null", "-",
    ])
    if decode.returncode != 0:
        errors.append(f"proxy decode probe failed: {decode.stderr.strip()[-200:]}")
    return errors


def preview_proxy_job(proj: dict[str, Any], identity: dict[str, Any]) -> None:
    final: Optional[Path] = None
    partial: Optional[Path] = None
    try:
        final, partial, metadata = preview_proxy_paths(proj["source_path"])
        final.parent.mkdir(parents=True, exist_ok=True)
        attempts = ["gpu", "cpu"] if preview_gpu_usable() else ["cpu"]
        attempt_errors: list[str] = []
        selected_device: Optional[str] = None
        for device in attempts:
            partial.unlink(missing_ok=True)
            fallback = device == "cpu"
            with _STATE_LOCK:
                PREVIEW_STATE.update(device=device, fallback=fallback)
            label = (
                "preparing GPU smooth preview"
                if device == "gpu"
                else "preparing smooth preview on CPU fallback"
            )
            try:
                set_status("preview", 3, label)
                command = preview_proxy_command(proj, partial, device=device)
                _run_ffmpeg_with_progress(
                    command, float(proj["duration_s"]), label, phase="preview",
                )
                set_status("preview", 96, f"verifying {device.upper()} smooth preview")
                errors = preview_proxy_integrity_errors(
                    proj["source_path"], str(partial)
                )
                if errors:
                    raise RuntimeError(
                        "integrity check failed: " + " | ".join(errors)
                    )
                selected_device = device
                break
            except Exception as attempt_error:
                attempt_errors.append(f"{device}: {attempt_error}")
                partial.unlink(missing_ok=True)
                if device == "gpu":
                    log.warning(
                        "GPU smooth preview failed; retrying on CPU: %s",
                        attempt_error,
                    )
                    continue
                raise RuntimeError(" | ".join(attempt_errors)) from attempt_error
        if selected_device is None:
            raise RuntimeError(" | ".join(attempt_errors) or "no preview device")
        os.replace(partial, final)
        partial = None
        stat = final.stat()
        fallback = selected_device == "cpu"
        atomic_write_json(metadata, {
            "source": identity,
            "proxy_size": stat.st_size,
            "proxy_mtime_ns": stat.st_mtime_ns,
            "created_at": utc_now(),
            "device": selected_device,
            "fallback": fallback,
        })
        with _STATE_LOCK:
            PREVIEW_STATE.update(
                running=False, identity=identity, error=None,
                device=selected_device, fallback=fallback,
            )
        set_status(
            "ready", 100,
            (
                "GPU smooth preview ready"
                if selected_device == "gpu"
                else "smooth preview ready on CPU fallback"
            ),
        )
        log.info("smooth preview ready on %s: %s", selected_device, final)
    except Exception as exc:
        if partial is not None:
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                log.warning("could not remove partial smooth preview: %s", partial)
        with _STATE_LOCK:
            PREVIEW_STATE.update(running=False, identity=identity, error=str(exc))
        fail(f"smooth preview failed: {exc}")
    finally:
        JOB_LOCK.release()


def _reliable_video_command(
    proj: dict[str, Any],
    keeps: list[tuple[float, float]],
    out: Path,
    encoder: str,
    compatibility: bool = False,
    audio_path: Optional[Path] = None,
    source_properties: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Build a continuous-CFR H.264/AAC export command."""
    properties = dict(proj.get("probe", {}))
    if source_properties:
        properties.update(
            {key: value for key, value in source_properties.items() if value is not None}
        )
    fps = max(1.0, float(properties.get("fps") or 25.0))
    fps_fraction = Fraction(fps).limit_denominator(1001)
    fps_expr = (
        str(fps_fraction.numerator)
        if fps_fraction.denominator == 1
        else f"{fps_fraction.numerator}/{fps_fraction.denominator}"
    )
    track_timescale = (
        fps_fraction.numerator * 1000
        if fps_fraction.denominator == 1
        else fps_fraction.numerator
    )
    source_offset = keeps[0][0]
    input_duration = keeps[-1][1] - source_offset
    shifted_keeps = [
        (start - source_offset, end - source_offset) for start, end in keeps
    ]
    selection = "+".join(
        f"between(t,{start:.6f},{end:.6f})" for start, end in shifted_keeps
    )
    has_audio = (
        audio_path is not None
        or properties.get("acodec") is not None
        or bool(properties.get("audio_streams"))
    )
    video_filters = [
        f"select='{selection}'",
        f"setpts=N/({fps_expr}*TB)",
        *_orientation_normalization_filters(properties),
        f"fps={fps_expr}",
    ]
    graph_parts = [
        f"[0:v]{','.join(video_filters)}[v]"
    ]
    maps = ["-map", "[v]"]
    if has_audio:
        if audio_path is not None:
            graph_parts.append(
                f"[1:a:0]asetpts=PTS-STARTPTS,"
                f"aresample={int(properties.get('sample_rate') or 48000)}:"
                "async=1:first_pts=0[a]"
            )
        else:
            audio_labels = []
            for index, (start, end) in enumerate(shifted_keeps):
                label = f"a{index}"
                graph_parts.append(
                    f"[0:a:0]atrim=start={start:.6f}:end={end:.6f},"
                    f"asetpts=PTS-STARTPTS[{label}]"
                )
                audio_labels.append(f"[{label}]")
            graph_parts.append(
                "".join(audio_labels)
                + f"concat=n={len(audio_labels)}:v=0:a=1,"
                "aresample=async=1:first_pts=0[a]"
            )
        maps.extend(["-map", "[a]"])

    if encoder == "h264_nvenc":
        quality = "23" if compatibility else "18"
        preset = "p4" if compatibility else "p6"
        video_args = [
            "-c:v", encoder, "-preset", preset, "-tune", "hq",
            "-rc", "vbr", "-cq", quality, "-b:v", "0",
            "-profile:v", "high", "-pix_fmt", "yuv420p",
        ]
    else:
        quality = "22" if compatibility else "18"
        preset = "veryfast" if compatibility else "fast"
        video_args = [
            "-c:v", "libx264", "-crf", quality, "-preset", preset,
            "-profile:v", "high", "-pix_fmt", "yuv420p",
        ]

    audio_args = (
        [
            "-c:a", "aac", "-b:a", "256k",
            "-ar", str(int(properties.get("sample_rate") or 48000)),
            "-ac", str(int(properties.get("channels") or 1)),
        ]
        if has_audio else ["-an"]
    )
    input_args = [
        "-ss", f"{source_offset:.6f}", "-t", f"{input_duration:.6f}",
        "-noautorotate", "-display_rotation", "0",
        "-i", proj["source_path"],
    ]
    if audio_path is not None:
        input_args.extend(["-i", str(audio_path)])
    return [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        *input_args,
        "-filter_complex", ";".join(graph_parts),
        *maps,
        "-map_metadata", "0", "-map_chapters", "-1",
        *video_args, *audio_args,
        "-r", fps_expr, "-fps_mode", "cfr",
        "-video_track_timescale", str(track_timescale),
        "-metadata:s:v:0", "rotate=0",
        "-max_muxing_queue_size", "4096",
        "-shortest", "-movflags", "+faststart",
        str(out),
    ]


def _reliable_video_export(
    proj: dict[str, Any],
    keeps: list[tuple[float, float]],
    out: Path,
    compatibility: bool = False,
    audio_path: Optional[Path] = None,
) -> str:
    """Export with GPU acceleration when usable, with one safe CPU fallback."""
    edited = sum(end - start for start, end in keeps)
    # Legacy project JSON may predate stored rotation metadata. The immutable
    # source is authoritative for every export so portrait phone media cannot
    # be flattened by stale project properties.
    source_properties = probe_media(proj["source_path"])
    encoders = (
        ["h264_nvenc", "libx264"]
        if _ffmpeg_encoder_usable("h264_nvenc")
        else ["libx264"]
    )
    first_error: Optional[Exception] = None
    for encoder in encoders:
        label = "NVIDIA GPU" if encoder == "h264_nvenc" else "CPU"
        try:
            command = _reliable_video_command(
                proj, keeps, out, encoder, compatibility=compatibility,
                audio_path=audio_path, source_properties=source_properties,
            )
            _run_ffmpeg_with_progress(
                command, edited, f"reliable {label} export"
            )
            return label
        except Exception as exc:
            if first_error is None:
                first_error = exc
            if encoder == "h264_nvenc" and "libx264" in encoders:
                log.warning("NVIDIA export failed; retrying on CPU: %s", exc)
                try:
                    out.unlink(missing_ok=True)
                except OSError:
                    pass
                continue
            raise
    raise RuntimeError(f"reliable export failed: {first_error}")


def validate_video_timeline(path: str, fps: float, expected_duration: float) -> list[str]:
    """Reject muxed video whose DTS cadence or packet durations are irregular."""
    if not FFPROBE:
        return ["ffprobe is required to verify reliable export timing"]
    result = _run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_packets", "-show_entries", "packet=dts_time,duration_time",
        "-of", "csv=p=0", path,
    ])
    if result.returncode != 0:
        return [f"could not inspect video timing: {result.stderr.strip()[:200]}"]

    frame_duration = 1.0 / max(1.0, fps)
    tolerance = max(0.002, frame_duration * 0.2)
    previous_dts: Optional[float] = None
    packet_count = non_monotonic = bad_spacing = bad_duration = 0
    first_dts = last_dts = None
    for line in result.stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) < 2 or "N/A" in fields[:2]:
            continue
        try:
            dts, duration = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        packet_count += 1
        if first_dts is None:
            first_dts = dts
        last_dts = dts
        if previous_dts is not None:
            delta = dts - previous_dts
            if delta <= 0:
                non_monotonic += 1
            elif abs(delta - frame_duration) > tolerance:
                bad_spacing += 1
        if abs(duration - frame_duration) > tolerance:
            bad_duration += 1
        previous_dts = dts

    errors: list[str] = []
    if packet_count == 0:
        return ["export contains no readable video packets"]
    if non_monotonic:
        errors.append(f"{non_monotonic} non-monotonic video timestamp(s)")
    if bad_spacing:
        errors.append(f"{bad_spacing} irregular frame interval(s)")
    if bad_duration:
        errors.append(f"{bad_duration} incorrect packet duration(s)")
    if first_dts is not None and last_dts is not None:
        timeline_duration = last_dts - first_dts + frame_duration
        if abs(timeline_duration - expected_duration) > max(0.25, 3 * frame_duration):
            errors.append(
                f"video timeline is {timeline_duration:.3f}s; "
                f"expected {expected_duration:.3f}s"
            )
    return errors


def reliable_export_integrity_errors(
    source_path: str, output_path: str, expected_duration: float,
) -> list[str]:
    """Property and packet-timing checks required before publishing an export."""
    source = probe_media(source_path)
    output = probe_media(output_path)
    fps = float(source.get("fps") or 25.0)
    errors: list[str] = []
    if abs(float(output["duration_s"]) - expected_duration) > max(0.25, 3.0 / fps):
        errors.append(
            f"duration {output['duration_s']:.3f}s vs expected {expected_duration:.3f}s"
        )
    if output.get("vcodec") != "h264":
        errors.append(f"video codec is {output.get('vcodec') or 'missing'}, expected h264")
    if output.get("video_streams") != 1:
        errors.append(f"video stream count is {output.get('video_streams')}, expected 1")
    expected_audio = 1 if source.get("audio_streams") else 0
    if output.get("audio_streams") != expected_audio:
        errors.append(
            f"audio stream count is {output.get('audio_streams')}, expected {expected_audio}"
        )
    if expected_audio and output.get("acodec") != "aac":
        errors.append(f"audio codec is {output.get('acodec') or 'missing'}, expected aac")
    if display_dimensions(source) != display_dimensions(output):
        errors.append(
            f"display dimensions changed from {display_dimensions(source)} "
            f"to {display_dimensions(output)}"
        )
    if abs(float(output.get("fps") or 0.0) - fps) > 0.01:
        errors.append(f"frame rate changed from {fps:g} to {output.get('fps')}")
    for key, label in (("sample_rate", "audio sample rate"), ("channels", "audio channels")):
        before, after = source.get(key), output.get(key)
        if expected_audio and before is not None and after is not None and before != after:
            errors.append(f"{label} changed from {before} to {after}")
    errors.extend(validate_video_timeline(output_path, fps, expected_duration))
    return errors


def export_job(mode: str, enhancement: Optional[dict[str, Any]] = None) -> None:
    """Reliable CFR export; legacy smart clients are routed to the same safe path."""
    out: Optional[Path] = None
    work_out: Optional[Path] = None
    audio_temp: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        proj = CURRENT["project"]
        assert proj is not None
        enhancement = validate_voice_enhancement(
            enhancement if enhancement is not None else proj.get("voice_enhancement")
        )
        cuts = cut_intervals_from_tokens(proj)
        keeps = keep_list(cuts, proj["duration_s"])
        if not keeps:
            raise RuntimeError("everything is cut - nothing to export")
        warnings: list[str] = []
        has_audio = proj.get("probe", {}).get("acodec") is not None
        if has_audio:
            set_status("export", 2, "snapping joins to real silence")
            try:
                level_profile = audio_energy_profile(proj["source_path"])
                silences = detect_silence_intervals(
                    proj["source_path"], float(proj["duration_s"]),
                    silence_db=float(level_profile["silence_db"]),
                )
                keeps = refine_export_keep_edges(
                    keeps, silences, float(proj["duration_s"])
                )
            except Exception as exc:
                log.warning("export silence snapping skipped: %s", exc)
                warnings.append("silence snapping was unavailable; original safe joins were used")
        edited = sum(b - a for a, b in keeps)
        audio_stem: Optional[Path] = None
        cleanup_status = "not_applicable"
        if has_audio:
            project_dir = project_directory_for_media(proj["source_path"])
            if project_dir is None:
                raise RuntimeError("project audio work folder is unavailable")
            audio_temp = tempfile.TemporaryDirectory(
                prefix=".retake-audio-", dir=str(project_dir)
            )
            audio_stem, enhancement_warnings, cleanup_status = prepare_export_audio(
                proj, keeps, enhancement, Path(audio_temp.name)
            )
            warnings.extend(enhancement_warnings)
        has_video = bool(proj["probe"].get("has_video"))
        if not has_video:
            set_status("export", 70, "encoding enhanced audio export")
            out = export_output_path(proj, "retake_cut")
            warnings.append("audio was fully re-encoded for sample-accurate edits")
            assert audio_stem is not None
            _encode_audio_stem(proj, audio_stem, out, edited)
            integrity_errors = audio_export_integrity_errors(
                proj["source_path"], str(out), edited
            )
            if integrity_errors:
                raise RuntimeError("integrity check failed: " + " | ".join(integrity_errors))
        else:
            compatibility = mode == "reencode"
            set_status("export", 3, "starting reliable video export")
            out = export_output_path(proj, "retake_cut", ".mp4")
            work_out = out.with_name(f"{out.stem}.partial{out.suffix}")
            encoder = _reliable_video_export(
                proj, keeps, work_out, compatibility=compatibility,
                audio_path=audio_stem,
            )
            set_status("export", 96, "verifying frame timing")
            integrity_errors = reliable_export_integrity_errors(
                proj["source_path"], str(work_out), edited
            )
            if integrity_errors:
                raise RuntimeError("integrity check failed: " + " | ".join(integrity_errors))
            os.replace(work_out, out)
            work_out = None
            if compatibility:
                warnings.append(
                    f"compatibility mode used the {encoder} encoder with a smaller-file setting"
                )

        warning = " | ".join(warnings) or None
        with _STATE_LOCK:
            LAST_EXPORT["path"] = str(out)
            LAST_EXPORT["warning"] = warning
            LAST_EXPORT["enhancement"] = cleanup_status
        set_status("ready", 100, f"exported: {out}")
        log.info("export ok: %s (%s)", out, warning or "verified smooth frame timing")
    except Exception as exc:
        cleanup = work_out if work_out is not None else out
        if cleanup is not None and cleanup.exists():
            try:
                cleanup.unlink()
            except OSError:
                log.warning("could not remove partial export: %s", cleanup)
        fail(f"reliable export failed: {exc}")
    finally:
        if audio_temp is not None:
            audio_temp.cleanup()
        JOB_LOCK.release()


def export_text(kind: str) -> Path:
    """EDL / CSV / TXT / SRT / markers.md — each a few lines, written to exports/."""
    proj = CURRENT["project"]
    assert proj is not None
    cuts = cut_intervals_from_tokens(proj)
    keeps = keep_list(cuts, proj["duration_s"])
    cut_seg_ids = fully_cut_segments(proj)

    if kind == "edl":
        out = export_output_path(proj, "retake", ".edl")
        fps = proj["probe"].get("fps", 25.0)
        lines = ["TITLE: RETAKE ROUGH CUT", "FCM: NON-DROP FRAME", ""]
        rec = 0.0
        for i, (a, b) in enumerate(keeps, 1):
            lines.append(
                f"{i:03d}  AX       V     C        "
                f"{_fmt_tc(a, fps)} {_fmt_tc(b, fps)} "
                f"{_fmt_tc(rec, fps)} {_fmt_tc(rec + (b - a), fps)}"
            )
            rec += b - a
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif kind == "csv":
        out = export_output_path(proj, "retake_keeps", ".csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["keep_start_s", "keep_end_s", "duration_s"])
            for a, b in keeps:
                w.writerow([f"{a:.3f}", f"{b:.3f}", f"{b - a:.3f}"])
    elif kind == "txt":
        out = export_output_path(proj, "transcript", ".txt")
        lines = []
        for seg in proj["segments"]:
            kept = [t["text"] for t in proj["tokens"]
                    if t["kind"] == "word" and t.get("seg") == seg["id"] and not t.get("cut")]
            if kept:
                lines.append(" ".join(kept))
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif kind == "srt":
        out = export_output_path(proj, "transcript", ".srt")
        blocks = []
        n = 1
        for seg in proj["segments"]:
            if seg["id"] in cut_seg_ids:
                continue
            blocks.append(
                f"{n}\n{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}\n{seg['text']}\n"
            )
            n += 1
        out.write_text("\n".join(blocks), encoding="utf-8")
    elif kind == "markers":
        project_dir = project_directory_for_media(proj["source_path"])
        if project_dir is None:
            raise RuntimeError("project export folder is unavailable")
        out = project_dir / "exports" / "markers.md"
        lines = ["# Markers", ""]
        for m in sorted(proj["markers"], key=lambda m: m["t"]):
            t = int(m["t"])
            lines.append(f"- [ ] {t // 60:02d}:{t % 60:02d} ({m['color']}) {m['note']}".rstrip())
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        raise ValueError(f"unknown export kind: {kind}")
    return out


def fully_cut_segments(proj: dict[str, Any]) -> set[int]:
    """Segment ids where every word token is cut."""
    total: dict[int, int] = {}
    cut: dict[int, int] = {}
    for t in proj["tokens"]:
        if t["kind"] != "word":
            continue
        sid = t["seg"]
        total[sid] = total.get(sid, 0) + 1
        if t.get("cut"):
            cut[sid] = cut.get(sid, 0) + 1
    return {sid for sid, n in total.items() if cut.get(sid, 0) == n}


# --------------------------------------------------------------------------
# HTTP API
# --------------------------------------------------------------------------

def _load_mcp_endpoint() -> tuple[Any, Any]:
    """The same MCP server retake_mcp.py serves over stdio, as an ASGI app.

    One tool surface, two transports. Optional: the editor runs unchanged when
    the `mcp` package is not installed, it just has no /mcp endpoint.
    """
    try:
        from retake_mcp import mcp as mcp_server
    except Exception as exc:  # missing dependency, or an import-time failure
        log.info("MCP endpoint disabled (%s); `pip install mcp` to enable %s",
                 exc, MCP_HTTP_PATH)
        return None, None
    try:
        # Mounted at MCP_HTTP_PATH, so the inner app owns the mount root.
        return mcp_server, mcp_server.streamable_http_app(streamable_http_path="/")
    except Exception as exc:
        log.warning("MCP endpoint could not be built: %s", exc)
        return None, None


MCP_SERVER, MCP_APP = _load_mcp_endpoint()


@asynccontextmanager
async def _lifespan(_app: "FastAPI") -> Any:
    """Run the MCP session manager alongside Retake, when it is available."""
    if MCP_SERVER is None:
        yield
        return
    async with MCP_SERVER.session_manager.run():
        log.info("MCP endpoint listening on %s", MCP_HTTP_PATH)
        yield


app = FastAPI(title="RETAKE", docs_url=None, redoc_url=None, lifespan=_lifespan)

if MCP_APP is not None:
    app.mount(MCP_HTTP_PATH, MCP_APP)

MIME_EXTRA = {".mkv": "video/x-matroska", ".m4a": "audio/mp4",
              ".webm": "video/webm", ".mov": "video/quicktime",
              ".flac": "audio/flac", ".opus": "audio/ogg"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        INDEX_HTML,
        media_type="text/html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/status")
def get_status() -> JSONResponse:
    with _STATE_LOCK:
        s = dict(STATUS)
        s["has_project"] = CURRENT["project"] is not None
        s["export"] = dict(LAST_EXPORT)
    s["preview"] = preview_status_payload()
    s["alignment"] = alignment_status_payload()
    return JSONResponse(s)


@app.get("/project")
def get_project() -> JSONResponse:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        payload = project_response_payload(proj) if proj is not None else None
    if proj is None:
        return JSONResponse({"error": "no project loaded"}, status_code=404)
    return JSONResponse(payload)


@app.get("/projects")
def list_projects() -> JSONResponse:
    rows = []
    for project_dir in project_directories():
        path = project_dir / "project.json"
        try:
            proj = json.loads(path.read_text(encoding="utf-8"))
            source = str(proj.get("source_path", ""))
            rows.append({
                "id": project_dir.name,
                "name": project_dir.name,
                "source_filename": proj.get("source_filename") or Path(source).name,
                "last_opened_at": proj.get("last_opened_at"),
                "available": bool(source and Path(source).is_file()),
            })
        except Exception as exc:
            rows.append({
                "id": project_dir.name, "name": project_dir.name,
                "source_filename": "Project is incomplete", "last_opened_at": None,
                "available": False, "error": str(exc),
            })
    rows.sort(key=lambda row: str(row.get("last_opened_at") or ""), reverse=True)
    return JSONResponse({"projects": rows})


@app.post("/projects/open")
def open_saved_project(payload: dict = Body(...)) -> JSONResponse:
    project_id = str(payload.get("id", ""))
    if not re.fullmatch(r"Project \d+", project_id):
        return JSONResponse({"error": "invalid project id"}, status_code=400)
    project_dir = PROJECTS_DIR / project_id
    path = project_dir / "project.json"
    if not path.is_file():
        return JSONResponse({"error": "project not found"}, status_code=404)
    try:
        proj = json.loads(path.read_text(encoding="utf-8"))
        migrated = ensure_voice_enhancement_state(proj)
        migrated = ensure_audio_gap_state(proj) or migrated
        source = Path(str(proj.get("source_path", "")))
        if not source.is_file():
            return JSONResponse(
                {"error": f"source media is unavailable: {source.name or 'unknown'}"},
                status_code=404,
            )
        proj["last_opened_at"] = utc_now()
        atomic_write_json(path, proj)
        with _STATE_LOCK:
            CURRENT["project"] = proj
            CURRENT["media_path"] = str(source)
            CURRENT["project_dir"] = str(project_dir)
            CURRENT["source_filename"] = proj.get("source_filename")
        set_status("ready", 100, "project loaded from disk")
        return JSONResponse({"loaded": "existing"})
    except Exception as exc:
        log.exception("saved project could not be opened")
        return JSONResponse({"error": f"project could not be opened: {exc}"}, status_code=500)


@app.post("/open")
def open_media(payload: dict = Body(...)) -> JSONResponse:
    """Load an existing project for this path, or start a transcription job."""
    path = str(payload.get("path", "")).strip().strip('"')
    requested_language = str(payload.get("language", "auto")).strip().lower()
    if requested_language == "auto":
        requested_language = ""
    if not path:
        return JSONResponse({"error": "no path given"}, status_code=400)
    p = Path(path).expanduser()
    source_filename = Path(str(payload.get("source_filename") or p.name)).name
    if not p.is_file():
        return JSONResponse({"error": f"file not found: {p}"}, status_code=404)

    project_dir = project_directory_for_media(str(p))
    if project_dir is None:
        return JSONResponse(
            {"error": "media must be transferred into a Retake project first"}, status_code=400
        )
    pj = project_dir / "project.json"
    if pj.exists():
        try:
            proj = json.loads(pj.read_text(encoding="utf-8"))
            if proj.get("schema_version") == 1:
                migrated = ensure_voice_enhancement_state(proj)
                migrated = ensure_audio_gap_state(proj) or migrated
                safe_clusters = sanitize_clusters(
                    proj.get("segments", []), proj.get("clusters", [])
                )
                if safe_clusters != proj.get("clusters", []):
                    dropped = len(proj.get("clusters", [])) - len(safe_clusters)
                    log.warning("discarded %d unsafe legacy retake cluster(s)", dropped)
                    proj["clusters"] = safe_clusters
                    migrated = True
                with _STATE_LOCK:
                    proj["last_opened_at"] = utc_now()
                    atomic_write_json(pj, proj)
                    CURRENT["project"] = proj
                    CURRENT["media_path"] = proj["source_path"]
                    CURRENT["project_dir"] = str(project_dir)
                    CURRENT["source_filename"] = proj.get("source_filename")
                set_status("ready", 100, "project loaded from disk")
                return JSONResponse({"loaded": "existing"})
        except Exception as e:
            log.warning("stale project json unreadable (%s); re-transcribing", e)

    if not JOB_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "another job is running"}, status_code=409)
    with _STATE_LOCK:
        CURRENT["project"] = None
        CURRENT["media_path"] = str(p.resolve())
        CURRENT["project_dir"] = str(project_dir)
        CURRENT["source_filename"] = source_filename or p.name
        AI_STATE.update(running=False, proposals=[], warnings=[], done=False)
    set_status("probe", 1, "starting")
    threading.Thread(
        target=transcribe_job, args=(str(p), requested_language or None), daemon=True
    ).start()
    return JSONResponse({"loaded": "transcribing"})


def upload_session_paths(session_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-f]{24}", session_id):
        raise ValueError("invalid upload session")
    transfer_dir = INCOMING_DIR / session_id
    return transfer_dir / "media.partial", transfer_dir / "transfer.json"


@app.post("/upload/start")
def start_upload(payload: dict = Body(...)) -> JSONResponse:
    name = Path(str(payload.get("name") or "media.bin")).name
    size = int(payload.get("size") or 0)
    fingerprint = str(payload.get("fingerprint") or "")[:200]
    if size <= 0:
        return JSONResponse({"error": "the selected file is empty"}, status_code=400)
    session_id = hashlib.sha256(f"{name}|{size}|{fingerprint}".encode()).hexdigest()[:24]
    partial, meta_path = upload_session_paths(session_id)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("name") == name and int(meta.get("size", -1)) == size:
                return JSONResponse({"session": session_id, "offset": partial.stat().st_size})
        except Exception:
            pass
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    partial.touch(exist_ok=True)
    atomic_write_json(meta_path, {
        "name": name, "size": size, "fingerprint": fingerprint,
        "created_at": utc_now(), "updated_at": utc_now(),
    })
    log.info("phone transfer started: %s (%d bytes) session=%s", name, size, session_id)
    return JSONResponse({"session": session_id, "offset": partial.stat().st_size})


@app.post("/upload/chunk/{session_id}")
async def upload_chunk(session_id: str, request: Request) -> JSONResponse:
    try:
        partial, meta_path = upload_session_paths(session_id)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        confirmed = partial.stat().st_size
        offset = int(request.headers.get("x-upload-offset", "-1"))
        if offset != confirmed:
            return JSONResponse(
                {"error": "upload offset changed", "offset": confirmed}, status_code=409
            )
        data = bytearray()
        async for piece in request.stream():
            data.extend(piece)
            if len(data) > UPLOAD_CHUNK_MAX_BYTES:
                return JSONResponse({"error": "upload chunk is too large"}, status_code=413)
        if confirmed + len(data) > int(meta["size"]):
            return JSONResponse({"error": "upload exceeds declared file size"}, status_code=400)
        with open(partial, "ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        meta["updated_at"] = utc_now()
        atomic_write_json(meta_path, meta)
        log.info("phone transfer progress: session=%s offset=%d/%d",
                 session_id, confirmed + len(data), int(meta["size"]))
        return JSONResponse({"offset": confirmed + len(data)})
    except FileNotFoundError:
        return JSONResponse({"error": "upload session expired"}, status_code=404)
    except Exception as exc:
        log.exception("upload chunk failed")
        return JSONResponse({"error": f"upload failed: {exc}"}, status_code=500)


@app.post("/upload/finish")
def finish_upload(payload: dict = Body(...)) -> JSONResponse:
    try:
        session_id = str(payload.get("session", ""))
        partial, meta_path = upload_session_paths(session_id)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        received = partial.stat().st_size
        if received != int(meta["size"]):
            return JSONResponse(
                {"error": "upload is incomplete", "offset": received}, status_code=409
            )
        name = Path(meta["name"]).name
        project_dir = next_project_directory()
        extension = Path(name).suffix
        dest = project_dir / "media" / f"original{extension}"
        os.replace(partial, dest)
        shutil.rmtree(meta_path.parent)
        log.info("phone transfer finalized: session=%s -> %s", session_id, dest)
        return JSONResponse({"path": str(dest), "project": project_dir.name, "source_filename": name})
    except FileNotFoundError:
        return JSONResponse({"error": "upload session expired"}, status_code=404)
    except Exception as exc:
        return JSONResponse({"error": f"upload could not be completed: {exc}"}, status_code=500)


@app.post("/upload/cancel")
def cancel_upload(payload: dict = Body(...)) -> JSONResponse:
    try:
        partial, meta_path = upload_session_paths(str(payload.get("session", "")))
        transfer_dir = meta_path.parent.resolve()
        transfer_dir.relative_to(INCOMING_DIR.resolve())
        if transfer_dir.exists():
            shutil.rmtree(transfer_dir)
        log.info("phone transfer cancelled: %s", transfer_dir.name)
        return JSONResponse({"ok": True})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.api_route("/media", methods=["GET", "HEAD"])
def media(request: Request) -> Response:
    """Serve original media with Starlette's disconnect-aware HTTP range support."""
    with _STATE_LOCK:
        path = CURRENT["media_path"]
    if not path or not Path(path).is_file():
        return JSONResponse({"error": "no media"}, status_code=404)
    ext = Path(path).suffix.lower()
    mime = MIME_EXTRA.get(ext) or mimetypes.guess_type(path)[0] or "application/octet-stream"
    return FileResponse(
        path, media_type=mime, headers={"Cache-Control": "no-cache"},
        stat_result=os.stat(path),
    )


@app.get("/preview/status")
def get_preview_status() -> JSONResponse:
    return JSONResponse(preview_status_payload())


@app.get("/alignment/status")
def get_alignment_status() -> JSONResponse:
    return JSONResponse(alignment_status_payload())


@app.post("/alignment/recalibrate")
def start_alignment_recalibration() -> JSONResponse:
    with _STATE_LOCK:
        current = CURRENT["project"]
        proj = json.loads(json.dumps(current)) if current is not None else None
    if proj is None:
        return JSONResponse({"error": "Open a project first."}, status_code=404)
    if not proj.get("probe", {}).get("audio_streams") and not proj.get(
        "probe", {}
    ).get("acodec"):
        return JSONResponse(
            {"error": "Accurate timing requires an audio stream."}, status_code=400
        )
    if alignment_python_path() is None:
        return JSONResponse(
            {"error": "Accurate timing runtime is not installed."}, status_code=409
        )
    try:
        identity = alignment_source_identity(proj)
    except (KeyError, FileNotFoundError, OSError):
        return JSONResponse({"error": "Source media is unavailable."}, status_code=404)
    status = alignment_status_payload()
    if status.get("ready"):
        return JSONResponse({"ok": True, "ready": True})
    if not JOB_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "another job is running"}, status_code=409)
    with _STATE_LOCK:
        ALIGNMENT_STATE.update(
            running=True, identity=identity, error=None,
            device="cuda", fallback=False, coverage=None,
        )
    set_status("align", 1, "starting GPU timing calibration")
    threading.Thread(
        target=recalibrate_alignment_job, args=(proj, identity), daemon=True
    ).start()
    return JSONResponse({"ok": True, "ready": False})


@app.post("/preview/create")
def create_preview_proxy() -> JSONResponse:
    with _STATE_LOCK:
        current = CURRENT["project"]
        proj = dict(current) if current is not None else None
    if proj is None:
        return JSONResponse({"error": "Open a project first."}, status_code=404)
    if not proj.get("probe", {}).get("has_video"):
        return JSONResponse(
            {"error": "Smooth Preview requires video media."}, status_code=400
        )
    try:
        identity = preview_source_identity(proj["source_path"])
    except (FileNotFoundError, OSError):
        return JSONResponse({"error": "Source media is unavailable."}, status_code=404)
    if _preview_proxy_record(proj["source_path"]) is not None:
        return JSONResponse({"ok": True, "ready": True})
    if not JOB_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "another job is running"}, status_code=409)
    gpu_usable = preview_gpu_usable()
    with _STATE_LOCK:
        PREVIEW_STATE.update(
            running=True, identity=identity, error=None,
            device="gpu" if gpu_usable else "cpu", fallback=not gpu_usable,
        )
    set_status(
        "preview", 1,
        (
            "starting GPU smooth preview"
            if gpu_usable
            else "starting smooth preview on CPU fallback"
        ),
    )
    threading.Thread(
        target=preview_proxy_job, args=(proj, identity), daemon=True
    ).start()
    return JSONResponse({"ok": True, "ready": False})


@app.api_route("/preview/media", methods=["GET", "HEAD"])
def preview_media(request: Request) -> Response:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        source_path = str(proj.get("source_path", "")) if proj is not None else ""
    if not source_path:
        return JSONResponse({"error": "Open a project first."}, status_code=404)
    record = _preview_proxy_record(source_path)
    if record is None:
        return JSONResponse({"error": "Smooth Preview is not ready."}, status_code=404)
    path = record["path"]
    return FileResponse(
        path, media_type="video/mp4", headers={"Cache-Control": "no-cache"},
        stat_result=os.stat(path),
    )


@app.post("/cuts")
def post_cuts(payload: dict = Body(...)) -> JSONResponse:
    """Debounced autosave from the UI: full set of cut token ids + gap threshold."""
    try:
        cut_ids = set(int(i) for i in payload.get("cut_ids", []))
    except (TypeError, ValueError):
        return JSONResponse({"error": "cut_ids must contain token IDs"}, status_code=400)
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        for t in proj["tokens"]:
            t["cut"] = t["id"] in cut_ids
        gt = payload.get("gap_threshold_s")
        if isinstance(gt, (int, float)) and 0.05 <= gt <= 30:
            proj["gap_threshold_s"] = float(gt)
        params = cut_composition_params(proj)
        response = {
            "ok": True,
            "word_cut_intervals": consecutive_word_cut_intervals(
                proj["tokens"], **params
            ),
            "transcript_cut_intervals": transcript_cut_intervals(
                proj["tokens"], **params
            ),
            "cut_intervals": cut_intervals_from_tokens(proj),
            "scoped_gap_candidate_ids": scoped_gap_candidate_ids(proj),
        }
    save_current_project()
    return JSONResponse(response)


@app.post("/gaps/detect")
def start_gap_detection(payload: dict = Body(default={})) -> JSONResponse:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        if proj.get("probe", {}).get("acodec") is None:
            return JSONResponse({"error": "this media has no audio track"}, status_code=400)
        ensure_audio_gap_state(proj)
        base = dict(proj.get("audio_gap_settings") or {})
    base.update(payload.get("settings") or {})
    try:
        settings = validate_audio_gap_settings(base)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if not JOB_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "another job is running"}, status_code=409)
    threading.Thread(
        target=audio_gap_detection_job, args=(settings,), daemon=True
    ).start()
    return JSONResponse({"ok": True})


@app.post("/gaps")
def save_audio_gaps(payload: dict = Body(...)) -> JSONResponse:
    try:
        with _STATE_LOCK:
            proj = CURRENT["project"]
            if proj is None:
                return JSONResponse({"error": "no project"}, status_code=404)
            ensure_audio_gap_state(proj)
            settings = validate_audio_gap_settings(
                payload.get("settings", proj["audio_gap_settings"])
            )
            gaps = validate_audio_gaps(
                payload.get("gaps", proj["audio_gaps"]),
                proj["audio_gaps"], float(proj["duration_s"]),
            )
            proj["audio_gap_settings"] = settings
            proj["audio_gaps"] = gaps
        save_current_project()
        return JSONResponse({"ok": True, "gaps": gaps, "settings": settings})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/waveform")
def get_waveform(start: float, end: float, points: int = 180) -> JSONResponse:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        media_path = str(proj["source_path"])
        duration = float(proj["duration_s"])
    start = max(0.0, min(duration, float(start)))
    end = max(start, min(duration, float(end)))
    points = max(40, min(500, int(points)))
    if end <= start or end - start > 120:
        return JSONResponse(
            {"error": "waveform range must be between 0 and 120 seconds"},
            status_code=400,
        )
    try:
        peaks = audio_waveform_peaks(media_path, start, end, points)
        return JSONResponse({"start": start, "end": end, "peaks": peaks})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/markers")
def post_markers(payload: dict = Body(...)) -> JSONResponse:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        markers = []
        for m in payload.get("markers", []):
            try:
                markers.append({"t": float(m["t"]),
                                "color": str(m.get("color", "coral"))[:24],
                                "note": str(m.get("note", ""))[:500]})
            except (KeyError, TypeError, ValueError):
                continue
        proj["markers"] = markers
    save_current_project()
    return JSONResponse({"ok": True})


@app.get("/assistant/available")
def assistant_available() -> JSONResponse:
    """The assistant is always available: nothing needs to be installed for it."""
    return JSONResponse({
        "available": True,
        "engine": "deterministic",
        "mcp": {"http_path": MCP_HTTP_PATH, "stdio_module": "retake_mcp"},
    })


@app.post("/assistant/run")
def assistant_run(payload: dict = Body(...)) -> JSONResponse:
    if CURRENT["project"] is None:
        return JSONResponse({"error": "no project"}, status_code=404)
    try:
        attachments, attachment_warnings = validate_ai_attachments(payload.get("attachments", []))
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    instructions = str(payload.get("instructions", ""))
    if len(instructions) > AI_INSTRUCTION_MAX_CHARS:
        return JSONResponse(
            {"error": f"instructions must be at most {AI_INSTRUCTION_MAX_CHARS:,} characters"},
            status_code=400,
        )
    raw_operations = payload.get("operations")
    if raw_operations is not None and not isinstance(raw_operations, list):
        return JSONResponse({"error": "operations must be a list"}, status_code=400)
    operations = [op for op in (raw_operations or []) if isinstance(op, dict)]
    if len(operations) > ASSISTANT_MAX_OPERATIONS:
        return JSONResponse(
            {"error": f"at most {ASSISTANT_MAX_OPERATIONS} operations per run"},
            status_code=400,
        )
    if not JOB_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "another job is running"}, status_code=409)
    with _STATE_LOCK:
        AI_STATE.update(running=True, proposals=[], warnings=[], done=False, mode=None)
    threading.Thread(
        target=assistant_job,
        args=(instructions, attachments, attachment_warnings, operations),
        daemon=True,
    ).start()
    return JSONResponse({"ok": True, "attachments": len(attachments),
                         "warnings": attachment_warnings})


@app.get("/assistant/result")
def assistant_result() -> JSONResponse:
    with _STATE_LOCK:
        return JSONResponse(dict(AI_STATE))


# --------------------------------------------------------------------------
# Assistant surface
#
# Everything an MCP client needs, expressed so that the client supplies
# language and Retake supplies the token selection. Mutating routes take
# dry_run and back the project up before they write, so a confident mistake is
# always previewable and always reversible.
# --------------------------------------------------------------------------

def assistant_backups(project_dir: Path) -> list[Path]:
    """Newest first."""
    return sorted(project_dir.glob("project.assistant-backup-*.json"), reverse=True)


_ASSISTANT_BACKUP_SEQ = itertools.count()


def write_assistant_backup(
    snapshot: dict[str, Any], project_dir: Path,
) -> Optional[str]:
    """Persist a pre-edit snapshot, then prune to ASSISTANT_BACKUP_KEEP.

    The caller passes the state as it was *before* its edit; taking the
    snapshot here would capture the mutation the backup exists to undo. The
    counter keeps names ordered when two edits land in the same second, since
    undo restores by filename order.
    """
    stamp = "".join(ch for ch in utc_now() if ch.isalnum())
    sequence = next(_ASSISTANT_BACKUP_SEQ)
    backup = project_dir / f"project.assistant-backup-{stamp}-{sequence:06d}.json"
    atomic_write_json(backup, snapshot)
    for stale in assistant_backups(project_dir)[ASSISTANT_BACKUP_KEEP:]:
        try:
            stale.unlink()
        except OSError:
            pass
    return backup.name


def assistant_edit_summary(proj: dict[str, Any]) -> dict[str, Any]:
    """What the current edit actually produces, without rendering anything."""
    duration = float(proj.get("duration_s") or 0.0)
    cuts = cut_intervals_from_tokens(proj)
    keeps = keep_list(cuts, duration)
    removed = sum(end - start for start, end in cuts)
    words = [t for t in proj.get("tokens", []) if t.get("kind") == "word"]
    return {
        "duration_s": round(duration, 3),
        "edited_duration_s": round(max(0.0, duration - removed), 3),
        "removed_s": round(removed, 3),
        "removed_fraction": round(removed / duration, 4) if duration > 0 else 0.0,
        "cut_intervals": [[round(a, 3), round(b, 3)] for a, b in cuts],
        "keep_intervals": [[round(a, 3), round(b, 3)] for a, b in keeps],
        "word_count": len(words),
        "cut_word_count": sum(1 for t in words if t.get("cut")),
        "segment_count": len(proj.get("segments", [])),
        "fully_cut_segments": sorted(fully_cut_segments(proj)),
        "cut_composition": cut_composition_params(proj),
    }


@app.get("/assistant/summary")
def get_assistant_summary() -> JSONResponse:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        project_dir = CURRENT.get("project_dir")
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        summary = assistant_edit_summary(proj)
        summary["language"] = proj.get("language")
        summary["aligned"] = project_uses_aligned_timing(proj)
        summary["source_filename"] = CURRENT.get("source_filename")
    summary["project"] = Path(project_dir).name if project_dir else None
    summary["undo_available"] = bool(project_dir and assistant_backups(Path(project_dir)))
    return JSONResponse(summary)


@app.get("/assistant/transcript")
def get_assistant_transcript(
    start: float = 0.0, end: float = -1.0, offset: int = 0,
    limit: int = ASSISTANT_TRANSCRIPT_PAGE, only: str = "all",
) -> JSONResponse:
    """Segments with their word token IDs, paginated because recordings are long."""
    if only not in {"all", "kept", "cut"}:
        return JSONResponse({"error": "only must be all, kept or cut"}, status_code=400)
    limit = max(1, min(int(limit), ASSISTANT_TRANSCRIPT_PAGE))
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        duration = float(proj.get("duration_s") or 0.0)
        upper = duration if end is None or end < 0 else float(end)
        words_by_segment: dict[Any, list[dict[str, Any]]] = {}
        for token in proj.get("tokens", []):
            if token.get("kind") != "word":
                continue
            words_by_segment.setdefault(token.get("seg"), []).append(token)

        rows: list[dict[str, Any]] = []
        for segment in proj.get("segments", []):
            interval = _timed_interval(segment)
            if interval is None or interval[1] < float(start) or interval[0] > upper:
                continue
            words = words_by_segment.get(segment.get("id"), [])
            cut_words = sum(1 for w in words if w.get("cut"))
            if only == "kept" and words and cut_words == len(words):
                continue
            if only == "cut" and cut_words == 0:
                continue
            rows.append({
                "segment_id": segment.get("id"),
                "start": round(interval[0], 3),
                "end": round(interval[1], 3),
                "text": segment.get("text", ""),
                "cut_words": cut_words,
                "word_count": len(words),
                "fully_cut": bool(words) and cut_words == len(words),
                "words": [
                    {"id": int(w["id"]), "text": w.get("text", ""),
                     "start": round(float(w["start"]), 3),
                     "end": round(float(w["end"]), 3), "cut": bool(w.get("cut"))}
                    for w in words
                ],
            })
    offset = max(0, int(offset))
    page = rows[offset:offset + limit]
    return JSONResponse({
        "segments": page, "offset": offset, "limit": limit,
        "returned": len(page), "total": len(rows),
        "next_offset": offset + len(page) if offset + len(page) < len(rows) else None,
        "duration_s": round(duration, 3),
    })


@app.post("/assistant/find")
def assistant_find(payload: dict = Body(...)) -> JSONResponse:
    """Locate a spoken phrase and report the exact token span, or why not."""
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        duration = float(proj.get("duration_s") or 0.0)
        phrase = str(payload.get("phrase", "") or payload.get("query", ""))
        if not phrase.strip():
            return JSONResponse({"error": "phrase is required"}, status_code=400)
        try:
            window_start = float(payload.get("start", 0.0) or 0.0)
            raw_end = payload.get("end")
            window_end = duration if raw_end is None else float(raw_end)
        except (TypeError, ValueError):
            return JSONResponse({"error": "start and end must be numbers"}, status_code=400)
        occurrence = payload.get("occurrence")
        if occurrence not in (None, "first", "last"):
            return JSONResponse({"error": "occurrence must be first or last"}, status_code=400)
        result = resolve_phrase_tokens(
            proj.get("tokens", []), phrase, window_start, window_end,
            all_matches=bool(payload.get("all_matches")), occurrence=occurrence,
        )
        token_by_id = {
            int(t["id"]): t for t in proj.get("tokens", []) if t.get("kind") == "word"
        }
    matched = [token_by_id[i] for i in result.get("token_ids", []) if i in token_by_id]
    if matched:
        result["start"] = round(min(float(t["start"]) for t in matched), 3)
        result["end"] = round(max(float(t["end"]) for t in matched), 3)
        result["text"] = " ".join(str(t.get("text", "")) for t in matched)
        result["already_cut"] = all(bool(t.get("cut")) for t in matched)
    return JSONResponse(result)


@app.get("/assistant/retakes")
def get_assistant_retakes() -> JSONResponse:
    """Repeated takes the app already grouped, with their token IDs."""
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        segments = proj.get("segments", [])
        by_index = dict(enumerate(segments))
        already_cut = fully_cut_segments(proj)
        words_by_segment: dict[Any, list[int]] = {}
        for token in proj.get("tokens", []):
            if token.get("kind") == "word":
                words_by_segment.setdefault(token.get("seg"), []).append(int(token["id"]))
        groups = []
        for cluster in sanitize_clusters(segments, proj.get("clusters", [])):
            members = []
            for index in cluster.get("members", []):
                segment = by_index.get(index)
                if segment is None:
                    continue
                members.append({
                    "segment_id": segment.get("id"),
                    "start": round(float(segment.get("start", 0.0)), 3),
                    "end": round(float(segment.get("end", 0.0)), 3),
                    "text": segment.get("text", ""),
                    "token_ids": words_by_segment.get(segment.get("id"), []),
                    "already_cut": segment.get("id") in already_cut,
                })
            if members:
                groups.append({"cluster_id": cluster.get("id"), "takes": members})
    return JSONResponse({"retakes": groups, "count": len(groups)})


def apply_assistant_token_edit(
    token_ids: list[int], mode: str, dry_run: bool,
) -> tuple[dict[str, Any], int]:
    """Cut/keep/toggle explicit word tokens, previewing the effect either way.

    This is a delta, unlike /cuts which replaces the whole selection. A client
    that only knows about the words it just searched for must not be able to
    silently restore everything else.
    """
    with _STATE_LOCK:
        proj = CURRENT["project"]
        raw_dir = CURRENT.get("project_dir")
        if proj is None:
            return {"error": "no project"}, 404
        by_id = {int(t["id"]): t for t in proj.get("tokens", []) if t.get("kind") == "word"}
        unknown = [i for i in token_ids if i not in by_id]
        if unknown:
            return {"error": "unknown token ids", "unknown_token_ids": unknown[:20]}, 400
        before = assistant_edit_summary(proj)
        changing: list[int] = []
        for token_id in token_ids:
            current = bool(by_id[token_id].get("cut"))
            target = True if mode == "cut" else False if mode == "keep" else not current
            if target != current:
                changing.append(token_id)
        # Captured before the mutation so undo has somewhere to go back to.
        snapshot = copy.deepcopy(proj) if not dry_run and changing else None
        for token_id in changing:
            by_id[token_id]["cut"] = not bool(by_id[token_id].get("cut"))
        after = assistant_edit_summary(proj)
        if dry_run:
            for token_id in changing:
                by_id[token_id]["cut"] = not bool(by_id[token_id].get("cut"))

    keys = ("edited_duration_s", "removed_s", "cut_word_count")
    response: dict[str, Any] = {
        "ok": True, "dry_run": dry_run, "mode": mode,
        "changed_token_ids": changing, "changed": len(changing),
        "before": {k: before[k] for k in keys},
        "after": {k: after[k] for k in keys},
    }
    if dry_run:
        response["note"] = "nothing was written; call again with dry_run=false to apply"
        return response, 200
    if snapshot is not None and raw_dir:
        response["backup"] = write_assistant_backup(snapshot, Path(raw_dir))
    save_current_project()
    return response, 200


@app.post("/assistant/cuts")
def assistant_cuts(payload: dict = Body(...)) -> JSONResponse:
    mode = str(payload.get("mode", "cut"))
    if mode not in {"cut", "keep", "toggle"}:
        return JSONResponse({"error": "mode must be cut, keep or toggle"}, status_code=400)
    raw_ids = payload.get("token_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        return JSONResponse({"error": "token_ids must be a non-empty list"}, status_code=400)
    try:
        token_ids = [int(value) for value in raw_ids]
    except (TypeError, ValueError):
        return JSONResponse({"error": "token_ids must be integers"}, status_code=400)
    if len(token_ids) > ASSISTANT_MAX_TOKEN_IDS:
        return JSONResponse(
            {"error": f"at most {ASSISTANT_MAX_TOKEN_IDS} token ids per call"},
            status_code=400,
        )
    body, status = apply_assistant_token_edit(
        token_ids, mode, payload.get("dry_run", True) is not False
    )
    return JSONResponse(body, status_code=status)


@app.post("/assistant/apply-proposals")
def assistant_apply_proposals(payload: dict = Body(default={})) -> JSONResponse:
    """Accept staged proposals from the last assistant run."""
    dry_run = payload.get("dry_run", True) is not False
    with _STATE_LOCK:
        proposals = list(AI_STATE.get("proposals") or [])
    if not proposals:
        return JSONResponse({"error": "no proposals are staged"}, status_code=404)
    wanted = payload.get("indexes")
    if wanted is None:
        selected = list(range(len(proposals)))
    elif isinstance(wanted, list):
        try:
            selected = [int(value) for value in wanted]
        except (TypeError, ValueError):
            return JSONResponse({"error": "indexes must be integers"}, status_code=400)
        invalid = [i for i in selected if not 0 <= i < len(proposals)]
        if invalid:
            return JSONResponse({"error": "index out of range", "invalid": invalid}, status_code=400)
    else:
        return JSONResponse({"error": "indexes must be a list"}, status_code=400)

    token_ids: list[int] = []
    skipped: list[dict[str, Any]] = []
    for index in selected:
        proposal = proposals[index]
        cuttable = (
            proposal.get("action") == "cut_phrase"
            and proposal.get("status") in {"exact", "approximate"}
        )
        if not cuttable:
            skipped.append({
                "index": index,
                "reason": proposal.get("status") or proposal.get("action") or "not cuttable",
            })
            continue
        token_ids.extend(int(value) for value in proposal.get("token_ids", []))
    token_ids = sorted(set(token_ids))
    if not token_ids:
        return JSONResponse(
            {"error": "none of the selected proposals resolve to cuttable words",
             "skipped": skipped},
            status_code=400,
        )
    body, status = apply_assistant_token_edit(token_ids, "cut", dry_run)
    if status == 200:
        skipped_indexes = {item["index"] for item in skipped}
        body["applied_proposals"] = [i for i in selected if i not in skipped_indexes]
        body["skipped_proposals"] = skipped
    return JSONResponse(body, status_code=status)


@app.post("/assistant/undo")
def assistant_undo(payload: dict = Body(default={})) -> JSONResponse:
    """Restore the newest assistant backup and consume it."""
    with _STATE_LOCK:
        raw_dir = CURRENT.get("project_dir")
        if CURRENT["project"] is None or not raw_dir:
            return JSONResponse({"error": "no project"}, status_code=404)
        project_dir = Path(raw_dir)
        backups = assistant_backups(project_dir)
        if not backups:
            return JSONResponse({"error": "nothing to undo"}, status_code=404)
        newest = backups[0]
        try:
            restored = json.loads(newest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return JSONResponse({"error": f"backup unreadable: {exc}"}, status_code=500)
        if not isinstance(restored, dict) or "tokens" not in restored:
            return JSONResponse({"error": "backup is not a project"}, status_code=500)
        CURRENT["project"] = restored
        summary = assistant_edit_summary(restored)
    save_current_project()
    try:
        newest.unlink()
    except OSError:
        pass
    return JSONResponse({
        "ok": True, "restored_from": newest.name,
        "remaining_undo_steps": len(assistant_backups(project_dir)),
        "edited_duration_s": summary["edited_duration_s"],
        "cut_word_count": summary["cut_word_count"],
    })


@app.post("/voice-enhancement")
def save_voice_enhancement(payload: dict = Body(...)) -> JSONResponse:
    try:
        settings = validate_voice_enhancement(payload.get("settings", payload))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        proj["voice_enhancement"] = settings
    save_current_project()
    return JSONResponse({"ok": True, "settings": settings})


@app.post("/export")
def run_export(payload: dict = Body(...)) -> JSONResponse:
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            return JSONResponse({"error": "no project"}, status_code=404)
        current_settings = proj.get("voice_enhancement")
    try:
        settings = validate_voice_enhancement(
            payload.get("voice_enhancement", current_settings)
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    mode = str(payload.get("mode", "smart"))
    if not JOB_LOCK.acquire(blocking=False):
        return JSONResponse({"error": "another job is running"}, status_code=409)
    with _STATE_LOCK:
        proj = CURRENT["project"]
        if proj is None:
            JOB_LOCK.release()
            return JSONResponse({"error": "no project"}, status_code=404)
        proj["voice_enhancement"] = settings
        LAST_EXPORT.update(path=None, warning=None, enhancement=None)
    save_current_project()
    threading.Thread(target=export_job, args=(mode, dict(settings)), daemon=True).start()
    return JSONResponse({"ok": True})


@app.get("/export/latest")
def get_latest_export() -> JSONResponse:
    if CURRENT["project"] is None:
        return JSONResponse({"error": "Open a project first."}, status_code=404)
    out = latest_media_export()
    if out is None:
        return JSONResponse(
            {"error": "No media export found. Please export first."},
            status_code=404,
        )
    try:
        size = out.stat().st_size
    except OSError:
        return JSONResponse(
            {"error": "No media export found. Please export first."},
            status_code=404,
        )
    return JSONResponse({"name": out.name, "size": size})


@app.get("/export/latest/download")
def download_latest_export() -> Response:
    out = latest_media_export()
    if out is None:
        return JSONResponse(
            {"error": "No media export found. Please export first."},
            status_code=404,
        )
    mime = MIME_EXTRA.get(out.suffix.lower()) or mimetypes.guess_type(out.name)[0]
    return FileResponse(
        out,
        media_type=mime or "application/octet-stream",
        filename=out.name,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.post("/export_text")
def run_export_text(payload: dict = Body(...)) -> JSONResponse:
    if CURRENT["project"] is None:
        return JSONResponse({"error": "no project"}, status_code=404)
    try:
        out = export_text(str(payload.get("kind", "")))
        return JSONResponse({"path": str(out)})
    except Exception as e:
        log.exception("text export failed")
        return JSONResponse({"error": f"export failed: {e}"}, status_code=500)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def lan_addresses() -> list[str]:
    """Private IPv4 addresses this computer answers on."""
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            ip = ipaddress.ip_address(address)
            if ip.is_private and not ip.is_loopback and not ip.is_link_local:
                addresses.add(address)
    except OSError:
        pass
    return sorted(addresses)


def lan_urls(port: int = PORT, scheme: str = "http") -> list[str]:
    return [f"{scheme}://{address}:{port}" for address in lan_addresses()]


def ensure_lan_certificate() -> Optional[tuple[Path, Path]]:
    """Self-signed certificate covering localhost and this machine's LAN IPs.

    Phones are the reason this exists. Over plain HTTP, iOS Safari refuses
    `navigator.wakeLock`, so the screen sleeps partway through a large transfer,
    the tab is suspended, and the upload stalls. HTTPS -- even a certificate the
    phone has to be told to trust once -- makes the wake lock available and the
    transfer runs to completion.

    Regenerated when the set of addresses changes or the certificate is close to
    expiring. It never leaves this machine and lives under the ignored models/
    folder; the private key is written before the certificate so a half-written
    pair is detected and rebuilt rather than served.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        log.info("HTTPS disabled: `pip install cryptography` to enable phone transfers "
                 "that survive a screen lock")
        return None

    addresses = lan_addresses()
    fingerprint = json.dumps({"hosts": addresses, "version": TLS_CERT_VERSION}, sort_keys=True)
    TLS_DIR.mkdir(parents=True, exist_ok=True)
    cert_path, key_path, meta_path = (
        TLS_DIR / "retake.crt", TLS_DIR / "retake.key", TLS_DIR / "retake.json"
    )

    if cert_path.exists() and key_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expires = datetime.fromisoformat(meta["expires_at"])
            fresh = expires - datetime.now(timezone.utc) > timedelta(days=14)
            if meta.get("fingerprint") == fingerprint and fresh:
                return cert_path, key_path
        except (OSError, ValueError, KeyError):
            pass

    try:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "Retake local"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Retake"),
        ])
        alternatives: list[x509.GeneralName] = [
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        ]
        for address in addresses:
            alternatives.append(x509.IPAddress(ipaddress.ip_address(address)))
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=TLS_CERT_DAYS)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(expires_at)
            .add_extension(x509.SubjectAlternativeName(alternatives), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        )
        key_path.write_bytes(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ))
        cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        atomic_write_json(meta_path, {
            "fingerprint": fingerprint,
            "expires_at": expires_at.isoformat(),
            "hosts": addresses,
        })
    except Exception as exc:
        log.warning("HTTPS certificate could not be created: %s", exc)
        return None
    log.info("HTTPS certificate ready for %s", ", ".join(addresses) or "localhost only")
    return cert_path, key_path


def main() -> None:
    resolve_ffmpeg()
    if not INDEX_HTML.exists():
        print("FATAL: index.html not found next to retake.py", file=sys.stderr)
        sys.exit(1)
    threading.Timer(1.2, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    local_url = f"http://localhost:{PORT}"
    tls = ensure_lan_certificate()
    https_urls = lan_urls(HTTPS_PORT, "https") if tls else []
    http_urls = lan_urls()

    print("\nRETAKE is ready")
    print(f"  This computer:       {local_url}")
    if https_urls:
        for index, url in enumerate(https_urls):
            label = "Phone / local network:" if index == 0 else "                      "
            print(f"  {label} {url}")
        print()
        print("  The phone will warn that the certificate is not trusted the first")
        print("  time. Choose Advanced, then continue -- it is this computer's own")
        print("  certificate. HTTPS is what lets the phone keep its screen awake, so")
        print("  a large transfer no longer stalls when the screen locks.")
        if http_urls:
            print(f"  Plain HTTP still works at {http_urls[0]} if you prefer.")
    elif http_urls:
        for index, url in enumerate(http_urls):
            label = "Phone / local network:" if index == 0 else "                      "
            print(f"  {label} {url}")
        print("  (No HTTPS: transfers will pause when the phone screen locks.)")
    else:
        print("  Phone / local network: no active private network address found")
    print()
    log.info("RETAKE listening on %s", local_url)

    if os.name == "nt" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if not tls:
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
        return

    # Both at once: the desktop keeps the plain-HTTP address it has always used,
    # and phones get the HTTPS one they need for a wake lock.
    certificate, key = tls
    plain = uvicorn.Server(uvicorn.Config(
        app, host="0.0.0.0", port=PORT, log_level="warning",
    ))
    threading.Thread(target=plain.run, daemon=True).start()
    try:
        uvicorn.run(
            app, host="0.0.0.0", port=HTTPS_PORT, log_level="warning",
            ssl_certfile=str(certificate), ssl_keyfile=str(key),
        )
    except OSError as exc:
        log.warning("HTTPS could not start on port %d (%s); serving HTTP only",
                    HTTPS_PORT, exc)
        plain.should_exit = True
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
