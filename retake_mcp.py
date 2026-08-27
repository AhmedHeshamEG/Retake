"""Retake as an MCP server.

Retake edits a transcript, not a timeline: every cut is a set of word tokens the
backend located itself. That is the whole reason this server is worth having.
An MCP client supplies the thing Retake cannot do -- understanding what the
creator meant -- and Retake supplies the thing a language model must not be
trusted with: deciding which real audio a phrase refers to. A client asks to
remove "the bit where I fumbled the sponsor read"; Retake answers with exact
token IDs and their timestamps, or says it could not find them. It never
guesses, and a client can never name a raw time span to delete.

Three properties hold for every mutating tool here:

* **Dry run by default.** `apply_cuts`, `apply_proposals`, `set_markers`,
  `set_voice_enhancement`, and `start_export` all preview first. The caller
  sees the exact token IDs and the resulting duration before anything is
  written, and has to come back with ``dry_run=False``.
* **Backed up before every write.** Edits push onto an undo stack of project
  snapshots. `undo` unwinds them one at a time.
* **Deltas, not replacements.** Cutting the words a client just searched for
  cannot silently restore every other edit in the project.

Running it
----------
stdio (Claude Desktop, Claude Code, Cursor, Zed, ...)::

    python retake_mcp.py

streamable HTTP, for remote and web clients::

    python retake_mcp.py --http --port 8711

Either transport exposes the same tools and talks to a running Retake at
``RETAKE_URL`` (default ``http://127.0.0.1:8710``). Start Retake first with
``python retake.py``; this server holds no state of its own.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Literal, Optional

import httpx
from mcp.server.mcpserver import MCPServer

RETAKE_URL = os.environ.get("RETAKE_URL", "http://127.0.0.1:8710").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("RETAKE_MCP_TIMEOUT", "60"))
PLAN_POLL_SECONDS = float(os.environ.get("RETAKE_MCP_PLAN_TIMEOUT", "120"))

INSTRUCTIONS = """Retake is a local transcript-based rough-cut editor for spoken video and audio.

Work in this order:

1. `get_status` to confirm Retake is running and something is open. If no
   project is open, `list_projects` then `open_project`.
2. `get_edit_summary` for the shape of the recording and the current edit.
3. `read_transcript` to read it. It is paginated -- follow `next_offset`.
   Every word carries a token ID; token IDs are the only currency for edits.
4. `find_phrase` to turn a remembered quote into token IDs, or `plan_edit`
   to resolve a whole list of instructions at once. `list_retakes` finds
   repeated takes of the same line.
5. Apply with `apply_cuts` or `apply_proposals`. Both preview by default;
   read the preview, then repeat with dry_run=False.

Never invent token IDs, and never assume a phrase is present -- ask
`find_phrase` and respect a `not_found` answer. Cuts absorb the pause and
breath around them automatically, so do not try to widen a selection to catch
silence. To remove a whole sentence, cut all of its word tokens.

`undo` reverses the last applied edit. Exports are long jobs: `start_export`
returns immediately and `get_job_status` reports progress."""


class RetakeUnavailable(RuntimeError):
    """Raised with a message a human can act on."""


async def _call(
    method: str, path: str, *, json: Any = None, params: Any = None
) -> Any:
    """One request to the local Retake backend, with legible failures."""
    url = f"{RETAKE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.request(method, url, json=json, params=params)
    except httpx.ConnectError as exc:
        raise RetakeUnavailable(
            f"Retake is not running at {RETAKE_URL}. Start it with "
            f"`python retake.py` in the Retake folder, or set RETAKE_URL if it "
            f"listens somewhere else. ({exc})"
        ) from exc
    except httpx.HTTPError as exc:
        raise RetakeUnavailable(f"Retake did not answer {path}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError:
        raise RetakeUnavailable(
            f"Retake returned a non-JSON reply from {path} "
            f"(HTTP {response.status_code})."
        ) from None
    if response.status_code >= 400:
        detail = payload.get("error") if isinstance(payload, dict) else payload
        raise RetakeUnavailable(f"{detail} (HTTP {response.status_code} from {path})")
    return payload


def _confirm(result: dict[str, Any], dry_run: bool, what: str) -> dict[str, Any]:
    """Attach the same next-step sentence to every previewed mutation."""
    if dry_run:
        result["applied"] = False
        result["next_step"] = f"Nothing was written. Repeat {what} with dry_run=false to apply."
    else:
        result["applied"] = True
    return result


mcp = MCPServer(
    name="retake",
    title="Retake",
    version="1.0.0",
    instructions=INSTRUCTIONS,
    website_url="http://127.0.0.1:8710",
)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

@mcp.tool()
async def get_status() -> dict[str, Any]:
    """Check that Retake is running and report what it is doing right now.

    Call this first. Returns the current phase (idle, transcribing, exporting,
    ...), progress, any error, and which project is open.
    """
    status = await _call("GET", "/status")
    result: dict[str, Any] = {"retake_url": RETAKE_URL, "status": status}
    try:
        result["project"] = await _call("GET", "/assistant/summary")
    except RetakeUnavailable as exc:
        result["project"] = None
        result["note"] = str(exc)
    return result


@mcp.tool()
async def list_projects() -> dict[str, Any]:
    """List every project Retake has on disk, newest naming first.

    `available` is false when the original media file has moved or been
    deleted; such a project cannot be opened.
    """
    return await _call("GET", "/projects")


@mcp.tool()
async def open_project(project_id: str) -> dict[str, Any]:
    """Open a saved project by its id (for example "Project 3").

    Opening never re-transcribes: the transcript, cuts, and markers are
    restored as they were left.
    """
    await _call("POST", "/projects/open", json={"id": project_id})
    return await _call("GET", "/assistant/summary")


@mcp.tool()
async def get_edit_summary() -> dict[str, Any]:
    """Summarize the open project and what the current edit produces.

    Includes original and edited duration, how much has been removed, the
    composed cut and keep intervals, word counts, and whether an undo step is
    available. Nothing is rendered -- this is the edit as currently composed.
    """
    return await _call("GET", "/assistant/summary")


@mcp.tool()
async def read_transcript(
    offset: int = 0,
    limit: int = 100,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    only: Literal["all", "kept", "cut"] = "all",
) -> dict[str, Any]:
    """Read the transcript as segments, each with its word tokens and IDs.

    Paginated: recordings run to thousands of segments. Follow `next_offset`
    until it is null. Narrow with `start_s`/`end_s` when you already know
    roughly where to look, and with `only` to see just what is currently kept
    or currently cut.

    Every word carries the token ID that `apply_cuts` expects.
    """
    params: dict[str, Any] = {
        "offset": offset, "limit": limit, "start": start_s, "only": only,
    }
    if end_s is not None:
        params["end"] = end_s
    return await _call("GET", "/assistant/transcript", params=params)


@mcp.tool()
async def find_phrase(
    phrase: str,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    all_matches: bool = False,
    occurrence: Optional[Literal["first", "last"]] = None,
) -> dict[str, Any]:
    """Locate a spoken phrase and return the exact word tokens that carry it.

    This is how a remembered quote becomes something you can cut. Retake
    matches the phrase against the transcript itself, so the answer is either
    an exact span, a strong approximate span, or an honest failure -- a
    `status` of "not_found" or "ambiguous" means do not proceed; narrow the
    window with `start_s`/`end_s`, or pick `occurrence`.

    Set `all_matches` only when the intent really is every occurrence.
    """
    payload: dict[str, Any] = {
        "phrase": phrase, "start": start_s, "all_matches": all_matches,
    }
    if end_s is not None:
        payload["end"] = end_s
    if occurrence is not None:
        payload["occurrence"] = occurrence
    return await _call("POST", "/assistant/find", json=payload)


@mcp.tool()
async def list_retakes() -> dict[str, Any]:
    """List groups of repeated takes -- the same line delivered more than once.

    Retake groups these when it transcribes. Each group's takes are in
    recording order, so the last one is usually the keeper and the earlier ones
    are the candidates for removal.
    """
    return await _call("GET", "/assistant/retakes")


# --------------------------------------------------------------------------
# Planning and editing
# --------------------------------------------------------------------------

@mcp.tool()
async def plan_edit(
    instructions: str = "",
    operations: Optional[list[dict[str, Any]]] = None,
    attachments: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """Resolve a list of edit instructions into reviewable, exact proposals.

    Use this for a whole edit pass rather than one phrase at a time. Retake
    parses what it can from `instructions` on its own -- quoted phrases,
    timestamps, cluster references, mixed Arabic and English -- and you supply
    `operations` for the lines it could not.

    Each operation is::

        {"line": 3,                  # the source line it came from
         "action": "cut_phrase" | "keep_phrase" | "cut_gap" | "needs_decision",
         "phrase": "copied verbatim from the user's line, never translated",
         "start": 61.5, "end": 74.0, # seconds; only if the line states them
         "all_matches": false,
         "occurrence": "first" | "last" | null,
         "reason": "short"}

    You are parsing language, not selecting audio: Retake resolves every
    operation to real tokens itself and reports the ones it could not match
    instead of guessing. Nothing is applied -- the result is staged for
    `apply_proposals`.

    `attachments` may carry reference scripts as ``{"name": ..., "content": ...}``.
    """
    payload: dict[str, Any] = {"instructions": instructions}
    if operations is not None:
        payload["operations"] = operations
    if attachments is not None:
        payload["attachments"] = attachments
    await _call("POST", "/assistant/run", json=payload)

    deadline = asyncio.get_event_loop().time() + PLAN_POLL_SECONDS
    while True:
        result = await _call("GET", "/assistant/result")
        if result.get("done"):
            break
        if asyncio.get_event_loop().time() > deadline:
            return {
                "done": False,
                "note": "still resolving; call get_proposals in a moment",
            }
        await asyncio.sleep(0.25)

    proposals = result.get("proposals") or []
    result["ready_to_apply"] = [
        index for index, item in enumerate(proposals)
        if item.get("action") == "cut_phrase"
        and item.get("status") in {"exact", "approximate"}
    ]
    result["needs_your_attention"] = [
        {"index": index, "status": item.get("status"),
         "instruction": item.get("instruction") or item.get("reason"),
         "action": item.get("action")}
        for index, item in enumerate(proposals)
        if item.get("status") not in {"exact", "approximate"}
    ]
    return result


@mcp.tool()
async def get_proposals() -> dict[str, Any]:
    """Read the proposals staged by the last `plan_edit`, without re-running it."""
    return await _call("GET", "/assistant/result")


@mcp.tool()
async def apply_cuts(
    token_ids: list[int],
    mode: Literal["cut", "keep", "toggle"] = "cut",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Cut, restore, or toggle a specific set of word tokens.

    Previews by default: you get the token IDs that would actually change and
    the resulting edited duration, and nothing is written. Repeat with
    ``dry_run=False`` to apply, which also pushes an undo step.

    This is a delta. Tokens you do not name keep their current state, so an
    edit here can never silently undo an unrelated one. Get token IDs from
    `read_transcript`, `find_phrase`, or `list_retakes` -- never invent them.

    Cuts already absorb the surrounding pause and breath up to the neighboring
    kept words, so there is no need to include extra tokens for silence.
    """
    result = await _call(
        "POST", "/assistant/cuts",
        json={"token_ids": token_ids, "mode": mode, "dry_run": dry_run},
    )
    return _confirm(result, dry_run, "apply_cuts")


@mcp.tool()
async def apply_proposals(
    indexes: Optional[list[int]] = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Accept proposals staged by `plan_edit`, by index. Omit `indexes` for all.

    Previews by default. Proposals that did not resolve to cuttable words are
    reported in `skipped_proposals` rather than being applied approximately.
    """
    payload: dict[str, Any] = {"dry_run": dry_run}
    if indexes is not None:
        payload["indexes"] = indexes
    result = await _call("POST", "/assistant/apply-proposals", json=payload)
    return _confirm(result, dry_run, "apply_proposals")


@mcp.tool()
async def undo() -> dict[str, Any]:
    """Reverse the most recent applied edit, restoring the project snapshot.

    Undo steps stack, so calling this repeatedly walks back through the edits
    made through this server. It does not apply to exports, which never change
    the project.
    """
    return await _call("POST", "/assistant/undo", json={})


@mcp.tool()
async def set_markers(
    markers: list[dict[str, Any]], dry_run: bool = True
) -> dict[str, Any]:
    """Replace the project's markers with this list.

    Each marker is ``{"t": seconds, "color": "coral", "note": "show image"}``.
    This replaces every marker, so read the current ones from
    `get_edit_summary` first if you mean to add rather than reset.
    """
    if dry_run:
        return {
            "applied": False, "would_set": len(markers), "markers": markers,
            "next_step": "Repeat set_markers with dry_run=false to apply.",
        }
    result = await _call("POST", "/markers", json={"markers": markers})
    result["applied"] = True
    return result


@mcp.tool()
async def set_voice_enhancement(
    enabled: Optional[bool] = None,
    output_loudness: Optional[float] = None,
    voice_leveling: Optional[float] = None,
    noise_cleanup: Optional[float] = None,
    original_detail: Optional[float] = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Adjust export-only voice enhancement. All values are percentages, 0-100.

    Enhancement never touches the source recording or the transcript; it is
    applied while encoding. The defaults (loudness 80, leveling 25, cleanup 25,
    detail 90) are deliberately conservative and usually want leaving alone.
    """
    summary = await _call("GET", "/assistant/summary")
    current = await _call("GET", "/project")
    settings = dict(current.get("voice_enhancement") or {})
    for key, value in (
        ("enabled", enabled), ("output_loudness", output_loudness),
        ("voice_leveling", voice_leveling), ("noise_cleanup", noise_cleanup),
        ("original_detail", original_detail),
    ):
        if value is not None:
            settings[key] = value
    if dry_run:
        return {
            "applied": False, "would_set": settings,
            "project": summary.get("project"),
            "next_step": "Repeat set_voice_enhancement with dry_run=false to apply.",
        }
    result = await _call("POST", "/voice-enhancement", json={"settings": settings})
    result["applied"] = True
    return result


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

@mcp.tool()
async def start_export(
    mode: Literal["smart", "compat"] = "smart",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Start rendering the edit to a media file. Returns immediately.

    "smart" is the default reliable-quality H.264/AAC encode; "compat" uses the
    same timeline at a smaller file size. Exports are long -- poll
    `get_job_status` rather than waiting. Only one heavy job runs at a time.

    Previews by default so the edit can be checked before spending the encode.
    """
    summary = await _call("GET", "/assistant/summary")
    if dry_run:
        return {
            "applied": False, "mode": mode,
            "would_export": {
                "project": summary.get("project"),
                "edited_duration_s": summary.get("edited_duration_s"),
                "removed_s": summary.get("removed_s"),
                "cut_intervals": len(summary.get("cut_intervals") or []),
            },
            "next_step": "Repeat start_export with dry_run=false to begin encoding.",
        }
    result = await _call("POST", "/export", json={"mode": mode})
    result["applied"] = True
    result["next_step"] = "Poll get_job_status until phase is ready or an error appears."
    return result


@mcp.tool()
async def get_job_status() -> dict[str, Any]:
    """Report the running job's phase, progress percentage, message, and error.

    Retake runs one heavy job at a time -- transcription, alignment, preview
    proxy, or export. Phase "ready" or "idle" means nothing is running.
    """
    return await _call("GET", "/status")


@mcp.tool()
async def export_text(
    kind: Literal["edl", "csv", "txt", "srt", "markers"],
) -> dict[str, Any]:
    """Write a text export of the current edit and return its path.

    These are fast and use original transcript timestamps: an EDL or CSV cut
    list, the kept transcript as plain text, subtitles, or the marker list.
    """
    return await _call("POST", "/export_text", json={"kind": kind})


@mcp.tool()
async def get_latest_export() -> dict[str, Any]:
    """Report the newest finished media export for the open project."""
    return await _call("GET", "/export/latest")


@mcp.tool()
async def recalibrate_timing(dry_run: bool = True) -> dict[str, Any]:
    """Re-time every word against the audio with forced alignment.

    Worth doing once when word boundaries feel late or early, which tightens
    every cut in the project. It needs the optional alignment runtime
    installed, runs once, is cached, and backs the project up first. Word IDs,
    text, cuts, and markers are all preserved.
    """
    status = await _call("GET", "/alignment/status")
    if dry_run:
        return {
            "applied": False, "alignment": status,
            "next_step": "Repeat recalibrate_timing with dry_run=false to start it.",
        }
    result = await _call("POST", "/alignment/recalibrate", json={})
    result["applied"] = True
    result["next_step"] = "Poll get_job_status; this is a long job."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Retake MCP server")
    parser.add_argument(
        "--http", action="store_true",
        help="serve streamable HTTP instead of stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8711)
    parser.add_argument(
        "--path", default="/mcp", help="HTTP path to mount the endpoint on",
    )
    args = parser.parse_args()
    if args.http:
        asyncio.run(
            mcp.run_streamable_http_async(
                host=args.host, port=args.port, streamable_http_path=args.path,
            )
        )
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
