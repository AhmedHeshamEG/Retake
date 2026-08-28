"""Unit tests for the cut-interval math in retake.py.

Run with:  python test_retake.py   (or pytest test_retake.py)
"""
from __future__ import annotations

import json
import os
import ssl
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
import retake

from retake import (
    clusters_from_similarity,
    deterministic_retake_proposals,
    deterministic_instruction_plan,
    consecutive_word_cut_intervals,
    cut_composition_params,
    cut_intervals_from_tokens,
    ensure_audio_gap_state,
    ensure_voice_enhancement_state,
    keep_list,
    loudness_target_lufs,
    merge_intervals,
    parse_silencedetect_output,
    refine_export_keep_edges,
    retake_text_match,
    sanitize_clusters,
    validate_ai_attachments,
    app,
    enforce_llm_budget,
    explicit_target_terms,
    is_detailed_edit_request,
    lan_urls,
    resolve_detailed_operations,
    resolve_phrase_tokens,
    timestamps_in_text,
    validate_planned_operations,
    validate_voice_enhancement,
)


@contextmanager
def temporary_project_root():
    old_projects, old_incoming = retake.PROJECTS_DIR, retake.INCOMING_DIR
    with tempfile.TemporaryDirectory() as folder:
        retake.PROJECTS_DIR = Path(folder)
        retake.INCOMING_DIR = retake.PROJECTS_DIR / ".incoming"
        retake.INCOMING_DIR.mkdir()
        try:
            yield retake.PROJECTS_DIR
        finally:
            retake.PROJECTS_DIR, retake.INCOMING_DIR = old_projects, old_incoming


def approx(got: list[tuple[float, float]], want: list[tuple[float, float]], tol: float = 1e-6) -> bool:
    return len(got) == len(want) and all(
        abs(a - c) <= tol and abs(b - d) <= tol for (a, b), (c, d) in zip(got, want)
    )


def test_empty() -> None:
    assert merge_intervals([]) == []
    assert approx(keep_list([], 10.0), [(0.0, 10.0)])


def test_adjacent() -> None:
    # touching within epsilon (1 ms) must merge
    assert approx(merge_intervals([(0.0, 1.0), (1.0005, 2.0)]), [(0.0, 2.0)])
    # a real gap must not merge
    assert approx(merge_intervals([(0.0, 1.0), (1.01, 2.0)]), [(0.0, 1.0), (1.01, 2.0)])


def test_overlapping() -> None:
    assert approx(merge_intervals([(0.0, 2.0), (1.0, 3.0)]), [(0.0, 3.0)])


def test_nested() -> None:
    assert approx(merge_intervals([(0.0, 5.0), (1.0, 2.0), (3.0, 4.0)]), [(0.0, 5.0)])


def test_unsorted() -> None:
    assert approx(
        merge_intervals([(6.0, 7.0), (0.0, 1.0), (0.5, 2.0)]),
        [(0.0, 2.0), (6.0, 7.0)],
    )


def test_zero_length_dropped() -> None:
    assert merge_intervals([(1.0, 1.0), (3.0, 2.0)]) == []


def test_keep_list_basic() -> None:
    keeps = keep_list([(1.0, 2.0), (4.0, 5.0)], 10.0)
    assert approx(keeps, [(0.0, 1.0), (2.0, 4.0), (5.0, 10.0)])


def test_keep_list_edges() -> None:
    # cut reaching both edges
    assert approx(keep_list([(0.0, 3.0), (8.0, 10.0)], 10.0), [(3.0, 8.0)])
    # cut extending past the end is clamped
    assert approx(keep_list([(9.0, 42.0)], 10.0), [(0.0, 9.0)])


def test_everything_cut() -> None:
    assert keep_list([(0.0, 10.0)], 10.0) == []
    assert keep_list([(0.0, 6.0), (5.0, 10.0)], 10.0) == []


def test_audio_silence_parser_handles_leading_and_trailing_gaps() -> None:
    output = """
    [silencedetect] silence_start: 0
    [silencedetect] silence_end: 1.25 | silence_duration: 1.25
    [silencedetect] silence_start: 8.5
    """
    assert approx(parse_silencedetect_output(output, 10.0), [(0.0, 1.25), (8.5, 10.0)])


def test_export_audio_gap_cuts_join_words_without_rewriting_timestamps() -> None:
    proj = {
        "tokens": [{"id": 0, "kind": "word", "text": "hello", "start": 1.0,
                    "end": 2.0, "seg": 0, "cut": True}],
        "audio_gaps": [{"id": "agap-1", "start": 1.8, "end": 3.0, "cut": True}],
    }
    # The only word is cut and nothing is kept before it, so the run also takes
    # the leading second; the legacy gap still extends it to 3.0.
    assert approx(cut_intervals_from_tokens(proj), [(0.0, 3.0)])
    assert proj["tokens"][0]["start"] == 1.0 and proj["tokens"][0]["end"] == 2.0


def test_consecutive_cut_words_bridge_unreported_and_cross_sentence_gaps() -> None:
    tokens = [
        {"id": 30, "kind": "word", "text": "Hi", "start": .001, "end": .28,
         "seg": 0, "cut": True},
        {"id": 31, "kind": "word", "text": "this", "start": .35, "end": .5,
         "seg": 0, "cut": True},
        {"id": 32, "kind": "gap", "start": .5, "end": 2.0, "cut": False},
        {"id": 33, "kind": "word", "text": "later", "start": 2.2, "end": 2.6,
         "seg": 1, "cut": True},
        {"id": 34, "kind": "word", "text": "keep", "start": 2.8, "end": 3.1,
         "seg": 1, "cut": False},
        {"id": 35, "kind": "word", "text": "remove", "start": 3.4, "end": 3.8,
         "seg": 1, "cut": True},
    ]
    assert approx(
        retake.consecutive_word_cut_intervals(tokens),
        [(.001, 2.6), (3.4, 3.8)],
    )


def test_consecutive_cut_composer_orders_legacy_tokens_and_ignores_bad_timing() -> None:
    tokens = [
        {"id": 9, "kind": "word", "start": 4.0, "end": 4.2, "cut": True},
        {"id": 7, "kind": "word", "start": "bad", "end": 3.9, "cut": True},
        {"id": 1, "kind": "word", "start": 1.0, "end": 1.3, "cut": True},
        {"id": 4, "kind": "word", "start": 2.0, "end": 2.3, "cut": False},
    ]
    assert approx(
        retake.consecutive_word_cut_intervals(tokens),
        [(1.0, 1.3), (4.0, 4.2)],
    )


def test_cut_api_returns_authoritative_runs_and_restore_splits_them() -> None:
    old_current = dict(retake.CURRENT)
    with temporary_project_root() as root:
        project_dir = retake.next_project_directory()
        source = project_dir / "media" / "original.wav"
        source.write_bytes(b"fixture")
        project = {
            "schema_version": 1,
            "source_path": str(source),
            "duration_s": 5.0,
            "gap_threshold_s": .35,
            "tokens": [
                {"id": 0, "kind": "word", "text": "one", "start": 0.1,
                 "end": .3, "seg": 0, "cut": False},
                {"id": 1, "kind": "gap", "start": .3, "end": 1.5, "cut": False},
                {"id": 2, "kind": "word", "text": "two", "start": 1.7,
                 "end": 2.0, "seg": 1, "cut": False},
                {"id": 3, "kind": "word", "text": "three", "start": 2.2,
                 "end": 2.5, "seg": 1, "cut": False},
            ],
            "audio_gaps": [
                {"id": "agap-seg-1", "detected_start": 2.05, "detected_end": 2.15,
                 "start": 2.05, "end": 2.15, "cut": False},
            ],
        }
        try:
            retake.CURRENT.update(
                project=project, media_path=str(source), project_dir=str(project_dir),
                source_filename=source.name,
            )
            client = TestClient(app)
            # A cut run absorbs the non-speech on both of its outer sides: it
            # runs from the previous kept word's end to the next kept word's
            # start, each held back by the unaligned safety handle (0.12s).
            # Cutting word 0 therefore also removes the leading 0.1s and the
            # 1.4s gap that follows it, up to 0.12s before "two".
            assert client.post("/cuts", json={"cut_ids": [0]}).json()[
                "transcript_cut_intervals"
            ] == [[0.0, 1.58]]

            # Nothing is kept, so the run reaches both ends of the recording.
            joined = client.post("/cuts", json={"cut_ids": [0, 2, 3]}).json()
            assert joined["word_cut_intervals"] == [[0.0, 5.0]]
            assert joined["transcript_cut_intervals"] == [[0.0, 5.0]]
            assert client.get("/project").json()["cut_intervals"] == [[0.0, 5.0]]

            restored = client.post("/cuts", json={"cut_ids": [0, 3]}).json()
            assert restored["transcript_cut_intervals"] == [[0.0, 1.58], [2.12, 5.0]]
            saved = (project_dir / "project.json").read_text(encoding="utf-8")
            assert "transcript_cut_intervals" not in saved and "word_cut_intervals" not in saved
        finally:
            retake.CURRENT.clear()
            retake.CURRENT.update(old_current)


def test_preview_consumes_backend_composed_transcript_intervals() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "transcript_cut_intervals" in html
    assert "S.transcriptCutIvs = (result.transcript_cut_intervals || [])" in html
    assert "const ivs = S.cutsSynced ? S.transcriptCutIvs" in html


def test_export_keep_edges_snap_to_nearby_silence_only() -> None:
    keeps = [(0.0, 5.0), (10.0, 25.0), (30.0, 40.0)]
    silences = [(4.82, 5.08), (9.78, 9.92), (25.12, 25.24), (29.91, 30.08)]
    assert refine_export_keep_edges(keeps, silences, 40.0) == [
        (0.0, 4.82), (9.92, 25.12), (30.08, 40.0)
    ]


def test_export_keep_edges_leave_missing_or_distant_silence_unchanged() -> None:
    keeps = [(0.0, 5.0), (10.0, 25.0), (30.0, 40.0)]
    assert refine_export_keep_edges(keeps, [], 40.0) == keeps
    assert refine_export_keep_edges(keeps, [(8.0, 8.2), (27.0, 27.2)], 40.0) == keeps


def test_export_keep_edges_reject_overlap_and_invalid_silence() -> None:
    keeps = [(4.95, 5.25)]
    silences = [(4.9, 5.3), (float("nan"), 9.0)]
    assert refine_export_keep_edges(keeps, silences, 10.0) == keeps


def _word(idx: int, start: float, end: float, cut: bool = False) -> dict:
    return {"id": idx, "kind": "word", "text": f"w{idx}", "start": start,
            "end": end, "seg": 0, "cut": cut}


def test_cut_run_absorbs_the_pause_around_deleted_speech() -> None:
    """The regression this whole behavior exists for.

    Whisper reports a deleted sentence as first-word-start .. last-word-end, so
    the breath before it and the pause after it used to belong to no cut and
    survived the edit as audible dead air between two kept sentences.
    """
    proj = {
        "duration_s": 6.0,
        "tokens": [
            _word(0, 0.0, 1.0),
            _word(1, 1.8, 2.4, cut=True),
            _word(2, 2.5, 3.0, cut=True),
            _word(3, 4.0, 5.0),
        ],
    }
    handle = cut_composition_params(proj)["safety_handle"]
    assert approx(cut_intervals_from_tokens(proj), [(1.0 + handle, 4.0 - handle)])
    # Only the two kept words remain, back to back, with the handle around them.
    assert approx(
        keep_list(cut_intervals_from_tokens(proj), 6.0),
        [(0.0, 1.0 + handle), (4.0 - handle, 6.0)],
    )


def test_cut_run_at_the_head_or_tail_absorbs_to_the_recording_edge() -> None:
    proj = {
        "duration_s": 6.0,
        "tokens": [_word(0, 0.5, 1.0, cut=True), _word(1, 2.0, 3.0),
                   _word(2, 4.0, 4.5, cut=True)],
    }
    handle = cut_composition_params(proj)["safety_handle"]
    assert approx(
        cut_intervals_from_tokens(proj),
        [(0.0, 2.0 - handle), (3.0 + handle, 6.0)],
    )


def test_absorbing_cut_never_enters_adjacent_kept_speech() -> None:
    """Back-to-back words: absorption must not eat into either neighbor."""
    tokens = [_word(0, 0.0, 1.0), _word(1, 1.0, 2.0, cut=True), _word(2, 2.0, 3.0)]
    params = cut_composition_params({"duration_s": 3.0, "tokens": tokens})
    handle = params["safety_handle"]
    cuts = consecutive_word_cut_intervals(tokens, **params)
    assert approx(cuts, [(1.0 + handle, 2.0 - handle)])
    for start, end in cuts:
        assert start >= 1.0 and end <= 2.0


def test_missing_duration_leaves_a_trailing_run_at_the_last_cut_word() -> None:
    tokens = [_word(0, 0.0, 1.0), _word(1, 2.0, 3.0, cut=True)]
    assert approx(
        consecutive_word_cut_intervals(tokens, safety_handle=0.1, absorb_pauses=True),
        [(1.1, 3.0)],
    )


def test_export_keep_edges_trim_much_further_than_they_restore() -> None:
    """Trimming only drops measured silence; restoring returns excluded audio."""
    # Keep start sits 1.5s inside a silence: trimming forward to speech is allowed.
    assert refine_export_keep_edges([(5.0, 10.0)], [(4.0, 6.5)], 10.0) == [(6.5, 10.0)]
    # Keep end sits 1.2s after speech stopped: trimming backward is allowed.
    assert refine_export_keep_edges([(0.0, 5.0)], [(3.8, 5.4)], 10.0) == [(0.0, 3.8)]
    # The same distance in the restoring direction is refused.
    assert refine_export_keep_edges([(5.0, 9.0)], [(3.0, 3.5)], 10.0) == [(5.0, 9.0)]
    # A small restore still recovers a clipped word release.
    assert refine_export_keep_edges([(0.0, 5.0)], [(5.2, 6.0)], 10.0) == [(0.0, 5.2)]


def test_export_keep_edge_restore_never_reaches_across_the_previous_keep() -> None:
    refined = refine_export_keep_edges(
        [(0.0, 4.0), (4.3, 9.0)], [(3.9, 4.05), (4.1, 4.2)], 10.0
    )
    for index, (start, end) in enumerate(refined):
        assert end > start
        if index:
            assert start >= refined[index - 1][1]


@contextmanager
def assistant_project():
    """A tiny open project on disk, restored afterwards."""
    old_current = dict(retake.CURRENT)
    old_projects = retake.PROJECTS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        retake.PROJECTS_DIR = root
        project_dir = root / "Project 1"
        (project_dir / "media").mkdir(parents=True)
        source = project_dir / "media" / "original.wav"
        source.write_bytes(b"RIFF0000WAVE")
        project = {
            "schema_version": 1, "source_path": str(source), "duration_s": 12.0,
            "language": "en",
            "tokens": [
                {"id": 0, "kind": "word", "text": "keep", "start": 0.0, "end": 1.0,
                 "seg": 0, "cut": False},
                {"id": 1, "kind": "word", "text": "this", "start": 1.1, "end": 1.8,
                 "seg": 0, "cut": False},
                {"id": 2, "kind": "word", "text": "remove", "start": 4.0, "end": 4.6,
                 "seg": 1, "cut": False},
                {"id": 3, "kind": "word", "text": "that", "start": 4.7, "end": 5.2,
                 "seg": 1, "cut": False},
                {"id": 4, "kind": "word", "text": "end", "start": 9.0, "end": 9.8,
                 "seg": 2, "cut": False},
            ],
            "segments": [
                {"id": 0, "start": 0.0, "end": 1.8, "text": "keep this"},
                {"id": 1, "start": 4.0, "end": 5.2, "text": "remove that"},
                {"id": 2, "start": 9.0, "end": 9.8, "text": "end"},
            ],
            "clusters": [], "markers": [],
        }
        retake.atomic_write_json(project_dir / "project.json", project)
        retake.CURRENT.update(
            project=project, media_path=str(source),
            project_dir=str(project_dir), source_filename="original.wav",
        )
        try:
            yield TestClient(app), project, project_dir
        finally:
            retake.CURRENT.clear()
            retake.CURRENT.update(old_current)
            retake.PROJECTS_DIR = old_projects


def test_assistant_edits_preview_before_they_write() -> None:
    """dry_run is the default and must leave the project completely alone."""
    with assistant_project() as (client, project, project_dir):
        preview = client.post("/assistant/cuts", json={"token_ids": [2, 3]}).json()
        assert preview["dry_run"] is True
        assert preview["changed_token_ids"] == [2, 3]
        # The preview reports the real consequence...
        assert preview["after"]["edited_duration_s"] < preview["before"]["edited_duration_s"]
        # ...without any of it having happened.
        assert [t["cut"] for t in project["tokens"]] == [False] * 5
        assert "backup" not in preview
        assert not retake.assistant_backups(project_dir)

        applied = client.post(
            "/assistant/cuts", json={"token_ids": [2, 3], "dry_run": False}
        ).json()
        assert applied["dry_run"] is False and applied["backup"]
        assert [t["cut"] for t in retake.CURRENT["project"]["tokens"]] == [
            False, False, True, True, False
        ]


def test_assistant_undo_unwinds_edits_in_reverse_order() -> None:
    with assistant_project() as (client, _project, _project_dir):
        client.post("/assistant/cuts", json={"token_ids": [2, 3], "dry_run": False})
        client.post("/assistant/cuts", json={"token_ids": [0, 1], "dry_run": False})
        cuts = lambda: [t["cut"] for t in retake.CURRENT["project"]["tokens"]]
        assert cuts() == [True, True, True, True, False]
        first = client.post("/assistant/undo", json={}).json()
        assert first["remaining_undo_steps"] == 1
        assert cuts() == [False, False, True, True, False]
        client.post("/assistant/undo", json={})
        assert cuts() == [False] * 5
        assert client.post("/assistant/undo", json={}).status_code == 404


def test_assistant_backup_holds_the_state_before_the_edit() -> None:
    """The regression that makes undo a no-op if the snapshot is taken late."""
    with assistant_project() as (client, _project, project_dir):
        client.post("/assistant/cuts", json={"token_ids": [2, 3], "dry_run": False})
        backups = retake.assistant_backups(project_dir)
        assert len(backups) == 1
        saved = json.loads(backups[0].read_text(encoding="utf-8"))
        assert [t["cut"] for t in saved["tokens"]] == [False] * 5


def test_assistant_cuts_are_a_delta_not_a_replacement() -> None:
    """Editing named tokens must never restore unrelated ones."""
    with assistant_project() as (client, _project, _project_dir):
        client.post("/assistant/cuts", json={"token_ids": [0], "dry_run": False})
        client.post("/assistant/cuts", json={"token_ids": [4], "dry_run": False})
        assert [t["cut"] for t in retake.CURRENT["project"]["tokens"]] == [
            True, False, False, False, True
        ]


def test_assistant_rejects_unknown_tokens_and_bad_modes() -> None:
    with assistant_project() as (client, _project, _project_dir):
        assert client.post("/assistant/cuts", json={"token_ids": [999]}).status_code == 400
        assert client.post(
            "/assistant/cuts", json={"token_ids": [0], "mode": "delete"}
        ).status_code == 400
        assert client.post("/assistant/cuts", json={"token_ids": []}).status_code == 400
        assert client.post(
            "/assistant/cuts", json={"token_ids": ["one"]}
        ).status_code == 400


def test_assistant_find_reports_misses_instead_of_guessing() -> None:
    with assistant_project() as (client, _project, _project_dir):
        hit = client.post("/assistant/find", json={"phrase": "remove that"}).json()
        assert hit["status"] == "exact" and hit["token_ids"] == [2, 3]
        assert hit["start"] == 4.0 and hit["end"] == 5.2
        miss = client.post(
            "/assistant/find", json={"phrase": "never spoken words here"}
        ).json()
        assert miss["status"] != "exact" and miss["token_ids"] == []
        assert client.post("/assistant/find", json={"phrase": "  "}).status_code == 400


def test_assistant_transcript_paginates_and_filters() -> None:
    with assistant_project() as (client, _project, _project_dir):
        first = client.get("/assistant/transcript", params={"limit": 2}).json()
        assert first["total"] == 3 and first["next_offset"] == 2
        assert [row["segment_id"] for row in first["segments"]] == [0, 1]
        assert first["segments"][0]["words"][0]["id"] == 0
        last = client.get(
            "/assistant/transcript", params={"limit": 2, "offset": 2}
        ).json()
        assert last["next_offset"] is None and last["returned"] == 1
        client.post("/assistant/cuts", json={"token_ids": [2, 3], "dry_run": False})
        only_cut = client.get("/assistant/transcript", params={"only": "cut"}).json()
        assert [row["segment_id"] for row in only_cut["segments"]] == [1]
        assert client.get(
            "/assistant/transcript", params={"only": "sideways"}
        ).status_code == 400


def test_assistant_summary_reports_the_composed_edit() -> None:
    with assistant_project() as (client, _project, _project_dir):
        client.post("/assistant/cuts", json={"token_ids": [2, 3], "dry_run": False})
        summary = client.get("/assistant/summary").json()
        assert summary["duration_s"] == 12.0
        assert summary["cut_word_count"] == 2
        assert summary["edited_duration_s"] < 12.0
        assert summary["undo_available"] is True
        # The absorbed pause is visible here, not just at export time.
        assert summary["cut_intervals"] == [[1.92, 8.88]]


def test_assistant_endpoints_require_an_open_project() -> None:
    old = dict(retake.CURRENT)
    retake.CURRENT.clear()
    retake.CURRENT.update(project=None, media_path=None, project_dir=None)
    try:
        client = TestClient(app)
        for method, path in (
            ("get", "/assistant/summary"), ("get", "/assistant/transcript"),
            ("get", "/assistant/retakes"),
        ):
            assert getattr(client, method)(path).status_code == 404, path
        assert client.post("/assistant/find", json={"phrase": "x"}).status_code == 404
        assert client.post("/assistant/cuts", json={"token_ids": [0]}).status_code == 404
        assert client.post("/assistant/undo", json={}).status_code == 404
    finally:
        retake.CURRENT.clear()
        retake.CURRENT.update(old)


def test_assistant_is_always_available_without_a_model() -> None:
    payload = TestClient(app).get("/assistant/available").json()
    assert payload["available"] is True
    assert payload["engine"] == "deterministic"
    assert payload["mcp"]["http_path"] == retake.MCP_HTTP_PATH


def test_no_local_llm_remains() -> None:
    """The GGUF path is gone; the deterministic resolver is not."""
    source = Path(__file__).with_name("retake.py").read_text(encoding="utf-8")
    for gone in ("llama_cpp", "list_ggufs", "LLM_DIR", ".gguf", "AI_SYSTEM_PROMPT"):
        assert gone not in source, gone
    for kept in ("resolve_phrase_tokens", "resolve_detailed_operations",
                 "validate_planned_operations", "deterministic_instruction_plan"):
        assert kept in source, kept


def test_mcp_server_exposes_the_documented_tool_surface() -> None:
    import asyncio

    import retake_mcp

    tools = {tool.name: tool for tool in asyncio.run(retake_mcp.mcp.list_tools())}
    expected = {
        "get_status", "list_projects", "open_project", "get_edit_summary",
        "read_transcript", "find_phrase", "list_retakes", "plan_edit",
        "get_proposals", "apply_cuts", "apply_proposals", "undo", "set_markers",
        "set_voice_enhancement", "start_export", "get_job_status", "export_text",
        "get_latest_export", "recalibrate_timing",
    }
    assert expected <= set(tools), expected - set(tools)
    # Every tool has to explain itself; MCP clients only see descriptions.
    for name, tool in tools.items():
        assert (tool.description or "").strip(), name


def test_mcp_mutating_tools_default_to_a_dry_run() -> None:
    import asyncio

    import retake_mcp

    tools = {tool.name: tool for tool in asyncio.run(retake_mcp.mcp.list_tools())}
    for name in ("apply_cuts", "apply_proposals", "set_markers",
                 "set_voice_enhancement", "start_export", "recalibrate_timing"):
        schema = tools[name].input_schema
        assert "dry_run" in schema["properties"], name
        assert schema["properties"]["dry_run"].get("default") is True, name
        assert "dry_run" not in schema.get("required", []), name


def test_mcp_endpoint_is_mounted_and_optional() -> None:
    assert retake.MCP_HTTP_PATH == "/mcp"
    mounted = [
        route for route in app.routes
        if getattr(route, "path", None) == retake.MCP_HTTP_PATH
    ]
    assert mounted, "the /mcp endpoint should be mounted when mcp is installed"
    source = Path(__file__).with_name("retake.py").read_text(encoding="utf-8")
    # The editor must still start when the optional dependency is absent.
    assert "MCP endpoint disabled" in source


SKILL_PATH = Path(__file__).with_name("skills") / "retake-brain" / "SKILL.md"


def skill_example_cutlist() -> str:
    """The example block out of SKILL.md, so the two can never drift apart."""
    skill = SKILL_PATH.read_text(encoding="utf-8")
    start = skill.index("**Cluster 2 ")
    return skill[start:skill.index("**Closing -")].strip()


def test_skill_example_cutlist_parses_completely() -> None:
    """Every command the skill documents must be one the parser understands."""
    cutlist = skill_example_cutlist()
    operations, covered = deterministic_instruction_plan(cutlist)
    lines = cutlist.splitlines()
    assert not (retake._actionable_instruction_lines(lines) - covered)

    by_action: dict[str, list[dict]] = {}
    for op in operations:
        by_action.setdefault(op["action"], []).append(op)
    assert set(by_action) == {"keep_phrase", "cut_phrase", "cut_gap", "needs_decision"}

    # "(أول مرة)" and "(آخر مرة)" are the skill's ordinal markers.
    assert by_action["cut_phrase"][0]["occurrence"] == "first"
    assert by_action["keep_phrase"][-1]["occurrence"] == "last"
    # The gap command carries both of its timestamps.
    gap = by_action["cut_gap"][0]
    assert (round(gap["start"], 1), round(gap["end"], 1)) == (75.5, 97.9)


def test_arabic_last_marker_scopes_cuts_and_keeps() -> None:
    """'آخر مرة' was silently ignored, so the wrong take was kept."""
    for line, action in (
        ('خلّي: "the same sentence" (آخر مرة)', "keep_phrase"),
        ('شيل: "the same sentence" (آخر مرة)', "cut_phrase"),
        ('keep: "the same sentence" (last)', "keep_phrase"),
    ):
        operations, _ = deterministic_instruction_plan(line)
        assert len(operations) == 1, line
        assert operations[0]["action"] == action, line
        assert operations[0]["occurrence"] == "last", line
    first, _ = deterministic_instruction_plan('شيل: "x" (أول مرة)')
    assert first[0]["occurrence"] == "first"


def test_decision_block_options_are_never_executed() -> None:
    """Both branches use the command grammar; running both would self-conflict."""
    cutlist = (
        'قرار مطلوب: بتقول "12 days" و"two weeks" — اختار الرقم الصح:\n'
        '- لو 12 days → خلّي: "it stayed for 12 days" + شيل: "it stayed for two weeks"\n'
        '- لو two weeks → خلّي: "it stayed for two weeks" + شيل: "it was 12 days"\n'
        '\n'
        'شيل: "a real command after the block"'
    )
    operations, _ = deterministic_instruction_plan(cutlist)
    actions = [op["action"] for op in operations]
    assert actions.count("needs_decision") == 1
    # Nothing from either branch became an edit...
    for op in operations:
        assert "12 days" not in op["phrase"] and "two weeks" not in op["phrase"]
    # ...but the block does not swallow the commands that follow it.
    assert any(op["phrase"] == "a real command after the block" for op in operations)


def test_cluster_headers_scope_the_search_window() -> None:
    """The skill promises headers bound the search; the parser must honour it."""
    cutlist = skill_example_cutlist()
    _lines, bounds = retake._instruction_context(cutlist, 600.0)
    operations, _ = deterministic_instruction_plan(cutlist)
    windows = {op["line"]: bounds[op["line"]] for op in operations}
    # Commands under "Cluster 2 (00:58.9 -> 01:37.9)" search only that range.
    assert all(
        (round(lo, 1), round(hi, 1)) == (58.9, 97.9)
        for line, (lo, hi) in windows.items() if line < 8
    )
    assert all(
        (round(lo, 1), round(hi, 1)) == (213.9, 257.7)
        for line, (lo, hi) in windows.items() if line > 8
    )


def test_skill_matches_the_app_it_drives() -> None:
    """Guidance that describes removed features is worse than no guidance."""
    skill = SKILL_PATH.read_text(encoding="utf-8")
    # The manual gap editor is gone. The skill may only mention its button in
    # order to disown it, never to recommend it.
    assert "There is no" in skill and "cut all gaps" in skill
    assert "Never suggest it." in skill
    assert "recommending the app" not in skill
    # Cuts absorb their own surrounding pause, so listing those gaps is noise.
    assert "absorbs the non-speech" in skill
    # The MCP loop it documents has to use the tools that exist.
    import asyncio

    import retake_mcp

    available = {t.name for t in asyncio.run(retake_mcp.mcp.list_tools())}
    for named in ("get_status", "list_projects", "open_project", "read_transcript",
                  "find_phrase", "plan_edit", "apply_proposals", "undo",
                  "start_export", "recalibrate_timing"):
        assert named in skill, named
        assert named in available, named


def test_voice_enhancement_defaults_validation_and_loudness_mapping() -> None:
    proj = {"tokens": []}
    assert ensure_voice_enhancement_state(proj)
    settings = validate_voice_enhancement(proj["voice_enhancement"])
    assert settings["enabled"] is True and settings["profile"] == "great"
    assert settings["noise_cleanup"] == 25.0 and settings["original_detail"] == 90.0
    assert loudness_target_lufs(0) == -24.0
    assert loudness_target_lufs(80) == -16.0
    assert loudness_target_lufs(100) == -14.0
    assert not ensure_voice_enhancement_state(proj)
    for bad in (-1, 101, float("nan"), True):
        try:
            validate_voice_enhancement({"noise_cleanup": bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid cleanup accepted: {bad!r}")


def test_retakes_do_not_chain_through_weak_links() -> None:
    segments = [
        {"id": i, "start": i * 4.0, "end": i * 4.0 + 3.0,
         "text": "repeat this complete sentence"}
        for i in range(3)
    ]
    similarity = [
        [1.0, 0.90, 0.70],
        [0.90, 1.0, 0.90],
        [0.70, 0.90, 1.0],
    ]
    clusters = clusters_from_similarity(segments, similarity)
    assert clusters == [{"id": 0, "members": [0, 1]}]


def test_unsafe_legacy_cluster_is_dropped() -> None:
    segments = [
        {"id": i, "start": i * 10.0, "end": i * 10.0 + 5.0, "text": str(i)}
        for i in range(12)
    ]
    assert sanitize_clusters(segments, [{"id": 0, "members": list(range(12))}]) == []


def test_partial_match_must_be_an_opening_prefix() -> None:
    assert retake_text_match(
        "the fifth one",
        "the fifth one and the last one we discuss today",
    )
    assert not retake_text_match(
        "but we care about the direction",
        "care about the direction in our case we do not care about the length",
    )


def test_deterministic_retakes_keep_final_take() -> None:
    segments = [
        {"id": i, "start": i * 3.0, "end": i * 3.0 + 2.0, "text": f"take {i}"}
        for i in range(3)
    ]
    proposals, ids = deterministic_retake_proposals(
        segments, [{"id": 0, "members": [0, 1, 2]}]
    )
    assert ids == {0, 1}
    assert [p["sentence_ids"] for p in proposals] == [[0], [1]]
    assert all(p["source"] == "retake rule" for p in proposals)


def test_text_attachments_are_bounded_and_binary_is_ignored() -> None:
    accepted, warnings = validate_ai_attachments([
        {"name": "script.md", "content": "reference script"},
        {"name": "bad.txt", "content": "binary\x00data"},
    ])
    assert accepted == [{"name": "script.md", "content": "reference script"}]
    assert any("binary" in warning for warning in warnings)


def test_model_hubs_are_forced_offline() -> None:
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_lan_urls_carry_a_scheme_and_port() -> None:
    assert all(url.startswith("https://") and url.endswith(":8443")
               for url in lan_urls(retake.HTTPS_PORT, "https"))


def test_lan_certificate_covers_localhost_and_every_lan_address() -> None:
    """iOS only grants a wake lock over HTTPS, so the phone must trust this."""
    try:
        from cryptography import x509
    except ImportError:  # optional dependency
        return
    old_dir = retake.TLS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        retake.TLS_DIR = Path(tmp) / "tls"
        try:
            pair = retake.ensure_lan_certificate()
            assert pair is not None
            certificate_path, key_path = pair
            assert certificate_path.exists() and key_path.exists()
            # A real TLS stack has to accept the pair, not just the file bytes.
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(str(certificate_path), str(key_path))

            certificate = x509.load_pem_x509_certificate(certificate_path.read_bytes())
            names = certificate.extensions.get_extension_for_class(
                x509.SubjectAlternativeName
            ).value
            hosts = {str(entry.value) for entry in names}
            assert "localhost" in hosts and "127.0.0.1" in hosts
            for address in retake.lan_addresses():
                assert address in hosts, address
            assert certificate.not_valid_after_utc > datetime.now(timezone.utc)
            # Second call reuses it rather than churning a new key.
            assert retake.ensure_lan_certificate() == pair
        finally:
            retake.TLS_DIR = old_dir


def test_serving_https_keeps_the_plain_http_address() -> None:
    """The desktop workflow must not move just because phones need TLS."""
    source = Path(retake.__file__).read_text(encoding="utf-8")
    serving = source[source.index("async def serve_forever("):]
    assert "port=PORT" in serving and "port=HTTPS_PORT" in serving
    assert "ssl_certfile=str(certificate)" in serving
    # A missing certificate leaves exactly the plain HTTP listener.
    assert "if tls is not None:" in serving
    main_source = source[source.index("def main() -> None:"):]
    assert "serving HTTP only" in main_source


def test_both_listeners_share_one_event_loop_and_one_mcp_manager() -> None:
    """The regression that stopped HTTPS from starting at all.

    Running the same app on two uvicorn servers runs the lifespan twice, and
    StreamableHTTPSessionManager.run() refuses a second call -- so the second
    listener died on startup. The manager also binds to the loop that started
    it, so the two listeners cannot live in separate threads.
    """
    source = Path(retake.__file__).read_text(encoding="utf-8")
    serving = source[source.index("async def serve_forever("):]
    # One loop: gathered coroutines, not a thread per server.
    assert "await asyncio.gather(" in serving
    assert "threading.Thread" not in serving
    # Stopping one listener stops the other, so Ctrl+C ends the process.
    assert "other.should_exit = True" in serving

    lifespan = source[source.index("async def _lifespan("):]
    lifespan = lifespan[:lifespan.index("app = FastAPI(")]
    assert "_MCP_MANAGER_RUNNING" in lifespan
    assert "or _MCP_MANAGER_RUNNING:" in lifespan


def test_port_clash_degrades_instead_of_exiting() -> None:
    import socket as _socket

    held = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    held.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    held.bind(("0.0.0.0", 0))
    held.listen(1)
    port = held.getsockname()[1]
    try:
        assert retake.port_is_free(port) is False
    finally:
        held.close()
    # A port nothing holds is reported free, and probing does not keep it.
    assert retake.port_is_free(port) is True
    assert retake.port_is_free(port) is True


def test_transfer_waits_for_the_tab_instead_of_burning_retries() -> None:
    """A locked phone suspends fetch; retrying against a sleeping tab is waste."""
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "function whenVisible()" in html
    assert "await whenVisible();" in html
    assert "resuming automatically" in html
    # The old dead end is gone.
    assert "Transfer paused safely on the laptop" not in html
    assert "connection kept dropping after" not in html
    # It gives up only after a long continuous outage, not after N attempts.
    assert "UPLOAD_STALL_LIMIT_MS = 5 * 60 * 1000" in html


def test_transfer_chunk_size_adapts_within_the_server_limit() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "function nextChunkSize(current, seconds)" in html
    assert "const UPLOAD_MAX_CHUNK = 8 * 1024 * 1024;" in html
    # The client must never propose a chunk the server would reject with 413.
    assert retake.UPLOAD_CHUNK_MAX_BYTES >= 8 * 1024 * 1024


def test_lan_urls_are_http_links() -> None:
    assert all(url.startswith("http://") and url.endswith(":8710") for url in lan_urls())


def test_chunk_upload_resumes_and_preserves_bytes() -> None:
    with temporary_project_root() as root:
        client = TestClient(app)
        name = f"retake-upload-test-{uuid.uuid4().hex}.bin"
        fingerprint = uuid.uuid4().hex
        start = client.post("/upload/start", json={"name": name, "size": 6, "fingerprint": fingerprint})
        assert start.status_code == 200
        session = start.json()["session"]
        first = client.post(
            f"/upload/chunk/{session}", content=b"abc", headers={"X-Upload-Offset": "0"}
        )
        assert first.json()["offset"] == 3
        resumed = client.post(
            "/upload/start", json={"name": name, "size": 6, "fingerprint": fingerprint}
        )
        assert resumed.json()["offset"] == 3
        wrong = client.post(
            f"/upload/chunk/{session}", content=b"x", headers={"X-Upload-Offset": "0"}
        )
        assert wrong.status_code == 409 and wrong.json()["offset"] == 3
        second = client.post(
            f"/upload/chunk/{session}", content=b"def", headers={"X-Upload-Offset": "3"}
        )
        assert second.json()["offset"] == 6
        done = client.post("/upload/finish", json={"session": session})
        assert done.status_code == 200
        output = Path(done.json()["path"])
        assert output.read_bytes() == b"abcdef"
        assert output == root / "Project 1" / "media" / "original.bin"
        assert (root / "Project 1" / "exports").is_dir()


def test_media_uses_native_http_range_delivery() -> None:
    old_current = dict(retake.CURRENT)
    with tempfile.TemporaryDirectory() as folder:
        media = Path(folder) / "range-test.mp4"
        media.write_bytes(b"0123456789")
        try:
            retake.CURRENT.update(
                project={"source_path": str(media)}, media_path=str(media),
                project_dir=folder, source_filename=media.name,
            )
            client = TestClient(app)
            whole = client.get("/media")
            assert whole.status_code == 200 and whole.content == b"0123456789"
            assert whole.headers["accept-ranges"] == "bytes"

            middle = client.get("/media", headers={"Range": "bytes=2-5"})
            assert middle.status_code == 206 and middle.content == b"2345"
            assert middle.headers["content-range"] == "bytes 2-5/10"

            suffix = client.get("/media", headers={"Range": "bytes=-3"})
            assert suffix.status_code == 206 and suffix.content == b"789"
            invalid = client.get("/media", headers={"Range": "bytes=99-"})
            assert invalid.status_code == 416

            head = client.head("/media")
            assert head.status_code == 200 and head.content == b""
            assert head.headers["content-length"] == "10"
        finally:
            retake.CURRENT.clear()
            retake.CURRENT.update(old_current)


def _write_ready_preview(source: Path, payload: bytes = b"preview-data") -> Path:
    final, _partial, metadata = retake.preview_proxy_paths(str(source))
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(payload)
    stat = final.stat()
    retake.atomic_write_json(metadata, {
        "source": retake.preview_source_identity(str(source)),
        "proxy_size": stat.st_size,
        "proxy_mtime_ns": stat.st_mtime_ns,
        "created_at": retake.utc_now(),
    })
    return final


def test_preview_paths_and_identity_are_project_confined() -> None:
    with temporary_project_root() as root:
        project_dir = retake.next_project_directory()
        source = project_dir / "media" / "original.mov"
        source.write_bytes(b"source")
        final, partial, metadata = retake.preview_proxy_paths(str(source))
        for path in (final, partial, metadata):
            path.resolve().relative_to(project_dir.resolve())
            assert path.parent == project_dir / "preview"
        before = retake.preview_source_identity(str(source))
        source.write_bytes(b"changed-source")
        after = retake.preview_source_identity(str(source))
        assert before != after
        assert before["version"] == retake.PREVIEW_PROXY_VERSION
        assert project_dir.parent == root


def test_preview_commands_use_correct_rotation_and_gpu_first_formats() -> None:
    properties = {
        "duration_s": 60.0, "vcodec": "h264", "acodec": "aac",
        "has_video": True, "fps": 60.0, "width": 3840, "height": 2160,
        "rotation": 90, "video_streams": 1, "audio_streams": 1,
    }
    old_ffmpeg, old_probe = retake.FFMPEG, retake.probe_media
    try:
        retake.FFMPEG = "ffmpeg"
        retake.probe_media = lambda _path: properties
        command = retake.preview_proxy_command(
            {"source_path": "phone.MOV", "probe": properties}, Path("proxy.mp4"),
            device="gpu",
        )
        joined = " ".join(command)
        assert "-hwaccel cuda -hwaccel_output_format cuda" in joined
        assert "-noautorotate -display_rotation 0" in joined
        assert "scale_cuda=w=1280:h=720:format=nv12" in joined
        assert "hwdownload,format=nv12,transpose=cclock" in joined
        assert "-c:v h264_nvenc" in joined and "-cq 24" in joined
        assert "-c:a aac -b:a 96k" in joined
        assert "-r 30 -fps_mode cfr" in joined
        assert "-g 15 -keyint_min 15" in joined
        assert "-pix_fmt yuv420p" in joined and "-movflags +faststart" in joined
        assert retake.preview_target_dimensions(properties) == (720, 1280)
        assert retake._preview_video_filter(
            {**properties, "rotation": 0}, 1280, 720, device="gpu"
        ).endswith("hwdownload,format=nv12")
        assert retake._preview_video_filter(
            {**properties, "rotation": 270}, 720, 1280, device="gpu"
        ).endswith("transpose=clock")

        cpu = retake.preview_proxy_command(
            {"source_path": "phone.MOV", "probe": properties},
            Path("proxy.mp4"), device="cpu",
        )
        cpu_joined = " ".join(cpu)
        assert "-hwaccel" not in cpu
        assert "scale=w=1280:h=720:flags=fast_bilinear" in cpu_joined
        assert "transpose=cclock" in cpu_joined
        assert "-c:v libx264 -preset veryfast -crf 25" in cpu_joined
        assert "-pix_fmt yuv420p" in cpu_joined
    finally:
        retake.FFMPEG, retake.probe_media = old_ffmpeg, old_probe


def test_preview_status_and_media_delivery_reject_stale_proxy() -> None:
    old_current = dict(retake.CURRENT)
    old_preview = dict(retake.PREVIEW_STATE)
    old_gpu = retake._PREVIEW_GPU_USABLE
    with temporary_project_root():
        project_dir = retake.next_project_directory()
        source = project_dir / "media" / "original.mov"
        source.write_bytes(b"source")
        project = {
            "source_path": str(source), "probe": {"has_video": True},
            "duration_s": 10.0,
        }
        try:
            retake.CURRENT.update(
                project=project, media_path=str(source), project_dir=str(project_dir),
                source_filename=source.name,
            )
            retake.PREVIEW_STATE.update(running=False, identity=None, error=None)
            retake._PREVIEW_GPU_USABLE = True
            client = TestClient(app)
            absent = client.get("/preview/status")
            assert absent.status_code == 200 and absent.json()["state"] == "absent"
            assert client.get("/preview/media").status_code == 404

            final = _write_ready_preview(source)
            ready = client.get("/preview/status").json()
            assert ready["state"] == "ready" and ready["size"] == final.stat().st_size
            middle = client.get("/preview/media", headers={"Range": "bytes=2-6"})
            assert middle.status_code == 206 and middle.content == b"eview"
            assert middle.headers["content-type"].startswith("video/mp4")
            head = client.head("/preview/media")
            assert head.status_code == 200 and head.content == b""

            source.write_bytes(b"source changed")
            assert client.get("/preview/status").json()["state"] == "stale"
            assert client.get("/preview/media").status_code == 404
        finally:
            retake.CURRENT.clear()
            retake.CURRENT.update(old_current)
            retake.PREVIEW_STATE.clear()
            retake.PREVIEW_STATE.update(old_preview)
            retake._PREVIEW_GPU_USABLE = old_gpu


def test_preview_create_allows_cpu_fallback_and_respects_heavy_job_lock() -> None:
    old_current = dict(retake.CURRENT)
    old_preview = dict(retake.PREVIEW_STATE)
    old_gpu = retake._PREVIEW_GPU_USABLE
    with temporary_project_root():
        project_dir = retake.next_project_directory()
        source = project_dir / "media" / "original.mov"
        source.write_bytes(b"source")
        try:
            retake.CURRENT.update(
                project={
                    "source_path": str(source), "probe": {"has_video": True},
                    "duration_s": 10.0,
                },
                media_path=str(source), project_dir=str(project_dir),
                source_filename=source.name,
            )
            retake.PREVIEW_STATE.update(running=False, identity=None, error=None)
            client = TestClient(app)
            retake._PREVIEW_GPU_USABLE = False
            assert retake.JOB_LOCK.acquire(blocking=False)
            try:
                busy = client.post("/preview/create")
                assert busy.status_code == 409
                assert busy.json()["error"] == "another job is running"
            finally:
                retake.JOB_LOCK.release()
        finally:
            retake.CURRENT.clear()
            retake.CURRENT.update(old_current)
            retake.PREVIEW_STATE.clear()
            retake.PREVIEW_STATE.update(old_preview)
            retake._PREVIEW_GPU_USABLE = old_gpu


def test_preview_job_publishes_atomically_after_validation() -> None:
    old_current = dict(retake.CURRENT)
    old_preview = dict(retake.PREVIEW_STATE)
    old_status = dict(retake.STATUS)
    old_command = retake.preview_proxy_command
    old_progress = retake._run_ffmpeg_with_progress
    old_integrity = retake.preview_proxy_integrity_errors
    with temporary_project_root():
        project_dir = retake.next_project_directory()
        source = project_dir / "media" / "original.mov"
        source.write_bytes(b"source")
        project = {
            "source_path": str(source), "duration_s": 10.0,
            "probe": {"has_video": True},
        }
        identity = retake.preview_source_identity(str(source))
        try:
            retake.CURRENT.update(
                project=project, media_path=str(source), project_dir=str(project_dir),
                source_filename=source.name,
            )
            retake.PREVIEW_STATE.update(running=True, identity=identity, error=None)
            retake.preview_proxy_command = (
                lambda _proj, out, device="gpu": ["fake-ffmpeg", device, str(out)]
            )

            def fake_progress(command, *_args, **_kwargs):
                Path(command[-1]).write_bytes(b"verified-proxy")

            retake._run_ffmpeg_with_progress = fake_progress
            retake.preview_proxy_integrity_errors = lambda *_: []
            assert retake.JOB_LOCK.acquire(blocking=False)
            retake.preview_proxy_job(project, identity)
            final, partial, _metadata = retake.preview_proxy_paths(str(source))
            assert final.read_bytes() == b"verified-proxy"
            assert not partial.exists()
            assert retake._preview_proxy_record(str(source)) is not None
            assert retake.PREVIEW_STATE["running"] is False
            assert retake.PREVIEW_STATE["device"] == "gpu"
            assert retake.PREVIEW_STATE["fallback"] is False
            assert retake.STATUS["phase"] == "ready"
        finally:
            if retake.JOB_LOCK.locked():
                retake.JOB_LOCK.release()
            retake.CURRENT.clear()
            retake.CURRENT.update(old_current)
            retake.PREVIEW_STATE.clear()
            retake.PREVIEW_STATE.update(old_preview)
            retake.STATUS.clear()
            retake.STATUS.update(old_status)
            retake.preview_proxy_command = old_command
            retake._run_ffmpeg_with_progress = old_progress
            retake.preview_proxy_integrity_errors = old_integrity


def test_preview_job_retries_once_on_cpu_after_gpu_failure() -> None:
    old_preview = dict(retake.PREVIEW_STATE)
    old_status = dict(retake.STATUS)
    old_gpu = retake._PREVIEW_GPU_USABLE
    old_command = retake.preview_proxy_command
    old_progress = retake._run_ffmpeg_with_progress
    old_integrity = retake.preview_proxy_integrity_errors
    with temporary_project_root():
        project_dir = retake.next_project_directory()
        source = project_dir / "media" / "original.mov"
        source.write_bytes(b"source")
        project = {
            "source_path": str(source), "duration_s": 10.0,
            "probe": {"has_video": True},
        }
        identity = retake.preview_source_identity(str(source))
        devices: list[str] = []
        try:
            retake._PREVIEW_GPU_USABLE = True
            retake.PREVIEW_STATE.update(
                running=True, identity=identity, error=None,
                device=None, fallback=False,
            )
            retake.preview_proxy_command = (
                lambda _proj, out, device="gpu": ["fake-ffmpeg", device, str(out)]
            )

            def fake_progress(command, *_args, **_kwargs):
                device = command[1]
                devices.append(device)
                if device == "gpu":
                    Path(command[-1]).write_bytes(b"failed-gpu-partial")
                    raise RuntimeError("simulated CUDA failure")
                Path(command[-1]).write_bytes(b"verified-cpu-proxy")

            retake._run_ffmpeg_with_progress = fake_progress
            retake.preview_proxy_integrity_errors = lambda *_: []
            assert retake.JOB_LOCK.acquire(blocking=False)
            retake.preview_proxy_job(project, identity)
            final, partial, metadata = retake.preview_proxy_paths(str(source))
            assert devices == ["gpu", "cpu"]
            assert final.read_bytes() == b"verified-cpu-proxy"
            assert not partial.exists()
            assert retake.PREVIEW_STATE["device"] == "cpu"
            assert retake.PREVIEW_STATE["fallback"] is True
            saved = json.loads(metadata.read_text(encoding="utf-8"))
            assert saved["device"] == "cpu"
            assert saved["fallback"] is True
        finally:
            if retake.JOB_LOCK.locked():
                retake.JOB_LOCK.release()
            retake.PREVIEW_STATE.clear()
            retake.PREVIEW_STATE.update(old_preview)
            retake.STATUS.clear()
            retake.STATUS.update(old_status)
            retake._PREVIEW_GPU_USABLE = old_gpu
            retake.preview_proxy_command = old_command
            retake._run_ffmpeg_with_progress = old_progress
            retake.preview_proxy_integrity_errors = old_integrity


def test_alignment_windows_are_bounded_and_padded() -> None:
    project = {
        "duration_s": 100.0,
        "segments": [
            {"start": 5.0, "end": 14.0, "text": "first section"},
            {"start": 14.2, "end": 25.0, "text": "second section"},
            {"start": 40.0, "end": 48.0, "text": "third section"},
        ],
    }
    windows = retake.alignment_segments_for_project(
        project, max_span=30.0, padding=2.5
    )
    assert windows == [
        {"start": 2.5, "end": 27.5,
         "text": "first section second section"},
        {"start": 37.5, "end": 50.5, "text": "third section"},
    ]


def test_aligned_timings_preserve_token_identity_and_reject_hallucinated_span() -> None:
    project = {
        "duration_s": 40.0,
        "tokens": [
            {"id": 1, "kind": "word", "text": "And",
             "start": 31.1, "end": 31.98, "seg": 1, "cut": True},
            {"id": 2, "kind": "word", "text": "if",
             "start": 31.98, "end": 32.86, "seg": 1, "cut": False},
            {"id": 3, "kind": "word", "text": "Cloud,",
             "start": 33.56, "end": 33.88, "seg": 1, "cut": False},
        ],
        "segments": [
            {"id": 1, "start": 31.1, "end": 33.88,
             "text": "And if Cloud,"},
        ],
        "markers": [{"t": 12.0, "note": "keep me"}],
    }
    result = {
        "device": "cuda", "model": "test-aligner", "elapsed_s": 1.0,
        "words": [
            {"word": "And", "start": 28.281, "end": 32.769, "score": 0.332},
            {"word": "if", "start": 32.789, "end": 32.889, "score": 0.845},
            {"word": "Cloud,", "start": 33.651, "end": 34.092, "score": 0.87},
        ],
    }
    old_coverage = retake.ALIGNMENT_MIN_COVERAGE
    try:
        retake.ALIGNMENT_MIN_COVERAGE = 0.6
        candidate, stats = retake.apply_aligned_word_timings(
            project, result, fallback=False
        )
    finally:
        retake.ALIGNMENT_MIN_COVERAGE = old_coverage
    assert [(t["id"], t["text"], t["cut"]) for t in candidate["tokens"]] == [
        (1, "And", True), (2, "if", False), (3, "Cloud,", False),
    ]
    assert (candidate["tokens"][0]["start"], candidate["tokens"][0]["end"]) == (
        31.1, 31.98
    )
    assert (candidate["tokens"][1]["start"], candidate["tokens"][1]["end"]) == (
        32.789, 32.889
    )
    assert candidate["markers"] == project["markers"]
    assert stats["device"] == "cuda" and stats["coverage"] == 0.666667


def test_alignment_recalibrates_legacy_gap_without_cutting_kept_words() -> None:
    project = {
        "duration_s": 4.0,
        "tokens": [
            {"id": 1, "kind": "word", "text": "before",
             "start": 1.0, "end": 1.4, "seg": 1, "cut": False},
            {"id": 2, "kind": "gap", "text": "…",
             "start": 1.4, "end": 2.8, "cut": True},
            {"id": 3, "kind": "word", "text": "after",
             "start": 2.8, "end": 3.1, "seg": 1, "cut": False},
        ],
        "segments": [
            {"id": 1, "start": 1.0, "end": 3.1, "text": "before after"},
        ],
    }
    result = {
        "device": "cuda", "model": "test", "elapsed_s": 0.1,
        "words": [
            {"word": "before", "start": 1.1, "end": 1.65, "score": 0.9},
            {"word": "after", "start": 2.1, "end": 2.5, "score": 0.9},
        ],
    }
    candidate, _stats = retake.apply_aligned_word_timings(
        project, result, fallback=False
    )
    gap = candidate["tokens"][1]
    assert (gap["id"], gap["cut"]) == (2, True)
    assert (gap["start"], gap["end"]) == (1.71, 2.04)
    assert retake.transcript_cut_intervals(
        candidate["tokens"], retake.WORD_CUT_SAFETY_S
    ) == [(1.71, 2.04)]


def test_forced_alignment_attempts_gpu_then_cpu() -> None:
    old_call = retake.alignment_worker_call
    old_state = dict(retake.ALIGNMENT_STATE)
    devices: list[str] = []
    try:
        def fake_call(_proj, device):
            devices.append(device)
            if device == "cuda":
                raise RuntimeError("simulated CUDA memory error")
            return {"device": "cpu", "words": []}

        retake.alignment_worker_call = fake_call
        result, fallback = retake.run_forced_alignment({})
        assert devices == ["cuda", "cpu"]
        assert result["device"] == "cpu" and fallback is True
    finally:
        retake.alignment_worker_call = old_call
        retake.ALIGNMENT_STATE.clear()
        retake.ALIGNMENT_STATE.update(old_state)


def test_aligned_project_cut_boundaries_protect_adjacent_kept_words() -> None:
    tokens = [
        {"id": 1, "kind": "word", "text": "Wikipedia",
         "start": 0.0, "end": 1.0, "cut": False},
        {"id": 2, "kind": "word", "text": "remove",
         "start": 1.0, "end": 1.8, "cut": True},
        {"id": 3, "kind": "word", "text": "this",
         "start": 1.8, "end": 2.1, "cut": True},
        {"id": 4, "kind": "word", "text": "Next",
         "start": 2.0, "end": 2.5, "cut": False},
    ]
    project = {
        "tokens": tokens, "audio_gaps": [],
        "alignment": {
            "status": "aligned", "version": retake.ALIGNMENT_VERSION,
            "coverage": 1.0,
        },
    }
    assert retake.consecutive_word_cut_intervals(tokens) == [(1.0, 2.1)]
    assert retake.cut_intervals_from_tokens(project) == [
        (1.06, 1.94)
    ]


def test_alignment_job_backs_up_and_publishes_without_changing_edits() -> None:
    old_current = dict(retake.CURRENT)
    old_state = dict(retake.ALIGNMENT_STATE)
    old_status = dict(retake.STATUS)
    old_runner = retake.run_forced_alignment
    with temporary_project_root():
        project_dir = retake.next_project_directory()
        source = project_dir / "media" / "original.wav"
        source.write_bytes(b"source-audio")
        project = {
            "schema_version": 1, "source_path": str(source),
            "duration_s": 5.0, "language": "en",
            "probe": {"acodec": "pcm_s16le", "audio_streams": 1},
            "tokens": [
                {"id": 1, "kind": "word", "text": "keep",
                 "start": 1.0, "end": 1.3, "seg": 1, "cut": False},
                {"id": 2, "kind": "word", "text": "remove",
                 "start": 1.3, "end": 1.8, "seg": 1, "cut": True},
                {"id": 3, "kind": "word", "text": "next",
                 "start": 1.8, "end": 2.1, "seg": 1, "cut": False},
            ],
            "segments": [
                {"id": 1, "start": 1.0, "end": 2.1,
                 "text": "keep remove next"},
            ],
            "markers": [{"t": 4.0, "note": "unchanged"}],
            "audio_gaps": [],
        }
        retake.atomic_write_json(project_dir / "project.json", project)
        identity = retake.alignment_source_identity(project)
        result = {
            "device": "cuda", "model": "test", "elapsed_s": 0.2,
            "words": [
                {"word": "keep", "start": 1.1, "end": 1.4, "score": 0.9},
                {"word": "remove", "start": 1.5, "end": 1.9, "score": 0.9},
                {"word": "next", "start": 2.0, "end": 2.3, "score": 0.9},
            ],
        }
        try:
            retake.CURRENT.update(
                project=project, media_path=str(source),
                project_dir=str(project_dir), source_filename=source.name,
            )
            retake.ALIGNMENT_STATE.update(
                running=True, identity=identity, error=None,
                device="cuda", fallback=False, coverage=None,
            )
            retake.run_forced_alignment = lambda _proj: (result, False)
            assert retake.JOB_LOCK.acquire(blocking=False)
            retake.recalibrate_alignment_job(project, identity)
            saved = json.loads(
                (project_dir / "project.json").read_text(encoding="utf-8")
            )
            assert [(t["id"], t["cut"]) for t in saved["tokens"]] == [
                (1, False), (2, True), (3, False),
            ]
            assert saved["tokens"][0]["start"] == 1.1
            assert saved["markers"] == project["markers"]
            assert saved["alignment"]["device"] == "cuda"
            assert list(project_dir.glob("project.alignment-backup-*.json"))
            assert retake.ALIGNMENT_STATE["coverage"] == 1.0
        finally:
            if retake.JOB_LOCK.locked():
                retake.JOB_LOCK.release()
            retake.CURRENT.clear()
            retake.CURRENT.update(old_current)
            retake.ALIGNMENT_STATE.clear()
            retake.ALIGNMENT_STATE.update(old_state)
            retake.STATUS.clear()
            retake.STATUS.update(old_status)
            retake.run_forced_alignment = old_runner


def test_future_transcription_enables_vad_and_forced_alignment_contract() -> None:
    source = Path(retake.__file__).read_text(encoding="utf-8")
    assert "vad_filter=True" in source
    assert 'vad_parameters={"min_silence_duration_ms": 200}' in source
    assert "condition_on_previous_text=False" in source
    assert "run_forced_alignment(" in source


def test_real_project_folders_increment_without_hashes() -> None:
    with temporary_project_root() as root:
        one = retake.next_project_directory()
        two = retake.next_project_directory()
        assert one == root / "Project 1"
        assert two == root / "Project 2"
        assert retake.project_directories() == [one, two]


def test_generic_editor_prompt_has_no_explicit_delete_target() -> None:
    default = (
        "This is an improvised video about cameras. Perform a conservative takes edit: "
        "discard false starts and weaker bad takes. Remove obvious filler and trim awkward dead air."
    )
    assert explicit_target_terms(default) == set()
    assert "sponsor" in explicit_target_terms("Delete the sponsor message.")


def test_budget_guard_refuses_an_implausibly_broad_automated_edit() -> None:
    """The guard that stops one confident mistake from deleting the recording."""
    segments = [
        {"id": i, "start": float(i * 5), "end": float(i * 5 + 4),
         "text": f"unique substantive section number {i}"}
        for i in range(20)
    ]
    broad = [
        {"sentence_ids": [s["id"]], "start": s["start"], "end": s["end"]}
        for s in segments
    ]
    assert not enforce_llm_budget(broad, segments, 100.0)
    assert enforce_llm_budget(broad[:2], segments, 100.0)
    assert enforce_llm_budget([], segments, 100.0)


def test_preview_has_single_flight_seek_controller() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "const skipState" in html
    assert "setInterval(skipTick" not in html
    assert 'player.addEventListener("seeked", finishCutSeek)' in html
    assert "Preview could not decode the next kept frame" not in html
    assert "function armCutSeekRecovery()" in html
    assert "skipState.retries < 2" in html
    assert "skipState.wasPlaying = !player.paused; skipState.retries = 0;\n  player.currentTime = target;" in html


def test_gpu_smooth_preview_ui_only_switches_backend_media_sources() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'id="smoothPreview"' in html
    assert 'await api("/preview/create", {})' in html
    assert 'api("/preview/status")' in html
    assert 'useProxy ? "/preview/media" : "/media"' in html
    assert "function switchPlayerMedia(useProxy, preserveState = true)" in html
    assert "muted: player.muted, volume: player.volume, rate: player.playbackRate" in html
    assert "player.currentTime = Math.min(saved.time" in html
    assert "Smooth Preview could not play, so the original preview was restored." in html
    assert 'preview:"Smooth Preview"' in html


def test_recalibrate_timing_ui_uses_backend_alignment_status() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'id="recalibrateTiming"' in html
    assert 'await api("/alignment/recalibrate", {})' in html
    assert 'api("/alignment/status")' in html
    assert "GPU first, CPU fallback" in html
    assert 'align:"Calibrating timing"' in html


def test_safe_api_calls_retry_but_side_effecting_jobs_do_not() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'new Set(["/cuts","/gaps","/markers","/voice-enhancement"])' in html
    assert "const attempts = safeToRetry ? 3 : 1" in html
    assert "/failed to fetch|networkerror/i" in html


def test_editor_page_is_never_stale_after_backend_restart() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, max-age=0"


def test_windows_server_and_media_streaming_are_disconnect_resilient() -> None:
    source = Path(retake.__file__).read_text(encoding="utf-8")
    media_start = source.index('def media(request: Request)')
    media_end = source.index('@app.post("/cuts")', media_start)
    media_source = source[media_start:media_end]
    assert "FileResponse(" in media_source
    assert "StreamingResponse" not in media_source
    assert "WindowsSelectorEventLoopPolicy" in source


def test_fullscreen_and_ai_navigation_use_exact_visible_targets() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "#player:fullscreen" in html and "height:100vh!important" in html
    assert "function navigateProposal(ids, forceSeek = false)" in html
    assert 'first?.scrollIntoView({block:"center",behavior:"smooth"})' in html


def test_export_gap_review_and_voice_enhancement_are_available() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    for restored in (
        'id="gapToggle"', 'id="gapPreview"', 'id="gapWave"',
        'id="gapStart"', 'id="gapEnd"', 'api("/gaps',
    ):
        assert restored in html
    assert "export-only cut layer" in html
    assert "Playing the exact gap-cleaned cut used for export." in html
    assert 'id="enhanceEnabled" type="checkbox" checked' in html
    assert "Voice Enhancement" in html and "Reset to Great" in html
    for control in ("enhanceLoudness", "enhanceLeveling", "enhanceNoise", "enhanceDetail"):
        assert f'id="{control}" type="range"' in html
    source = Path(retake.__file__).read_text(encoding="utf-8")
    assert '@app.post("/gaps")' in source and '@app.get("/waveform")' in source
    assert "return merge_intervals(transcript_cuts + gap_cuts)" in source


def test_audio_gap_state_migration_preserves_transcript_timing() -> None:
    proj = {"tokens": [{"id": 0, "start": 1.0, "end": 2.0}]}
    before = json.loads(json.dumps(proj["tokens"]))
    assert ensure_audio_gap_state(proj)
    assert proj["audio_gaps"] == [] and not proj["audio_gaps_analyzed"]
    assert proj["tokens"] == before
    assert not ensure_audio_gap_state(proj)


def test_default_and_legacy_smart_modes_use_reliable_export() -> None:
    source = Path(retake.__file__).read_text(encoding="utf-8")
    start = source.index("def export_job")
    end = source.index("def export_text", start)
    export_job_source = source[start:end]
    assert "_reliable_video_export(" in export_job_source
    assert "_smartcut_export(" not in export_job_source
    assert 'compatibility = mode == "reencode"' in export_job_source


def test_reliable_export_command_enforces_continuous_mp4_timing() -> None:
    proj = {
        "source_path": "input.MOV",
        "probe": {"fps": 30.0, "acodec": "aac"},
    }
    command = retake._reliable_video_command(
        proj, [(1.0, 2.0), (4.0, 5.5)], Path("output.mp4"), "h264_nvenc"
    )
    joined = " ".join(command)
    assert "h264_nvenc" in command and "-cq" in command and "18" in command
    assert "-fps_mode cfr" in joined
    assert "-r 30" in joined
    assert "-video_track_timescale 30000" in joined
    assert "-movflags +faststart" in joined
    assert "setpts=N/(30*TB),fps=30" in joined
    assert "-ss 1.000000 -t 4.500000" in joined
    assert "-noautorotate -display_rotation 0" in joined
    assert "atrim=start=0.000000:end=1.000000" in joined
    assert "aresample=async=1:first_pts=0" in joined


def test_reliable_export_reprobes_and_normalizes_phone_orientation() -> None:
    saved_properties = {
        "fps": 30.0, "acodec": "aac", "sample_rate": 48000, "channels": 1,
        "width": 3840, "height": 2160,
    }
    actual_properties = {
        **saved_properties, "rotation": 90, "video_streams": 1, "audio_streams": 1,
    }
    proj = {"source_path": "phone.MOV", "probe": saved_properties}
    captured: list[list[str]] = []
    old_probe = retake.probe_media
    old_encoder_usable = retake._ffmpeg_encoder_usable
    old_run = retake._run_ffmpeg_with_progress
    try:
        retake.probe_media = lambda _path: actual_properties
        retake._ffmpeg_encoder_usable = lambda _encoder: False
        retake._run_ffmpeg_with_progress = (
            lambda command, *_args, **_kwargs: captured.append(command)
        )
        assert retake._reliable_video_export(
            proj, [(0.0, 2.0)], Path("output.mp4")
        ) == "CPU"
        joined = " ".join(captured[0])
        assert "transpose=cclock" in joined
        assert "-noautorotate -display_rotation 0" in joined
        assert "-metadata:s:v:0 rotate=0" in joined
        assert "setpts=N/(30*TB),transpose=cclock,fps=30" in joined
    finally:
        retake.probe_media = old_probe
        retake._ffmpeg_encoder_usable = old_encoder_usable
        retake._run_ffmpeg_with_progress = old_run


def test_reliable_export_maps_prepared_audio_with_source_properties() -> None:
    proj = {
        "source_path": "input.MOV",
        "probe": {"fps": 30.0, "acodec": "aac", "sample_rate": 48000, "channels": 2},
    }
    command = retake._reliable_video_command(
        proj, [(1.0, 2.0), (4.0, 5.5)], Path("output.mp4"), "libx264",
        audio_path=Path("enhanced.wav"),
    )
    joined = " ".join(command)
    assert "-i input.MOV -i enhanced.wav" in joined
    assert "[1:a:0]asetpts=PTS-STARTPTS,aresample=48000:async=1:first_pts=0[a]" in joined
    assert "-b:a 256k -ar 48000 -ac 2" in joined
    assert "[0:a:0]atrim=" not in joined


def test_enhancement_filter_order_keeps_loudness_independent_from_cleanup() -> None:
    source = Path(retake.__file__).read_text(encoding="utf-8")
    prepare_start = source.index("def prepare_export_audio")
    prepare_end = source.index("def _ffmpeg_audio_cut", prepare_start)
    prepare = source[prepare_start:prepare_end]
    assert prepare.index("_run_deepfilter_worker") < prepare.index("_normalize_audio_stem")
    normalize_start = source.index("def _normalize_audio_stem")
    normalize_end = source.index("def prepare_export_audio", normalize_start)
    normalize = source[normalize_start:normalize_end]
    assert "leveling = _leveling_filter" in normalize
    assert 'f"{leveling},loudnorm=I=' in normalize and "alimiter=limit=" in normalize
    assert "aresample={sample_rate}" in normalize


def test_export_refinement_runs_after_keep_composition_without_text_export_changes() -> None:
    source = Path(retake.__file__).read_text(encoding="utf-8")
    start = source.index("def export_job")
    end = source.index("def export_text", start)
    media_export = source[start:end]
    assert media_export.index("keep_list(cuts") < media_export.index("refine_export_keep_edges")
    text_export = source[end:source.index("def fully_cut_segments", end)]
    assert "refine_export_keep_edges" not in text_export


def test_display_dimensions_apply_phone_rotation() -> None:
    assert retake.display_dimensions(
        {"width": 3840, "height": 2160, "rotation": 90}
    ) == (2160, 3840)
    assert retake.display_dimensions(
        {"width": 3840, "height": 2160, "rotation": 0}
    ) == (3840, 2160)
    assert retake._orientation_normalization_filters({"rotation": 90}) == [
        "transpose=cclock"
    ]
    assert retake._orientation_normalization_filters({"rotation": 270}) == [
        "transpose=clock"
    ]
    assert retake._orientation_normalization_filters({"rotation": 180}) == [
        "hflip", "vflip"
    ]


def test_video_timeline_validation_rejects_smartcut_style_gaps() -> None:
    old_ffprobe, old_run = retake.FFPROBE, retake._run
    try:
        retake.FFPROBE = "ffprobe"
        retake._run = lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0,
            stdout="-0.033333,0.033333\n0.000000,0.033333\n0.033333,0.033333\n",
            stderr="",
        )
        assert retake.validate_video_timeline("good.mp4", 30.0, 0.1) == []

        retake._run = lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0,
            stdout="0.000000,0.001667\n0.100000,0.001667\n0.101667,0.001667\n",
            stderr="",
        )
        errors = retake.validate_video_timeline("bad.mov", 30.0, 0.135)
        assert any("irregular frame interval" in error for error in errors)
        assert any("incorrect packet duration" in error for error in errors)
    finally:
        retake.FFPROBE, retake._run = old_ffprobe, old_run


def test_latest_media_export_is_discovered_and_downloaded_safely() -> None:
    old_current = dict(retake.CURRENT)
    with temporary_project_root() as root:
        project_dir = retake.next_project_directory()
        source = project_dir / "media" / "original.MOV"
        source.write_bytes(b"source")
        exports = project_dir / "exports"
        older = exports / "original.retake_cut.MOV"
        newer = exports / "original.retake_cut.1.mp4"
        older.write_bytes(b"older")
        newer.write_bytes(b"newest")
        (exports / "original.transcript.txt").write_text("not media", encoding="utf-8")
        (exports / "other.retake_cut.mp4").write_bytes(b"not this project export")
        os.utime(older, ns=(1_000_000_000, 1_000_000_000))
        os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
        try:
            retake.CURRENT.update(
                project={"source_path": str(source)},
                media_path=str(source),
                project_dir=str(project_dir),
                source_filename="video.mp4",
            )
            assert retake.latest_media_export() == newer.resolve()
            client = TestClient(app)
            latest = client.get("/export/latest")
            assert latest.status_code == 200
            assert latest.json() == {"name": newer.name, "size": len(b"newest")}
            downloaded = client.get("/export/latest/download")
            assert downloaded.status_code == 200 and downloaded.content == b"newest"
            assert "attachment" in downloaded.headers["content-disposition"]
            assert newer.name in downloaded.headers["content-disposition"]
            assert downloaded.headers["cache-control"] == "no-store, max-age=0"
        finally:
            retake.CURRENT.clear()
            retake.CURRENT.update(old_current)


def test_latest_export_reports_when_a_project_or_export_is_missing() -> None:
    old_current = dict(retake.CURRENT)
    try:
        retake.CURRENT.update(
            project=None, media_path=None, project_dir=None, source_filename=None,
        )
        client = TestClient(app)
        assert client.get("/export/latest").status_code == 404
        with temporary_project_root():
            project_dir = retake.next_project_directory()
            source = project_dir / "media" / "original.wav"
            source.write_bytes(b"source")
            retake.CURRENT.update(
                project={"source_path": str(source)},
                media_path=str(source),
                project_dir=str(project_dir),
                source_filename="audio.wav",
            )
            response = client.get("/export/latest")
            assert response.status_code == 404
            assert response.json()["error"] == "No media export found. Please export first."
    finally:
        retake.CURRENT.clear()
        retake.CURRENT.update(old_current)


def test_download_latest_export_tile_uses_direct_browser_transfer() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'id="expDownload"' in html
    assert "Download Latest Export" in html
    assert "Reliable Quality" in html
    assert 'bindExport("expSmart", "reliable")' in html
    assert 'await api("/export/latest")' in html
    assert 'window.location.assign("/export/latest/download")' in html


def word_tokens(words: list[tuple[str, float, float]]) -> list[dict]:
    return [
        {"id": index, "kind": "word", "text": text, "start": start, "end": end,
         "seg": 0, "cut": False}
        for index, (text, start, end) in enumerate(words)
    ]


def test_detailed_mode_requires_concrete_quotes_times_or_clusters() -> None:
    assert not is_detailed_edit_request("Remove bad takes and awkward filler")
    assert is_detailed_edit_request('شيل: "the wrong take"')
    assert is_detailed_edit_request('remove phrase "wrong take" at 00:05')
    assert is_detailed_edit_request("Cluster 2 — الحكومة (00:58.9 → 01:15.5)")


def test_timestamp_parser_handles_mixed_formats_and_arabic_digits() -> None:
    assert timestamps_in_text("00:58.9 → 01:15.5") == [58.9, 75.5]
    assert timestamps_in_text("٠٢:١٥ → ٠٢:١٩") == [135.0, 139.0]
    assert timestamps_in_text("12 days inside 20 companies") == []


def test_exact_phrase_resolution_never_expands_to_sentence() -> None:
    tokens = word_tokens([
        ("keep", 0.0, 0.3), ("remove", 0.3, 0.6), ("only", 0.6, 0.9),
        ("these", 0.9, 1.2), ("keep", 1.2, 1.5),
    ])
    match = resolve_phrase_tokens(tokens, "remove only these", 0.0, 2.0)
    assert match["status"] == "exact"
    assert match["token_ids"] == [1, 2, 3]


def test_timestamp_is_a_hard_boundary_for_repeated_phrase() -> None:
    tokens = word_tokens([
        ("remove", 4.8, 5.0), ("this", 5.0, 5.2),
        ("remove", 424.8, 425.0), ("this", 425.0, 425.2),
    ])
    near_start = resolve_phrase_tokens(tokens, "remove this", 2.0, 8.0)
    assert near_start["status"] == "exact" and near_start["token_ids"] == [0, 1]
    assert 2 not in near_start["token_ids"] and 3 not in near_start["token_ids"]


def test_all_matches_and_ordinal_matches_are_explicit() -> None:
    tokens = word_tokens([
        ("you", 1.0, 1.2), ("you", 2.0, 2.2), ("you", 3.0, 3.2),
    ])
    all_result = resolve_phrase_tokens(tokens, "you", 0.0, 4.0, all_matches=True)
    first_result = resolve_phrase_tokens(tokens, "you", 0.0, 4.0, occurrence="first")
    last_result = resolve_phrase_tokens(tokens, "you", 0.0, 4.0, occurrence="last")
    assert all_result["token_ids"] == [0, 1, 2]
    assert first_result["token_ids"] == [0]
    assert last_result["token_ids"] == [2]


def test_common_mixed_language_edit_list_uses_fast_deterministic_plan() -> None:
    instructions = (
        'Cluster 0 (00:00 → 00:08)\n'
        'شيل كله: "test one two"\n'
        'شيل: gap 00:08 → 00:12.3\n'
        'خلّي: "and you publish it"\n'
        'شيل: كل الـ "you" من 08:31.5 للآخر'
    )
    raw, covered = deterministic_instruction_plan(instructions)
    assert covered == {2, 3, 4, 5}
    assert [item["action"] for item in raw] == [
        "cut_phrase", "cut_gap", "keep_phrase", "cut_phrase"
    ]
    assert raw[-1]["all_matches"] is True


def test_keep_operations_block_overlapping_cut_operations() -> None:
    tokens = word_tokens([
        ("and", 1.0, 1.2), ("you", 1.2, 1.4), ("publish", 1.4, 1.8), ("it", 1.8, 2.0),
    ])
    operations = [
        {"operation_id": 1, "line": 1, "action": "cut_phrase", "phrase": "you publish",
         "start": 0.0, "end": 3.0, "instruction": "cut you publish", "reason": "cut"},
        {"operation_id": 2, "line": 2, "action": "keep_phrase", "phrase": "and you publish it",
         "start": 0.0, "end": 3.0, "instruction": "keep and you publish it", "reason": "keep"},
    ]
    results = resolve_detailed_operations(operations, tokens)
    assert results[0]["status"] == "conflict" and results[0]["token_ids"] == []
    assert results[1]["status"] == "exact" and results[1]["token_ids"] == [0, 1, 2, 3]


def test_gap_resolution_returns_only_gap_tokens() -> None:
    tokens = word_tokens([("before", 0.0, 0.5)])
    tokens.extend([
        {"id": 1, "kind": "gap", "start": 8.04, "end": 12.28, "cut": False},
        {"id": 2, "kind": "word", "text": "after", "start": 12.28, "end": 12.6,
         "seg": 1, "cut": False},
    ])
    operation = {"operation_id": 1, "line": 1, "action": "cut_gap", "phrase": "",
                 "start": 8.0, "end": 12.3, "instruction": "cut gap", "reason": "gap"}
    result = resolve_detailed_operations([operation], tokens)[0]
    assert result["status"] == "exact" and result["token_ids"] == [1]


def test_planner_validation_preserves_original_instruction_and_reports_omissions() -> None:
    instructions = 'Cluster 0 (00:00 → 00:08)\nشيل: "wrong take"\nخلّي: "right take"'
    parsed = {"operations": [
        {"line": 2, "action": "cut_phrase", "phrase": "wrong take",
         "start": None, "end": None, "reason": "remove"},
    ]}
    operations, unresolved = validate_planned_operations(parsed, instructions, 20.0)
    assert operations[0]["instruction"] == 'شيل: "wrong take"'
    assert (operations[0]["start"], operations[0]["end"]) == (0.0, 8.0)
    assert unresolved[0]["instruction"] == 'خلّي: "right take"'
    assert unresolved[0]["status"] == "not_found"


def test_planner_can_resolve_an_explicit_numbered_block_reference() -> None:
    instructions = (
        'بلوك 24: "it stayed for two weeks" ×3\n'
        'في الحالتين شيل بلوك 24 بالكامل.'
    )
    parsed = {"operations": [{
        "line": 2, "action": "cut_phrase", "phrase": "it stayed for two weeks",
        "start": None, "end": None, "all_matches": True, "reason": "remove block 24",
    }]}
    operations, unresolved = validate_planned_operations(parsed, instructions, 100.0)
    assert not unresolved
    assert operations[0]["instruction"] == "في الحالتين شيل بلوك 24 بالكامل."
    assert operations[0]["phrase"] == "it stayed for two weeks"


def test_from_timestamp_to_end_uses_inherited_cluster_end() -> None:
    instructions = (
        'Cluster 11 (08:16.9 → 08:43.5)\n'
        'شيل: كل الـ "you" من 08:31.5 للآخر'
    )
    raw, _ = deterministic_instruction_plan(instructions)
    operations, unresolved = validate_planned_operations(
        {"operations": raw}, instructions, 600.0
    )
    assert not unresolved
    assert (operations[0]["start"], operations[0]["end"]) == (511.5, 523.5)


def test_ordered_longer_followup_disambiguates_prefix_take() -> None:
    tokens = word_tokens([
        ("this", 1.0, 1.1), ("happened", 1.1, 1.3),
        ("this", 2.0, 2.1), ("happened", 2.1, 2.3), ("again", 2.3, 2.5),
        ("keep", 3.1, 3.2), ("these", 3.2, 3.3), ("words", 3.3, 3.4),
    ])
    operations = [
        {"operation_id": 1, "line": 1, "action": "cut_phrase", "phrase": "this happened",
         "start": 0.0, "end": 3.0, "instruction": "cut this happened", "reason": "first"},
        {"operation_id": 2, "line": 2, "action": "cut_phrase", "phrase": "this happened again",
         "start": 0.0, "end": 3.0, "instruction": "cut this happened again", "reason": "second"},
    ]
    results = resolve_detailed_operations(operations, tokens)
    assert results[0]["status"] == "exact" and results[0]["token_ids"] == [0, 1]
    assert results[1]["status"] == "exact" and results[1]["token_ids"] == [2, 3, 4]


def test_frontend_accepts_exact_token_ids_not_sentences() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "function proposalTokenIds(p)" in html
    assert "selected.forEach((p) => ids.push(...proposalTokenIds(p)))" in html
    assert "p.sentence_ids.forEach((s) => ids.push(...segAndInnerGaps(s)))" not in html


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
