"""Unit tests for the cut-interval math in retake.py.

Run with:  python test_retake.py   (or pytest test_retake.py)
"""
from __future__ import annotations

import os
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient
import retake

from retake import (
    build_ai_chunks,
    clusters_from_similarity,
    deterministic_retake_proposals,
    deterministic_instruction_plan,
    cut_intervals_from_tokens,
    ensure_audio_gap_state,
    gap_bulk_cut_bounds,
    keep_list,
    merge_intervals,
    parse_silencedetect_output,
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
    validate_audio_gaps,
    validate_llm_proposal,
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


def test_audio_gap_cuts_merge_with_whisper_token_cuts_without_rewriting_words() -> None:
    proj = {
        "tokens": [{"id": 0, "kind": "word", "text": "hello", "start": 1.0,
                    "end": 2.0, "seg": 0, "cut": True}],
        "audio_gaps": [{"id": "agap-1", "start": 1.8, "end": 3.0, "cut": True}],
    }
    assert approx(cut_intervals_from_tokens(proj), [(1.0, 3.0)])
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
            assert client.post("/cuts", json={"cut_ids": [0]}).json()[
                "transcript_cut_intervals"
            ] == [[.1, .3]]

            joined = client.post("/cuts", json={"cut_ids": [0, 2, 3]}).json()
            assert joined["word_cut_intervals"] == [[.1, 2.5]]
            assert joined["transcript_cut_intervals"] == [[.1, 2.5]]
            assert joined["scoped_gap_candidate_ids"] == []
            assert client.get("/project").json()["cut_intervals"] == [[.1, 2.5]]

            restored = client.post("/cuts", json={"cut_ids": [0, 3]}).json()
            assert restored["transcript_cut_intervals"] == [[.1, .3], [2.2, 2.5]]
            assert restored["scoped_gap_candidate_ids"] == ["agap-seg-1"]
            saved = (project_dir / "project.json").read_text(encoding="utf-8")
            assert "transcript_cut_intervals" not in saved and "word_cut_intervals" not in saved
            assert "scoped_gap_candidate_ids" not in saved
        finally:
            retake.CURRENT.clear()
            retake.CURRENT.update(old_current)


def test_preview_consumes_backend_composed_transcript_intervals() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "transcript_cut_intervals" in html
    assert "S.transcriptCutIvs = (result.transcript_cut_intervals || [])" in html
    assert "const ivs = S.cutsSynced ? S.transcriptCutIvs" in html


def test_scoped_gap_candidates_exclude_fully_deleted_sentences_only() -> None:
    proj = {
        "tokens": [
            {"id": 0, "kind": "word", "start": 0.0, "end": .5, "seg": 0, "cut": True},
            {"id": 1, "kind": "word", "start": 1.0, "end": 1.5, "seg": 0, "cut": True},
            {"id": 2, "kind": "word", "start": 2.0, "end": 2.5, "seg": 1, "cut": True},
            {"id": 3, "kind": "word", "start": 3.0, "end": 3.2, "seg": 2, "cut": False},
            {"id": 4, "kind": "word", "start": 3.4, "end": 3.6, "seg": 2, "cut": True},
            {"id": 5, "kind": "word", "start": 3.8, "end": 4.0, "seg": 2, "cut": True},
        ],
        "audio_gaps": [
            {"id": "deleted-sentence", "detected_start": .6, "detected_end": .9},
            {"id": "deleted-boundary", "detected_start": 1.6, "detected_end": 1.9},
            {"id": "kept-boundary", "detected_start": 2.6, "detected_end": 2.9},
            {"id": "partial-sentence", "detected_start": 3.65, "detected_end": 3.75},
        ],
    }
    assert retake.scoped_gap_candidate_ids(proj) == [
        "kept-boundary", "partial-sentence"
    ]


def test_scoped_gap_eligibility_is_derived_without_mutating_projects() -> None:
    proj = {
        "tokens": [
            {"id": 0, "kind": "word", "start": 1.0, "end": 1.4,
             "seg": 0, "cut": False},
        ],
        "audio_gaps": [
            {"id": "agap-1", "detected_start": 1.1, "detected_end": 1.3,
             "start": 1.1, "end": 1.3, "cut": False},
        ],
    }
    payload = retake.project_response_payload(proj)
    assert payload["scoped_gap_candidate_ids"] == ["agap-1"]
    assert "scoped_gap_candidate_ids" not in proj


def test_show_gaps_has_separate_scoped_action_and_keeps_remove_all() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'id="gapApply"' in html and ">Remove all gaps</button>" in html
    assert 'id="gapScoped"' in html and "Remove gaps in kept sentences" in html
    assert '$("gapScoped").hidden = !S.showGaps' in html
    assert "S.scopedGapIds.has(gap.id) && bulkBounds(gap)" in html
    assert '$("gapScoped").addEventListener("click"' in html


def test_bulk_gap_cut_keeps_natural_pause_evenly() -> None:
    gap = {"detected_start": 8.0, "detected_end": 10.0}
    assert gap_bulk_cut_bounds(gap, .12) == (8.06, 9.94)
    assert gap_bulk_cut_bounds(gap, 2.0) is None


def test_audio_gap_validation_keeps_stable_detector_ids() -> None:
    existing = [{"id": "agap-1", "detected_start": 1.0, "detected_end": 2.0,
                 "start": 1.0, "end": 2.0, "cut": False, "manual": False}]
    saved = validate_audio_gaps(
        [{"id": "agap-1", "start": 1.1, "end": 1.9, "cut": True, "manual": True},
         {"id": "invented", "start": 4, "end": 5, "cut": True}],
        existing, 10.0,
    )
    assert len(saved) == 1 and saved[0]["id"] == "agap-1"
    assert saved[0]["cut"] is True and saved[0]["manual"] is True


def test_old_projects_gain_empty_optional_gap_state() -> None:
    proj = {"tokens": []}
    assert ensure_audio_gap_state(proj)
    assert proj["audio_gaps"] == [] and proj["audio_gaps_analyzed"] is False
    assert not ensure_audio_gap_state(proj)


def test_ai_chunks_never_split_clusters() -> None:
    segments = [{"id": i, "text": "word " * 900} for i in range(6)]
    clusters = [{"id": 0, "members": [1, 3]}]  # spans indices 1..3
    chunks = build_ai_chunks(segments, clusters, max_words=2000)
    # indices 1,2,3 must land in exactly one chunk together
    holding = [c for c in chunks if 1 in c]
    assert len(holding) == 1 and {1, 2, 3} <= set(holding[0])
    # order preserved, everything covered exactly once
    flat = [i for c in chunks for i in c]
    assert flat == sorted(flat) == list(range(6))


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


def test_ai_rejects_generic_unique_content_and_select_everything() -> None:
    segments = [
        {"id": i, "start": float(i * 5), "end": float(i * 5 + 4),
         "text": f"unique substantive section number {i}"}
        for i in range(20)
    ]
    by_id = {s["id"]: s for s in segments}
    invalid = validate_llm_proposal(
        {"candidate_id": 3, "category": "explicit_target", "evidence": "unique substantive",
         "reason": "remove it"},
        by_id, set(by_id), set(), set(),
    )
    assert invalid is None
    broad = [
        {"sentence_ids": [s["id"]], "start": s["start"], "end": s["end"]}
        for s in segments
    ]
    assert not enforce_llm_budget(broad, segments, 100.0)


def test_preview_has_single_flight_seek_controller() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "const skipState" in html
    assert "setInterval(skipTick" not in html
    assert 'player.addEventListener("seeked", finishCutSeek)' in html
    assert "Preview could not decode the next kept frame" not in html
    assert "function armCutSeekRecovery()" in html
    assert "skipState.retries < 2" in html
    assert "skipState.wasPlaying = !player.paused; skipState.retries = 0;\n  player.currentTime = target;" in html


def test_safe_api_calls_retry_but_side_effecting_jobs_do_not() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'new Set(["/cuts","/gaps","/markers"])' in html
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


def test_real_audio_gaps_are_optional_and_old_slider_is_gone() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'id="gapToggle"' in html and 'aria-pressed="false"' in html
    assert 'id="gapWave"' in html and 'id="gapStart"' in html and 'id="gapEnd"' in html
    assert "S.gapSuggestions.size" in html and "Apply ${S.gapSuggestions.size} suggestions" in html
    assert 'id="gapSlider"' not in html


def test_exact_quality_export_never_silently_falls_back() -> None:
    source = Path(retake.__file__).read_text(encoding="utf-8")
    start = source.index("def export_job")
    end = source.index("def export_text", start)
    export_job_source = source[start:end]
    assert "VideoExportQuality.NEAR_LOSSLESS" in source
    assert "falling back to re-encode" not in export_job_source
    assert "_ffmpeg_reencode_cut(proj, keeps, out)" in export_job_source
    assert 'elif mode == "reencode"' in export_job_source


def test_latest_media_export_is_discovered_and_downloaded_safely() -> None:
    old_current = dict(retake.CURRENT)
    with temporary_project_root() as root:
        project_dir = retake.next_project_directory()
        source = project_dir / "media" / "original.mp4"
        source.write_bytes(b"source")
        exports = project_dir / "exports"
        older = exports / "original.retake_cut.mp4"
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
