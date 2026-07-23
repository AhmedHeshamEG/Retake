"""Unit tests for the cut-interval math in retake.py.

Run with:  python test_retake.py   (or pytest test_retake.py)
"""
from __future__ import annotations

import json
import os
import subprocess
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


def test_legacy_audio_gap_cuts_are_inert_without_rewriting_words() -> None:
    proj = {
        "tokens": [{"id": 0, "kind": "word", "text": "hello", "start": 1.0,
                    "end": 2.0, "seg": 0, "cut": True}],
        "audio_gaps": [{"id": "agap-1", "start": 1.8, "end": 3.0, "cut": True}],
    }
    assert approx(cut_intervals_from_tokens(proj), [(1.0, 2.0)])
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
            assert client.get("/project").json()["cut_intervals"] == [[.1, 2.5]]

            restored = client.post("/cuts", json={"cut_ids": [0, 3]}).json()
            assert restored["transcript_cut_intervals"] == [[.1, .3], [2.2, 2.5]]
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
    assert 'new Set(["/cuts","/markers","/voice-enhancement"])' in html
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


def test_gap_editor_is_removed_and_voice_enhancement_is_default_on() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    for removed in ('id="gapToggle"', 'id="gapWave"', 'id="gapStart"', 'id="gapEnd"', 'api("/gaps'):
        assert removed not in html
    assert 'id="enhanceEnabled" type="checkbox" checked' in html
    assert "Voice Enhancement" in html and "Reset to Great" in html
    for control in ("enhanceLoudness", "enhanceLeveling", "enhanceNoise", "enhanceDetail"):
        assert f'id="{control}" type="range"' in html
    source = Path(retake.__file__).read_text(encoding="utf-8")
    assert '@app.post("/gaps")' not in source and '@app.get("/waveform")' not in source


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
