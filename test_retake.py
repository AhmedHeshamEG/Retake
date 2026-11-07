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

import numpy as np
from fastapi.testclient import TestClient
import retake
import splice
import transcription

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
    trim_interior_silence,
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


def _word(idx: int, start: float, end: float, cut: bool = False, aligned: bool = True) -> dict:
    word = {"id": idx, "kind": "word", "text": f"w{idx}", "start": start,
            "end": end, "seg": 0, "cut": cut}
    if aligned:
        word["alignment_score"] = 0.9
    return word


def _synthetic_track(duration: float, loud: list[tuple[float, float]],
                     soft: list[tuple[float, float]] = (), quiet_dips: list[tuple[float, float]] = ()):
    """A recording with speech where ``loud`` says, a breath where ``soft`` says.

    Speech is a 180 Hz tone with a little noise, a breath is noise 22 dB under
    it, and the room is noise far below both. ``quiet_dips`` carves silence
    back out of speech, for the dip inside a word.
    """
    rate = splice.SAMPLE_RATE
    rng = np.random.default_rng(7)
    samples = rng.normal(0.0, 0.0004, int(duration * rate)).astype(np.float32)
    t = np.arange(len(samples)) / rate
    for start, end in loud:
        span = (t >= start) & (t < end)
        samples[span] += 0.3 * np.sin(2 * np.pi * 180 * t[span]) + rng.normal(0, 0.02, span.sum())
    for start, end in soft:
        span = (t >= start) & (t < end)
        samples[span] += rng.normal(0.0, 0.025, span.sum())
    for start, end in quiet_dips:
        span = (t >= start) & (t < end)
        samples[span] = rng.normal(0.0, 0.0004, span.sum())
    return splice.track_from_samples(samples)


def _plan(proj: dict, track) -> dict:
    proj.setdefault("alignment", {"status": "aligned", "version": retake.ALIGNMENT_VERSION,
                                  "coverage": 1.0})
    return retake.plan_joins(proj, track)


def test_join_lands_in_the_quiet_between_the_two_words() -> None:
    """The cut a person would make after zooming in.

    A deleted filler sits between two kept words, with a decoy dip inside it.
    Each join lands in the real quiet next to the kept word, keeps a short
    handle of it, and is never tempted by the decoy.
    """
    track = _synthetic_track(20.0, [(9.0, 9.6), (9.75, 9.95), (10.2, 10.8)],
                             quiet_dips=[(9.82, 9.86)])
    proj = {"duration_s": 20.0, "tokens": [
        _word(1, 9.0, 9.6), _word(2, 9.75, 9.95, cut=True), _word(3, 10.2, 10.8)]}
    plan = _plan(proj, track)
    (_, kept_end), (next_start, _) = plan["keeps"][0], plan["keeps"][1]
    assert 9.6 < kept_end < 9.75 and 9.95 < next_start < 10.2
    assert not 9.80 <= kept_end <= 9.88 and not 9.80 <= next_start <= 9.88
    assert not any(join["on_sound"] for join in plan["joins"])


def test_join_finds_the_quiet_when_whisper_timing_is_off() -> None:
    """A word alignment could not confirm is trusted as a wall, not as a time.

    Whisper says the kept word ends at 9.45 and the deleted one starts at 9.5;
    the audio says 9.6 and 9.75. The join still lands in the real pause, after
    all of the kept word and before any of the deleted one.
    """
    track = _synthetic_track(20.0, [(9.0, 9.6), (9.75, 10.0), (10.4, 11.0)])
    proj = {"duration_s": 20.0, "tokens": [
        _word(1, 9.0, 9.45, aligned=False), _word(2, 9.5, 10.0, cut=True, aligned=False),
        _word(3, 10.4, 11.0, aligned=False)]}
    plan = _plan(proj, track)
    kept_end = plan["keeps"][0][1]
    assert 9.6 < kept_end < 9.75
    assert not any(join["on_sound"] for join in plan["joins"])


def test_a_breath_is_never_split_by_a_join() -> None:
    """A breath is louder than the quiet, so a join is on one side of it or the other."""
    proj = {"duration_s": 20.0, "tokens": [
        _word(0, 5.0, 5.5), _word(1, 9.0, 9.6, cut=True), _word(2, 10.6, 11.2)]}
    track = _synthetic_track(20.0, [(5.0, 5.5), (9.0, 9.6), (10.6, 11.2)], soft=[(10.3, 10.55)])
    plan = _plan(proj, track)
    next_start = plan["keeps"][1][0]
    assert not 10.3 <= next_start <= 10.55
    assert 9.6 < next_start < 10.6    # none of the deleted word, all of the kept one
    assert not any(join["on_sound"] for join in plan["joins"])


def test_pause_cut_is_placed_in_the_quiet_and_never_splits_a_breath() -> None:
    """The bucket most bad joins came from.

    The pause detector's threshold called the breath silence, so a pause cut
    used to end in the middle of it. Now the breath goes with the pause, whole,
    and the kept side starts in the quiet just before the next word.
    """
    track = _synthetic_track(20.0, [(5.0, 6.0), (8.0, 9.0)], soft=[(7.55, 7.9)])
    proj = {
        "duration_s": 20.0,
        "tokens": [_word(1, 5.0, 6.0), _word(2, 8.0, 9.0)],
        "audio_gaps": [{"id": "agap-1", "detected_start": 6.05, "detected_end": 7.75,
                        "start": 6.11, "end": 7.69, "cut": True, "manual": False}],
        "audio_gap_settings": {"keep_pause_s": 0.12},
    }
    plan = _plan(proj, track)
    (_, kept_end), (next_start, _) = plan["keeps"][0], plan["keeps"][1]
    assert 6.0 < kept_end < 6.2
    assert not 7.55 <= next_start <= 7.9
    assert next_start < 8.0
    assert not any(join["on_sound"] for join in plan["joins"])


def test_pause_cut_drawn_by_hand_is_used_exactly() -> None:
    track = _synthetic_track(20.0, [(5.0, 6.0), (8.0, 9.0)])
    proj = {
        "duration_s": 20.0,
        "tokens": [_word(1, 5.0, 6.0), _word(2, 8.0, 9.0)],
        "audio_gaps": [{"id": "agap-1", "detected_start": 6.05, "detected_end": 7.95,
                        "start": 6.5, "end": 7.0, "cut": True, "manual": True}],
    }
    assert _plan(proj, track)["keeps"] == [(0.0, 6.5), (7.0, 20.0)]


def test_words_said_in_one_breath_join_at_the_quietest_instant_and_are_reported() -> None:
    """No pause at all: the join goes to the dip between the words, and the
    self-check says so, because this one is worth a listen."""
    track = _synthetic_track(20.0, [(9.0, 10.8)])
    levels = track.db.copy()
    dip = track.index(9.62)
    levels[dip - 1:dip + 2] -= 12.0       # a consonant closure, not silence
    track.db = levels
    proj = {"duration_s": 20.0, "tokens": [
        _word(1, 9.0, 9.6), _word(2, 9.6, 9.95, cut=True), _word(3, 9.95, 10.8)]}
    plan = _plan(proj, track)
    kept_end = plan["keeps"][0][1]
    assert abs(kept_end - 9.62) <= 0.02
    glued = [join for join in plan["joins"] if join["how"] == "glued"]
    assert glued and all(join["on_sound"] for join in glued)


def test_short_quiet_puts_the_join_in_its_middle() -> None:
    """In a 40 ms dip the far edge is already the next word."""
    track = _synthetic_track(20.0, [(9.0, 9.6), (9.64, 10.0), (10.5, 11.0)])
    proj = {"duration_s": 20.0, "tokens": [
        _word(1, 9.0, 9.6), _word(2, 9.64, 10.0, cut=True), _word(3, 10.5, 11.0)]}
    kept_end = _plan(proj, track)["keeps"][0][1]
    assert 9.605 <= kept_end <= 9.635


def test_every_join_side_fades_and_no_duration_changes() -> None:
    chain = retake._keep_audio_chain(1.0, 2.0, True, True)
    assert "afade=t=in:st=0:d=0.0060" in chain
    assert "afade=t=out:st=0.994000:d=0.0060" in chain
    assert "afade" not in retake._keep_audio_chain(1.0, 2.0, False, False)
    fades = retake._keep_fades([(0.0, 5.0), (6.0, 10.0)], 10.0)
    assert fades == [(False, True), (True, False)]
    graph = retake._audio_concat_graph([(0.0, 5.0), (6.0, 10.0)], duration=10.0)
    assert graph.count("afade") == 2 and "concat=n=2" in graph


def test_joins_report_their_place_on_the_edited_timeline() -> None:
    track = _synthetic_track(20.0, [(2.0, 3.0), (5.0, 6.0), (8.0, 9.0)])
    proj = {"duration_s": 20.0, "tokens": [
        _word(1, 2.0, 3.0), _word(2, 5.0, 6.0, cut=True), _word(3, 8.0, 9.0)]}
    plan = _plan(proj, track)
    first_keep = plan["keeps"][0]
    end_join, start_join = plan["joins"][0], plan["joins"][1]
    assert end_join["side"] == "end" and start_join["side"] == "start"
    assert end_join["edited_time"] == round(first_keep[1] - first_keep[0], 3)
    assert start_join["edited_time"] == end_join["edited_time"]
    assert plan["edited_duration"] == round(sum(b - a for a, b in plan["keeps"]), 3)


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


def test_interior_silence_is_shortened_to_a_natural_pause() -> None:
    """Dead air between two kept sentences belongs to no cut, so only this
    reaches it."""
    keeps = [(0.0, 30.0)]
    trimmed = trim_interior_silence(keeps, [(10.0, 18.0)], 60.0)
    assert trimmed == [(0.0, 10.175), (17.825, 30.0)]
    # What is left is a real pause, balanced across the join: half the retained
    # silence stays on each side, so the speaker still breathes.
    kept_pause = (10.175 - 10.0) + (18.0 - 17.825)
    assert round(kept_pause, 3) == retake.EXPORT_INTERIOR_PAUSE_KEEP_S
    assert round(8.0 - kept_pause, 3) == round(17.825 - 10.175, 3)


def test_interior_silence_leaves_speech_rhythm_alone() -> None:
    # Below the floor: this is how the person talks, not dead air.
    assert trim_interior_silence([(0.0, 30.0)], [(10.0, 10.8)], 60.0) == [(0.0, 30.0)]
    assert trim_interior_silence([(0.0, 30.0)], [], 60.0) == [(0.0, 30.0)]
    # A silence touching an edge is edge snapping's business, not this one.
    assert trim_interior_silence([(10.0, 30.0)], [(8.0, 14.0)], 60.0) == [(10.0, 30.0)]
    assert trim_interior_silence([(10.0, 30.0)], [(26.0, 33.0)], 60.0) == [(10.0, 30.0)]


def test_interior_silence_keeps_intervals_ordered_and_non_overlapping() -> None:
    trimmed = trim_interior_silence(
        [(0.0, 40.0), (50.0, 80.0)], [(5.0, 9.0), (20.0, 24.0), (60.0, 70.0)], 100.0
    )
    for index, (start, end) in enumerate(trimmed):
        assert end > start
        if index:
            assert start >= trimmed[index - 1][1]
    assert sum(end - start for start, end in trimmed) < 70.0


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


def test_project_mcp_config_points_at_the_servers_own_interpreter() -> None:
    """A globally-installed `mcp` is exactly what this config exists to avoid."""
    config = json.loads(
        (Path(retake.__file__).with_name(".mcp.json")).read_text(encoding="utf-8")
    )
    server = config["mcpServers"]["retake"]
    assert server["type"] == "stdio"
    assert ".venv" in server["command"]
    script = Path(retake.__file__).with_name(server["args"][0])
    assert script.name == "retake_mcp.py" and script.exists()
    # It must talk to the editor's own address, not guess one.
    assert server["env"]["RETAKE_URL"].endswith(str(retake.PORT))


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


def test_skill_is_written_entirely_in_english() -> None:
    """The skill edits Arabic recordings; it does not write Arabic itself."""
    skill = SKILL_PATH.read_text(encoding="utf-8")
    arabic = [ch for ch in skill if "؀" <= ch <= "ۿ"]
    assert not arabic, f"Arabic characters left in SKILL.md: {''.join(arabic)[:80]}"
    # The English command grammar it now documents is the one the parser reads.
    for form in ('cut: "', 'keep: "', "cut gap:", "Needs a decision:",
                 "(first)", "(last)", "(every time)"):
        assert form in skill, form
    # Quoted transcript text still stays in its original language.
    assert "in the transcript's own" in skill and "language" in skill


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
    assert "function whenRunnable()" in html
    # ...and only after a chunk has already failed. A working link is never
    # made to wait for the tab to become visible.
    assert html.count("await whenRunnable();") == 1
    assert "· retrying…" in html
    # The old dead ends are gone.
    assert "Transfer paused safely on the laptop" not in html
    assert "connection kept dropping after" not in html
    assert "Transfer interrupted" not in html


def test_transfer_keeps_the_phone_awake_without_a_certificate() -> None:
    """wakeLock needs HTTPS; an audio loop keeps a locked phone uploading."""
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "function startKeepAlive()" in html
    assert "function stopKeepAlive()" in html
    assert "keepAliveEl.loop = true" in html
    # Playback has to start inside the tap that opens the picker.
    assert 'startKeepAlive(); $("mediaFile").click();' in html
    # Nothing may push the creator towards trusting a self-signed certificate.
    assert "Open Retake's https:// address instead" not in html


def test_transfer_offers_resume_rather_than_a_restart() -> None:
    """A phone that dropped the file keeps every byte already on the laptop."""
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "needsReselect" in html
    assert "nothing was lost" in html
    assert 'S.uploadFile ? "Retry" : "Resume"' in html


def test_a_phone_uploads_its_file_in_one_request() -> None:
    """Reading a picked file in slices is the one shape iOS cannot do.

    WebKit invalidates the handle behind a file picked out of the photo library
    about a minute after it is opened (bugs.webkit.org 228683), so a slice-per-
    request transfer of a large video gets a little way and then stops dead
    with nothing to report -- which is exactly what happened: four lanes, one
    chunk each, then silence. Handing the whole File to one request puts the
    reading inside the browser, which streams it off the phone over the single
    handle it opened itself. It is also what tus does by default.
    """
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "file.slice(from)" in html
    # Nothing may read the file itself, and nothing may run several at once.
    assert "FileReader" not in html
    assert "UPLOAD_LANES" not in html
    assert "nextChunkSize" not in html
    # The server must take that body as it arrives rather than collect it,
    # because there is no size at which holding a phone video in memory is
    # reasonable.
    source = Path(retake.__file__).read_text(encoding="utf-8")
    assert "async for piece in request.stream():" in source
    assert "_write_chunk_at, session_id, partial, offset + written, bytes(piece)" in source
    assert "upload chunk is too large" not in source


def test_a_transfer_cannot_wait_forever_and_shows_what_has_left_the_phone() -> None:
    """A stalled transfer with no error and a bar on zero was the whole failure.

    fetch resolves when the reply arrives and reports nothing before then, so a
    request whose bytes stopped moving neither settles nor fails, and every
    recovery path lives in a `catch` that could not be reached. XMLHttpRequest
    reports what has actually gone, which both feeds the figure on screen and
    is what the watchdog measures.
    """
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert "xhr.upload.onprogress" in html
    assert "const UPLOAD_SILENCE_LIMIT_MS = 60 * 1000;" in html
    # The deadline must abandon the request, not merely complain about it.
    assert "xhr.abort();" in html
    # And the stretch where nothing can move -- the phone exporting the video
    # before a byte of it exists -- says so rather than reading as zero.
    assert "Preparing the video on your phone" in html


def test_a_dropped_transfer_asks_the_laptop_what_it_actually_has() -> None:
    """A request can die after its bytes landed, so the client must not guess.

    And when two goes in a row move nothing, the file behind the picker has
    gone stale -- the one failure only the person can fix, by picking it again.
    """
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'start = await api("/upload/start", descriptor);' in html
    assert "offset = Math.max(offset, landed);" in html
    assert "stale.needsReselect = true;" in html
    # A phone that walks away mid-request is a pause, not a server error.
    source = Path(retake.__file__).read_text(encoding="utf-8")
    assert "except ClientDisconnect:" in source


def test_a_locked_screen_does_not_pause_between_chunks() -> None:
    """The audio loop keeps the tab running, so hidden must not mean waiting."""
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'document.visibilityState === "visible" || keepAliveRunning()' in html
    assert "keepAliveRunning() ? 1500" not in html


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
        # Re-sending a range that already arrived is harmless: lanes retry, and
        # a resumed transfer re-sends whatever the laptop had not written down.
        again = client.post(
            f"/upload/chunk/{session}", content=b"abc", headers={"X-Upload-Offset": "0"}
        )
        assert again.status_code == 200 and again.json()["offset"] == 3
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


def test_chunks_may_arrive_out_of_order() -> None:
    """Parallel lanes finish in whatever order the link allows."""
    with temporary_project_root():
        client = TestClient(app)
        name = f"retake-upload-test-{uuid.uuid4().hex}.bin"
        fingerprint = uuid.uuid4().hex
        start = client.post(
            "/upload/start", json={"name": name, "size": 9, "fingerprint": fingerprint}
        )
        session = start.json()["session"]
        tail = client.post(
            f"/upload/chunk/{session}", content=b"ghi", headers={"X-Upload-Offset": "6"}
        )
        # The tail is stored, but a resume can only start where the run from
        # zero ends, so nothing is confirmed yet.
        assert tail.status_code == 200 and tail.json()["offset"] == 0
        assert client.post("/upload/finish", json={"session": session}).status_code == 409
        client.post(f"/upload/chunk/{session}", content=b"def", headers={"X-Upload-Offset": "3"})
        head = client.post(
            f"/upload/chunk/{session}", content=b"abc", headers={"X-Upload-Offset": "0"}
        )
        assert head.json()["offset"] == 9
        done = client.post("/upload/finish", json={"session": session})
        assert done.status_code == 200
        assert Path(done.json()["path"]).read_bytes() == b"abcdefghi"


def test_upload_rejects_offsets_past_the_declared_size() -> None:
    with temporary_project_root():
        client = TestClient(app)
        name = f"retake-upload-test-{uuid.uuid4().hex}.bin"
        start = client.post(
            "/upload/start",
            json={"name": name, "size": 4, "fingerprint": uuid.uuid4().hex},
        )
        session = start.json()["session"]
        assert client.post(
            f"/upload/chunk/{session}", content=b"x", headers={"X-Upload-Offset": "9"}
        ).status_code == 409
        assert client.post(
            f"/upload/chunk/{session}", content=b"toolong", headers={"X-Upload-Offset": "0"}
        ).status_code == 400


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
    # Each section on its own audio, padded only into its half of a pause, so
    # one take can never be aligned onto its neighbour.
    assert windows == [
        {"start": 2.5, "end": 14.1, "text": "first section"},
        {"start": 14.1, "end": 27.5, "text": "second section"},
        {"start": 37.5, "end": 50.5, "text": "third section"},
    ]
    # Sections that overlap are aligned together.
    project["segments"][1]["start"] = 13.0
    windows = retake.alignment_segments_for_project(project, max_span=30.0, padding=0.5)
    assert windows[0] == {
        "start": 4.5, "end": 25.5, "text": "first section second section",
    }


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


def test_transcription_is_configured_for_completeness_not_speed() -> None:
    """Nothing may be dropped before the transcript exists.

    The VAD used to run ahead of Whisper and delete what it judged non-speech,
    so a softly begun sentence never reached the model; the turbo model and
    int8 quantization then cost accuracy on what did. Each of these is what
    keeps whole sentences from going missing, so each is pinned here.
    """
    source = Path(retake.__file__).read_text(encoding="utf-8")
    decoding = Path(transcription.__file__).read_text(encoding="utf-8")
    assert "vad_filter=False" in decoding, "pre-Whisper VAD drops whole sentences"
    assert "vad_filter=True" not in source + decoding
    assert '("large-v3", "cuda", "float16")' in source
    # The comment above the list still names turbo, to say why it is gone.
    assert '("large-v3-turbo"' not in source
    # Whisper discards nothing itself; its no-speech probability is read here.
    assert transcription.PHRASE_OPTIONS == {
        "no_speech_threshold": None, "log_prob_threshold": None,
    }
    assert transcription.DECODE_OPTIONS["beam_size"] == 10
    # Phrase by phrase, so a take cannot inherit a neighbour's wording.
    assert transcription.DECODE_OPTIONS["condition_on_previous_text"] is False
    assert transcription.DECODE_OPTIONS["vad_filter"] is False
    # Words are never offered to Whisper ahead of the audio: name hints made
    # it type those names into silent clips.
    assert "hotwords" not in decoding and "initial_prompt" not in decoding
    assert "transcription.transcribe_recording(" in source
    assert "run_forced_alignment(" in source


# ---------------------------------------------------------------------------
# Transcription: one phrase at a time, and splitting a phrase that hides a take
# ---------------------------------------------------------------------------

def recording(*bursts: tuple[float, float], seconds: float) -> np.ndarray:
    """Quiet room tone with loud noise wherever someone is 'speaking'."""
    rng = np.random.default_rng(7)
    rate = transcription.SAMPLE_RATE
    audio = rng.normal(0.0, 12.0, int(seconds * rate))
    for start, end in bursts:
        audio[int(start * rate): int(end * rate)] = rng.normal(
            0.0, 4000.0, int(end * rate) - int(start * rate)
        )
    return audio.astype(np.int16)


class HeardWord:
    def __init__(self, word: str, start: float, end: float) -> None:
        self.word, self.start, self.end, self.probability = word, start, end, 0.9


class HeardSegment:
    def __init__(self, words: list[tuple[str, float, float]], no_speech: float) -> None:
        self.words = [HeardWord(*word) for word in words]
        self.no_speech_prob = no_speech


class ScriptedWhisper:
    """Answers each decode from a script, and remembers what it was handed."""

    def __init__(self, *answers: list[HeardSegment]) -> None:
        self.answers = list(answers)
        self.clips: list[float] = []
        self.options: list[dict] = []

    def transcribe(self, samples, **options):
        self.clips.append(len(samples) / transcription.SAMPLE_RATE)
        self.options.append(options)
        return (self.answers.pop(0) if self.answers else []), None


def heard_words(heard: dict) -> list[dict]:
    return [word for segment in heard["segments"] for word in segment["words"]]


def test_speech_map_finds_the_islands_and_ends_phrases_on_pauses() -> None:
    audio = recording((1.0, 3.0), (3.2, 5.0), (6.0, 8.0), seconds=10.0)
    levels = transcription.frame_levels(audio)
    islands = transcription.audible_islands(
        levels, transcription.audible_threshold(levels)
    )
    assert [(round(a, 1), round(b, 1)) for a, b in islands] == [
        (1.0, 3.0), (3.2, 5.0), (6.0, 8.0),
    ]
    # A breath-length dip stays inside a phrase; a one-second pause ends it.
    # Padding is real silence, never more than half the pause.
    phrases = transcription.plan_phrases(islands, levels, 10.0)
    assert [(round(a, 2), round(b, 2)) for a, b in phrases] == [
        (0.8, 5.2), (5.8, 8.2),
    ]


def test_speech_with_no_pause_is_split_at_a_trough_never_dropped() -> None:
    audio = recording((0.0, 70.0), seconds=70.0)
    levels = transcription.frame_levels(audio)
    islands = transcription.audible_islands(
        levels, transcription.audible_threshold(levels)
    )
    phrases = transcription.plan_phrases(islands, levels, 70.0)
    assert len(phrases) >= 3
    assert all(end - start <= transcription.PHRASE_MAX_S + 1e-6 for start, end in phrases)
    assert phrases[0][0] == 0.0 and abs(phrases[-1][1] - 70.0) < 0.02


def test_two_takes_with_a_pause_between_them_are_each_heard_alone() -> None:
    """Two takes in one decode come back as one; here they never share one."""
    audio = recording((1.0, 3.0), (4.0, 6.0), seconds=8.0)
    model = ScriptedWhisper(
        [HeardSegment([("the", 0.2, 0.7), ("junk", 0.7, 1.4), ("arrives.", 1.4, 2.1)], 0.02)],
        [HeardSegment([("The", 0.2, 0.7), ("junk", 0.7, 1.4), ("arrives", 1.4, 2.1)], 0.03)],
    )
    heard = transcription.transcribe_recording(model, audio, "en")
    texts = [[word["text"] for word in segment["words"]] for segment in heard["segments"]]
    assert texts == [["the", "junk", "arrives."], ["The", "junk", "arrives"]]
    second = heard["segments"][1]["words"]
    assert 3.7 <= second[0]["start"] and second[-1]["end"] <= 6.3
    assert len(model.clips) == 2 and all(clip < 3.0 for clip in model.clips)
    assert all(options["no_speech_threshold"] is None for options in model.options)
    assert all(options["vad_filter"] is False for options in model.options)


def test_noise_is_not_given_words_and_is_reported_as_a_sound() -> None:
    """A cough decoded on its own comes back as "Thank you." -- it is no such thing."""
    audio = recording((1.0, 3.0), (4.5, 5.2), (6.5, 7.0), seconds=8.0)
    model = ScriptedWhisper(
        [HeardSegment([("hello", 0.2, 0.8), ("there", 0.8, 1.5)], 0.02)],
        [HeardSegment([("Thank", 0.1, 0.3), ("you.", 0.3, 0.5)], 0.93)],
        # Confident by log-probability, a little unsure it is speech at all.
        [HeardSegment([("Thanks", 0.1, 0.4), ("for", 0.4, 0.5), ("watching!", 0.5, 0.6)], 0.4)],
    )
    heard = transcription.transcribe_recording(model, audio, "en")
    words = heard_words(heard)
    assert [word["text"] for word in words] == ["hello", "there"]
    assert [noise["text"] for noise in heard["dropped"]] == [
        "Thank you.", "Thanks for watching!",
    ]
    sounds = transcription.non_speech_sounds(heard["islands"], words)
    assert [(round(a, 1), round(b, 1)) for a, b in sounds] == [(4.5, 5.2), (6.5, 7.0)]


def test_a_clearly_spoken_thank_you_is_kept() -> None:
    audio = recording((1.0, 2.0), seconds=3.0)
    model = ScriptedWhisper(
        [HeardSegment([("Thank", 0.2, 0.6), ("you.", 0.6, 1.1)], 0.02)],
    )
    heard = transcription.transcribe_recording(model, audio, "en")
    assert [word["text"] for word in heard_words(heard)] == ["Thank", "you."]


def test_a_word_stretched_over_a_hidden_take_splits_the_phrase() -> None:
    """Two takes a breath apart share a phrase; Whisper hides one under a long word."""
    # A 0.25 s dip: shorter than a phrase break, but a real drop in level.
    audio = recording((1.0, 3.0), (3.25, 5.5), seconds=7.0)
    model = ScriptedWhisper(
        # The whole phrase: "random" drawn from the first take over the second.
        [HeardSegment([
            ("we", 0.2, 0.5), ("pick", 0.5, 0.9), ("random", 0.9, 4.9), ("us.", 4.9, 5.3),
        ], 0.01)],
        # Left half, alone.
        [HeardSegment([("we", 0.2, 0.5), ("pick", 0.5, 0.9), ("the", 0.9, 1.1), ("number", 1.1, 1.8)], 0.01)],
        # Right half, alone.
        [HeardSegment([
            ("we", 0.1, 0.4), ("pick", 0.4, 0.8), ("random", 0.8, 1.4),
            ("to", 1.4, 1.6), ("us.", 1.6, 2.0),
        ], 0.01)],
    )
    heard = transcription.transcribe_recording(model, audio, "en")
    words = heard_words(heard)
    assert [word["text"] for word in words] == [
        "we", "pick", "the", "number", "we", "pick", "random", "to", "us.",
    ]
    assert all(b["start"] >= a["end"] - 1e-6 for a, b in zip(words, words[1:]))
    assert max(word["end"] - word["start"] for word in words) < 1.0
    split = heard["splits"][0]
    assert split["kept"] is True and 3.0 <= split["at"] <= 3.25
    # Both halves were heard whole: neither is a scrap.
    assert len(model.clips) == 3 and min(model.clips[1:]) > 2.0


def test_a_split_that_finds_nothing_more_is_not_kept() -> None:
    audio = recording((1.0, 3.0), (3.25, 5.5), seconds=7.0)
    whole = [HeardSegment([("well", 0.2, 0.6), ("sooo", 0.6, 4.9)], 0.02)]
    model = ScriptedWhisper(
        whole,
        [HeardSegment([("well", 0.2, 0.6)], 0.02)],
        [HeardSegment([("So...", 0.1, 2.0)], 0.1)],
    )
    heard = transcription.transcribe_recording(model, audio, "en")
    assert [word["text"] for word in heard_words(heard)] == ["well", "sooo"]
    assert heard["splits"][0]["kept"] is False


def test_continuous_speech_is_never_split_inside_a_word() -> None:
    """A drawn-out word with no dip in level under it is left whole."""
    audio = recording((1.0, 5.0), seconds=6.0)
    model = ScriptedWhisper(
        [HeardSegment([("well", 0.2, 0.6), ("sooo", 0.6, 4.1)], 0.02)],
    )
    heard = transcription.transcribe_recording(model, audio, "en")
    assert [word["text"] for word in heard_words(heard)] == ["well", "sooo"]
    assert heard["splits"] == [] and len(model.clips) == 1


def test_language_is_judged_across_the_recording_not_from_its_first_window() -> None:
    """An accented opening must not have the whole recording written in Arabic."""

    class Listener:
        def __init__(self) -> None:
            self.answers = [
                [("ar", 0.71), ("en", 0.26)],
                [("en", 0.74), ("ar", 0.23)],
                [("en", 0.96), ("ar", 0.04)],
            ]

        def detect_language(self, audio):
            ranked = self.answers.pop(0)
            return ranked[0][0], ranked[0][1], ranked

    audio = recording((0.0, 3.0), seconds=3.0)
    windows = [[(0.0, 1.0)], [(1.0, 2.0)], [(2.0, 3.0)]]
    assert transcription.detect_language(Listener(), audio, windows) == "en"
    assert transcription.detect_language(Listener(), audio, []) == "en"


def test_the_tail_of_a_word_is_not_taken_for_a_missing_word() -> None:
    """A spoken number is timed short; its last syllable is not a retake."""
    islands = [(1.0, 3.0), (6.0, 7.0)]
    words = [
        {"text": "since", "start": 1.0, "end": 1.5},
        {"text": "1992", "start": 1.5, "end": 2.2},
    ]
    stretches = transcription.wordless_stretches(islands, words)
    # 0.68 s runs on from "1992" and is left alone; the island that stands by
    # itself is long enough to matter.
    assert [(s["start"], s["end"], s["attached"]) for s in stretches] == [(6.0, 7.0, False)]
    assert (
        transcription.plausible_word_seconds("1992")
        > transcription.plausible_word_seconds("since")
    )
    # "%" is the word "percent", not a mark with no length to speak of.
    assert transcription.plausible_word_seconds("%") >= 0.9


def test_sound_tokens_are_woven_in_and_every_token_is_renumbered() -> None:
    tokens = [
        {"id": 0, "kind": "word", "text": "one", "start": 1.0, "end": 1.4, "seg": 0, "cut": False},
        {"id": 1, "kind": "gap", "start": 1.4, "end": 3.0, "cut": False},
        {"id": 2, "kind": "word", "text": "two", "start": 3.0, "end": 3.4, "seg": 1, "cut": False},
    ]
    woven = retake.weave_sound_tokens(
        tokens, [(1.9, 2.1), (3.1, 3.2), (5.0, 5.3)], duration=6.0, gap_threshold=0.35,
    )
    assert [(t["id"], t["kind"], t["start"], t["end"]) for t in woven] == [
        (0, "gap", 0.0, 1.0),
        (1, "word", 1.0, 1.4),
        (2, "gap", 1.4, 1.9),
        (3, "sound", 1.9, 2.1),
        (4, "gap", 2.1, 3.0),
        (5, "word", 3.0, 3.4),
        (6, "gap", 3.4, 5.0),
        (7, "sound", 5.0, 5.3),
        (8, "gap", 5.3, 6.0),
    ]
    # The sound that final word timing reaches into is dropped, not kept on
    # top of the word; and the caller's tokens are left as they were.
    assert woven[5]["text"] == "two" and woven[5]["seg"] == 1
    assert [t["id"] for t in tokens] == [0, 1, 2]


def test_a_cut_sound_removes_itself_and_alignment_leaves_it_where_it_was_measured() -> None:
    tokens = [
        {"id": 0, "kind": "word", "text": "one", "start": 1.0, "end": 1.4, "seg": 0, "cut": False},
        {"id": 1, "kind": "sound", "start": 1.9, "end": 2.1, "cut": True},
        {"id": 2, "kind": "word", "text": "two", "start": 3.0, "end": 3.4, "seg": 0, "cut": False},
    ]
    # Only the sound goes: the pause around it is not a cut word's pause.
    assert retake.transcript_cut_intervals(
        tokens, retake.WORD_CUT_SAFETY_S, absorb_pauses=True, duration=5.0
    ) == [(1.9, 2.1)]
    retake.recalibrate_legacy_gap_timings(tokens, 5.0)
    assert (tokens[1]["start"], tokens[1]["end"]) == (1.9, 2.1)


def test_editor_shows_sounds_and_cuts_them_like_words() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'if (t.kind === "sound")' in html
    assert 'class="snd' in html and "#transcript .snd.cut" in html
    # A sound Whisper heard words in, that the wave did not confirm, is marked.
    assert "#transcript .snd.maybe" in html and "t.maybe_speech" in html
    # Same token path as a word: an ID, so click, undo, save and export follow.
    assert 'data-id="${t.id}" title="Sound with no words' in html
    assert 'id="soundCut"' in html
    assert "toggleTokens(S.soundIds, S.soundIds.some((id) => !S.cut.has(id)))" in html


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


def test_every_output_plays_the_planned_joins() -> None:
    """Export, text exports and preview all use plan_joins, so they agree."""
    source = Path(retake.__file__).read_text(encoding="utf-8")
    start = source.index("def export_job")
    end = source.index("def export_text", start)
    media_export = source[start:end]
    assert "planned_joins(proj)" in media_export
    text_export = source[end:source.index("def fully_cut_segments", end)]
    assert "final_keeps(proj)" in text_export
    plan = source[source.index("def plan_joins"):source.index("def planned_joins")]
    # Interior dead air is shortened only after the edges are settled, so the
    # two never argue about the same silence.
    assert plan.index("keep_list(cuts") < plan.index("trim_interior_silence")


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


# ---------------------------------------------------------------------------
# Display orientation
# ---------------------------------------------------------------------------

def test_every_display_matrix_orientation_maps_to_one_filter_chain() -> None:
    """All eight orientations, including the four that carry a reflection.

    The filter chains below were each verified against FFmpeg's own autorotate
    output on a generated file carrying that exact matrix.
    """
    assert len(retake._DISPLAY_MATRIX_ORIENTATION) == 8
    assert set(retake._DISPLAY_MATRIX_ORIENTATION.values()) == {
        (rotation, mirrored)
        for rotation in (0, 90, 180, 270)
        for mirrored in (False, True)
    }
    assert set(retake._ORIENTATION_FILTERS) == set(
        retake._DISPLAY_MATRIX_ORIENTATION.values()
    )
    assert retake._ORIENTATION_FILTERS[(0, False)] == []
    assert retake._ORIENTATION_FILTERS[(0, True)] == ["hflip"]
    assert retake._ORIENTATION_FILTERS[(180, False)] == ["hflip", "vflip"]
    assert retake._ORIENTATION_FILTERS[(180, True)] == ["vflip"]
    assert retake._ORIENTATION_FILTERS[(90, True)] == ["transpose=clock_flip"]
    assert retake._ORIENTATION_FILTERS[(270, True)] == ["transpose=cclock_flip"]


def _matrix(a: int, b: int, c: int, d: int) -> dict[str, str]:
    unit = 65536
    return {
        "displaymatrix":
            f"\n00000000: {a * unit} {b * unit} 0"
            f"\n00000001: {c * unit} {d * unit} 0"
            f"\n00000002: 0 0 1073741824\n"
    }


def test_a_mirror_is_never_mistaken_for_a_half_turn() -> None:
    """The bug that left phone footage mirrored, or upside down.

    ffprobe derives its scalar `rotation` with atan2, which is undefined under
    reflection: a pure horizontal mirror and a true 180 rotation both report
    -180, and a vertical flip reports 0. Reading that number alone applied a
    half turn to a mirrored clip, and nothing at all to a flipped one.
    """
    half_turn = _matrix(-1, 0, 0, -1)
    mirror = _matrix(-1, 0, 0, 1)
    vertical = _matrix(1, 0, 0, -1)
    # The two the scalar cannot tell apart.
    half_turn["rotation"] = mirror["rotation"] = -180
    vertical["rotation"] = 0

    assert retake.parse_display_matrix(half_turn) == (180, False)
    assert retake.parse_display_matrix(mirror) == (0, True)
    assert retake.parse_display_matrix(vertical) == (180, True)

    def filters(side_data: dict[str, str]) -> list[str]:
        rotation, mirrored = retake.stream_orientation({"side_data_list": [side_data]})
        return retake._orientation_normalization_filters(
            {"rotation": rotation, "mirrored": mirrored}
        )

    assert filters(half_turn) == ["hflip", "vflip"]
    assert filters(mirror) == ["hflip"]
    # Reported as rotation 0, so the old code emitted no filter and the export
    # came out upside down.
    assert filters(vertical) == ["vflip"]


def test_display_matrix_beats_the_legacy_rotate_tag() -> None:
    """Only the matrix can describe a mirror, so it wins where both exist."""
    stream = {"tags": {"rotate": "90"}, "side_data_list": [_matrix(-1, 0, 0, 1)]}
    assert retake.stream_orientation(stream) == (0, True)
    # With no matrix the tag is all there is, and it can never mean a mirror.
    assert retake.stream_orientation({"tags": {"rotate": "270"}}) == (270, False)
    assert retake.stream_orientation({}) == (0, False)
    # A malformed matrix must not silently become "upright".
    assert retake.parse_display_matrix({"displaymatrix": "nonsense"}) is None


def test_a_mirror_does_not_change_display_dimensions() -> None:
    portrait = {"width": 1920, "height": 1080, "rotation": 270, "mirrored": True}
    assert retake.display_dimensions(portrait) == (1080, 1920)
    upright = {"width": 1920, "height": 1080, "rotation": 180, "mirrored": True}
    assert retake.display_dimensions(upright) == (1920, 1080)


def test_proxy_integrity_rejects_a_surviving_mirror() -> None:
    source = Path(retake.__file__).read_text(encoding="utf-8")
    assert "proxy display matrix still carries a mirror" in source


# ---------------------------------------------------------------------------
# Gap scopes and the length filter
# ---------------------------------------------------------------------------

def _gap_project() -> dict[str, Any]:
    """Two sentences: the first kept, the second cut away entirely."""
    return {
        "duration_s": 30.0,
        "tokens": [
            {"id": 0, "kind": "word", "text": "one", "start": 1.0, "end": 2.0,
             "seg": 0, "cut": False},
            {"id": 1, "kind": "word", "text": "two", "start": 6.0, "end": 7.0,
             "seg": 0, "cut": False},
            {"id": 2, "kind": "word", "text": "three", "start": 15.0, "end": 16.0,
             "seg": 1, "cut": True},
            {"id": 3, "kind": "word", "text": "four", "start": 20.0, "end": 21.0,
             "seg": 1, "cut": True},
        ],
        "segments": [
            {"id": 0, "start": 1.0, "end": 7.0, "text": "one two"},
            {"id": 1, "start": 15.0, "end": 21.0, "text": "three four"},
        ],
        "audio_gaps": [
            # Inside the kept sentence: removing it really does shorten the cut.
            {"id": "agap-1", "detected_start": 2.0, "detected_end": 6.0,
             "start": 2.0, "end": 6.0, "cut": False, "manual": False},
            # Inside the sentence that is cut away: already inside the cut.
            {"id": "agap-2", "detected_start": 16.0, "detected_end": 20.0,
             "start": 16.0, "end": 20.0, "cut": False, "manual": False},
        ],
    }


def test_gap_scopes_separate_kept_speech_from_removed_speech() -> None:
    partition = retake.partition_gap_candidates(_gap_project())
    assert partition["kept"] == ["agap-1"]
    assert partition["removed"] == ["agap-2"]
    # Every detected gap lands in exactly one scope.
    assert sorted(partition["kept"] + partition["removed"]) == ["agap-1", "agap-2"]
    # The old single-scope helper still answers for the existing UI contract.
    assert retake.scoped_gap_candidate_ids(_gap_project()) == ["agap-1"]


def test_both_gap_scopes_reach_the_browser() -> None:
    payload = retake.project_response_payload(_gap_project())
    assert payload["scoped_gap_candidate_ids"] == ["agap-1"]
    assert payload["removed_gap_candidate_ids"] == ["agap-2"]
    # The payload must not mutate the project it describes.
    assert "removed_gap_candidate_ids" not in _gap_project()


def test_gap_length_filter_is_separate_from_detection_settings() -> None:
    """Narrowing what is offered must not re-run detection.

    The gear panel decides what counts as a silence at all and re-analyses the
    audio; the length filter only narrows what has already been found, so it
    cannot discard detected gaps or the manual boundary edits made to them.
    """
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'id="gapLonger"' in html
    assert "function gapLongerThan()" in html
    assert "function gapsInScope(scope)" in html
    # It filters on the detector's own evidence, never on the edited bounds.
    assert "gap.detected_end - gap.detected_start < minimum" in html
    # All three scopes are offered, and each reports its own count.
    for control in ("gapScopeAll", "gapScoped", "gapScopeRemoved"):
        assert f'id="{control}"' in html
    assert "removed_gap_candidate_ids" in html
    # Changing the filter must not call the detection endpoint.
    assert '$("gapLonger").addEventListener("input", updateGapControls)' in html


def test_gap_length_filter_accepts_a_range_and_shows_the_lengths_it_has() -> None:
    """A ceiling is optional, and the lengths present are visible while choosing.

    "Longer than X" cannot express "the breaths but not the long thinking
    pauses". A second, empty-by-default box turns the same control into an
    interval without changing what it does when left alone, and the histogram
    beside it is the detected silences themselves, so the numbers to type come
    from the recording rather than from guessing.
    """
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'id="gapShorter"' in html
    # Empty means no ceiling, so the default behaviour is exactly as before.
    assert "function gapShorterThan()" in html
    assert 'if (!raw) return Infinity;' in html
    assert "gap.detected_end - gap.detected_start > maximum" in html
    # The histogram is drawn from the same detected gaps, and cannot re-detect.
    assert "function gapHistogramBuckets()" in html
    assert "function renderGapHistogram()" in html
    assert '$("gapShorter").addEventListener("input", updateGapControls)' in html


# ---------------------------------------------------------------------------
# Phone transfer
# ---------------------------------------------------------------------------

def test_a_transfer_opens_its_file_and_reads_its_descriptor_once() -> None:
    """Per-chunk disk work is what a lane actually spends its time on.

    Re-reading the session JSON and reopening the partial file on every chunk
    put two synchronous disk operations on the event loop per chunk, which
    every lane then queued behind.
    """
    source = Path(retake.__file__).read_text(encoding="utf-8")
    chunk = source[source.index('@app.post("/upload/chunk/{session_id}")'):]
    chunk = chunk[:chunk.index("@app.post(\"/upload/finish\")")]
    assert "_UPLOAD_META" in chunk, "the descriptor should be served from memory"
    assert "meta_path.read_text" in chunk  # only as the cold-start fallback
    assert chunk.count("meta_path.read_text") == 1
    assert "open(partial" not in chunk, "the session's file stays open"
    # The shared handle is positional, so lanes never reorder one another.
    assert "def _write_chunk_at(session_id: str, path: Path" in source
    assert "handle.seek(offset)" in source


def test_a_transfer_releases_its_handle_before_the_file_moves() -> None:
    """Windows will not rename or unlink a file that is still open."""
    source = Path(retake.__file__).read_text(encoding="utf-8")
    finish = source[source.index('@app.post("/upload/finish")'):]
    finish = finish[:finish.index('@app.post("/upload/cancel")')]
    assert finish.index("_close_upload_handle") < finish.index("os.replace")
    cancel = source[source.index('@app.post("/upload/cancel")'):]
    cancel = cancel[:cancel.index('@app.api_route("/media"')]
    assert cancel.index("_forget_upload_session") < cancel.index("shutil.rmtree")
    assert "_close_upload_handle(session_id)" in source


# --------------------------------------------------------------------------
# Joins in the preview, Apple Lossless playback, second alignment pass
# --------------------------------------------------------------------------

@contextmanager
def _current_project(project: dict, track=None):
    old_current = dict(retake.CURRENT)
    old_load = retake.load_track
    try:
        retake.CURRENT.update(project=project, media_path=project.get("source_path"),
                              project_dir=None, source_filename="x")
        if track is not None:
            retake.load_track = lambda _path: track
        yield TestClient(app)
    finally:
        retake.load_track = old_load
        retake.CURRENT.clear()
        retake.CURRENT.update(old_current)


def test_joins_endpoint_plans_exactly_the_edit_it_is_sent() -> None:
    """Preview asks for the edit it is showing, not whatever was last saved."""
    track = _synthetic_track(20.0, [(2.0, 3.0), (3.8, 4.6), (5.4, 6.4)])
    project = {
        "duration_s": 20.0, "source_path": "unused.wav",
        "probe": {"acodec": "pcm_s16le"},
        "alignment": {"status": "aligned", "version": retake.ALIGNMENT_VERSION, "coverage": 1.0},
        "tokens": [_word(1, 2.0, 3.0), _word(2, 3.8, 4.6), _word(3, 5.4, 6.4)],
        "audio_gaps": [],
    }
    with _current_project(project, track) as client:
        nothing = client.post("/joins", json={"cut_ids": []}).json()
        assert nothing["available"] and nothing["keeps"] == [[0.0, 20.0]]
        cut = client.post("/joins", json={"cut_ids": [2]}).json()
        kept_end = next(end for _start, end in cut["keeps"] if 3.0 <= end < 3.8)
        next_start = next(start for start, _end in cut["keeps"] if 4.6 < start <= 5.4)
        assert kept_end < next_start
        assert not any(join["on_sound"] for join in cut["joins"])
        # The saved project itself was not touched.
        assert not any(token.get("cut") for token in project["tokens"])


def test_joins_endpoint_says_so_when_there_is_no_audio() -> None:
    project = {"duration_s": 5.0, "source_path": "x.mp4", "probe": {"acodec": None},
               "tokens": [_word(1, 1.0, 2.0)], "audio_gaps": []}
    with _current_project(project) as client:
        assert client.post("/joins", json={"cut_ids": [1]}).json() == {"available": False}


def test_overview_is_the_whole_recording_normalised() -> None:
    track = _synthetic_track(10.0, [(2.0, 3.0)])
    project = {"duration_s": 10.0, "source_path": "unused.wav", "probe": {"acodec": "aac"},
               "tokens": [], "audio_gaps": []}
    with _current_project(project, track) as client:
        peaks = client.get("/overview?points=200").json()["peaks"]
    assert len(peaks) == 200 and max(peaks) == 1.0
    assert peaks[50] > 0.5 and peaks[150] < 0.05      # loud at 2.5 s, room tone at 7.5 s


def test_apple_lossless_audio_gets_a_playable_copy_and_nothing_else_does() -> None:
    def project(codec, video=False):
        return {"probe": {"acodec": codec, "has_video": video}}
    assert retake.needs_listen_copy(project("alac"))
    for codec in ("aac", "mp3", "flac", "opus", "pcm_s16le"):
        assert not retake.needs_listen_copy(project(codec))
    assert not retake.needs_listen_copy(project("alac", video=True))   # Smooth Preview covers video
    assert not retake.needs_listen_copy(None)


def test_media_serves_the_playable_copy_for_apple_lossless() -> None:
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "original.m4a"
        source.write_bytes(b"alac")
        copy_path = Path(folder) / "listen.flac"
        copy_path.write_bytes(b"fLaC-bytes")
        project = {"duration_s": 1.0, "source_path": str(source),
                   "probe": {"acodec": "alac", "has_video": False}, "tokens": []}
        old_ensure = retake.ensure_listen_copy
        retake.ensure_listen_copy = lambda _proj: copy_path
        try:
            with _current_project(project) as client:
                response = client.get("/media")
        finally:
            retake.ensure_listen_copy = old_ensure
        assert response.status_code == 200
        assert response.content == b"fLaC-bytes"
        assert response.headers["content-type"].startswith("audio/flac")


def test_second_pass_windows_are_walled_by_aligned_neighbours() -> None:
    proj = {"duration_s": 12.0, "tokens": [
        _word(1, 1.0, 1.5), _word(2, 1.6, 2.0, aligned=False), _word(3, 2.1, 2.4, aligned=False),
        _word(4, 3.0, 3.5), _word(5, 3.5, 3.52, aligned=False), _word(6, 3.53, 4.0),
        _word(7, 9.0, 9.4, aligned=False)]}
    windows = retake.unaligned_word_windows(proj)
    assert [(w["start"], w["end"], w["ids"]) for w in windows] == [
        (1.5, 3.0, [2, 3]),       # between w1 and w4
        (4.0, 12.0, [7]),         # after w6, to the end
    ]
    # w5 sits in a 30 ms slot between two aligned words: nothing to search.


def test_second_pass_accepts_only_believable_in_order_timings() -> None:
    proj = {"duration_s": 30.0, "tokens": [
        _word(1, 1.0, 1.5), _word(2, 1.6, 2.0, aligned=False), _word(3, 2.1, 2.4, aligned=False),
        _word(4, 3.0, 3.5)]}
    windows = retake.unaligned_word_windows(proj)
    rescued = retake.apply_rescued_word_timings(proj, windows, [[
        {"word": "w2", "start": 1.7, "end": 2.05, "score": 0.6},
        {"word": "w3", "start": 2.2, "end": 2.6, "score": 0.1},     # too doubtful
    ]])
    assert rescued == 1
    w2, w3 = proj["tokens"][1], proj["tokens"][2]
    assert (w2["start"], w2["end"], w2["alignment_pass"]) == (1.7, 2.05, 2)
    assert "alignment_score" not in w3 and w3["start"] == 2.1
    # Out of order, or outside the walls: refused whole.
    proj["tokens"][2].pop("alignment_score", None)
    proj["tokens"][1].pop("alignment_score")
    windows = retake.unaligned_word_windows(proj)
    assert retake.apply_rescued_word_timings(proj, windows, [[
        {"word": "w2", "start": 2.3, "end": 2.5, "score": 0.9},
        {"word": "w3", "start": 1.6, "end": 1.9, "score": 0.9},
    ]]) == 0
    assert retake.apply_rescued_word_timings(proj, windows, [[
        {"word": "w2", "start": 0.5, "end": 0.9, "score": 0.9},
    ]]) == 0


def test_a_version_one_alignment_still_counts_but_offers_recalibration() -> None:
    proj = {"alignment": {"status": "aligned", "version": 1, "coverage": 0.95}}
    assert retake.project_uses_aligned_timing(proj)
    assert retake.ALIGNMENT_VERSION > 1


def test_preview_plays_the_planned_joins_and_checks_cuts_every_frame() -> None:
    html = Path(retake.INDEX_HTML).read_text(encoding="utf-8")
    assert 'api("/joins"' in html
    assert "S.planned.sig === sig" in html
    assert "requestAnimationFrame(frameTick)" in html
    assert 'id="stripCanvas"' in html and 'id="nextFlag"' in html


def test_real_projects_have_no_join_on_sound_that_had_a_quiet_spot() -> None:
    """The audit behind the join planner, run against the recordings on this machine.

    Slow (it decodes every edited project), so it runs only when asked:
    RETAKE_REAL_PROJECTS=1 python -m pytest -k real_projects
    A join may only sit on sound when the words either side were said in one
    breath; anything else is the planner missing quiet that was there.
    """
    import os
    import pytest
    if not os.environ.get("RETAKE_REAL_PROJECTS"):
        pytest.skip("set RETAKE_REAL_PROJECTS=1 to audit the projects on this machine")
    checked = 0
    for path in sorted(retake.PROJECTS_DIR.glob("Project */project.json")):
        proj = json.loads(path.read_text(encoding="utf-8"))
        if not any(token.get("cut") for token in proj.get("tokens", [])):
            continue
        if not Path(str(proj.get("source_path", ""))).is_file():
            continue
        plan = retake.planned_joins(proj)
        if plan is None:
            continue
        checked += 1
        missed = [join for join in plan["joins"] if join["on_sound"] and join["how"] != "glued"]
        assert not missed, f"{path.parent.name}: joins on sound with quiet nearby: {missed[:5]}"
        # With aligned timing, words run together are the only joins left on
        # sound, and there are few of them. A project never aligned is only as
        # good as Whisper's guesses until Recalibrate Timing is run on it.
        if plan["joins"] and retake.project_uses_aligned_timing(proj):
            share = sum(join["on_sound"] for join in plan["joins"]) / len(plan["joins"])
            assert share <= 0.10, f"{path.parent.name}: {share:.0%} of joins on sound"
    assert checked, "no edited projects with media were found"


def test_long_filter_graph_goes_to_a_script_file():
    import retake
    short = ["ffmpeg", "-filter_complex", "anull", "out.wav"]
    assert retake._spill_long_filter_graph(short) == (short, None)
    graph = "a" * (retake.FILTER_GRAPH_INLINE_LIMIT + 1)
    cmd, script = retake._spill_long_filter_graph(["ffmpeg", "-filter_complex", graph, "out.wav"])
    try:
        assert cmd == ["ffmpeg", "-filter_complex_script", str(script), "out.wav"]
        assert script.read_text(encoding="utf-8") == graph
    finally:
        script.unlink(missing_ok=True)


def test_an_unaligned_word_is_settled_between_its_aligned_neighbours() -> None:
    """The second pass placed "time." and left "the" on the wrong side of it."""
    tokens = [
        {"id": 1, "kind": "word", "text": "of", "start": 78.754, "end": 78.854, "alignment_score": 0.6},
        {"id": 2, "kind": "word", "text": "the", "start": 79.94, "end": 80.06},
        {"id": 3, "kind": "word", "text": "time.", "start": 79.195, "end": 79.356, "alignment_score": 0.4},
        {"id": 4, "kind": "word", "text": "one", "start": 84.0, "end": 84.2},
    ]
    assert retake.settle_rough_word_timings(tokens) == 1
    assert (tokens[1]["start"], tokens[1]["end"]) == (78.854, 79.195)
    # The last word, unaligned but already in order, is left where Whisper put it.
    assert (tokens[3]["start"], tokens[3]["end"]) == (84.0, 84.2)
    starts = [token["start"] for token in tokens]
    assert starts == sorted(starts)


def test_a_settled_word_keeps_its_length_instead_of_filling_the_pause() -> None:
    """ "is" was placed over the start of "B"; "B" must not swallow the next five seconds."""
    tokens = [
        {"id": 1, "kind": "word", "text": "is", "start": 331.999, "end": 332.199, "alignment_score": 0.9},
        {"id": 2, "kind": "word", "text": "B", "start": 332.05, "end": 332.4},
        {"id": 3, "kind": "word", "text": "so", "start": 337.794, "end": 338.015, "alignment_score": 0.9},
    ]
    assert retake.settle_rough_word_timings(tokens) == 1
    assert tokens[1]["start"] == 332.199
    assert tokens[1]["end"] - tokens[1]["start"] <= 0.36


def test_numbers_and_symbols_are_handed_to_the_aligner_as_spoken() -> None:
    """The aligner has no letters for "37"; it must hear "thirty seven"."""
    forms = retake.spoken_forms
    assert forms("37,") == ["thirty", "seven"]
    assert forms("77") == ["seventy", "seven"]
    assert forms("42.") == ["forty", "two"]
    assert forms("5%.") == ["five", "percent"]
    assert forms("%") == ["percent"]
    assert forms("200") == ["two", "hundred"]
    assert forms(",000") == ["thousand"]
    assert forms("200,000") == ["two", "hundred", "thousand"]
    assert forms("1992") == ["nineteen", "ninety", "two"]
    assert forms("2017") == ["twenty", "seventeen"]
    assert forms("2005") == ["two", "thousand", "five"]
    assert forms("3.5") == ["three", "point", "five"]
    assert forms("21st") == ["twenty", "first"]
    assert forms("r1") == ["ar", "one"]
    assert forms("-40") == ["forty"]
    assert forms("GPT-4o") == ["gee", "pee", "tee", "four", "oh"]
    assert forms("B") == ["bee"] and forms("D?") == ["dee"]
    assert forms("AI") == ["ay", "eye"] and forms("AIs") == ["ay", "eyes"]
    assert forms("A") == ["A"] and forms("I") == ["I"] and forms("NASA") == ["NASA"]
    assert forms("don't") == ["don't"]
    assert forms("time.") == ["time"]
    assert forms("...") == ["..."]
    # Other languages keep their words exactly; their aligners differ.
    assert forms("37", "ar") == ["37"]
    assert retake.spoken_text("were 37, 73, 7, 77") == (
        "were thirty seven seventy three seven seventy seven"
    )


def test_a_number_takes_its_time_from_every_word_it_is_said_as() -> None:
    """ "The top answers were 37, 73, 7, 77" -- the numbers run for four seconds."""
    project = {
        "duration_s": 140.0, "language": "en",
        "tokens": [
            {"id": 1, "kind": "word", "text": "were", "start": 131.586, "end": 131.826, "seg": 0, "cut": False},
            {"id": 2, "kind": "word", "text": "37,", "start": 131.846, "end": 131.906, "seg": 0, "cut": False},
            {"id": 3, "kind": "word", "text": "73,", "start": 132.246, "end": 132.366, "seg": 0, "cut": False},
            {"id": 4, "kind": "word", "text": "7,", "start": 132.406, "end": 132.507, "seg": 0, "cut": False},
            {"id": 5, "kind": "word", "text": "77", "start": 132.527, "end": 132.587, "seg": 0, "cut": False},
        ],
        "segments": [{"id": 0, "start": 131.586, "end": 132.587, "text": "were 37, 73, 7, 77"}],
    }
    spoken = [
        ("were", 131.6, 131.8), ("thirty", 132.0, 132.6), ("seven", 132.7, 133.3),
        ("seventy", 133.9, 134.5), ("three", 134.6, 135.0), ("seven", 135.5, 136.0),
        ("seventy", 136.6, 137.0), ("seven", 137.1, 137.5),
    ]
    result = {
        "device": "cuda", "elapsed_s": 1.0,
        "words": [{"word": w, "start": s, "end": e, "score": 0.9} for w, s, e in spoken],
    }
    candidate, stats = retake.apply_aligned_word_timings(project, result, fallback=False)
    timed = {t["text"]: (t["start"], t["end"]) for t in candidate["tokens"] if t["kind"] == "word"}
    assert timed["37,"] == (132.0, 133.3)
    assert timed["73,"] == (133.9, 135.0)
    assert timed["7,"] == (135.5, 136.0)
    assert timed["77"] == (136.6, 137.5)
    assert stats["coverage"] == 1.0


def test_a_number_half_placed_by_the_aligner_is_not_timed_by_that_half() -> None:
    timings = retake.aligned_token_timings(
        ["said", "37"],
        [{"word": "said", "start": 1.0, "end": 1.3, "score": 0.9},
         {"word": "thirty", "start": 1.4, "end": 1.8, "score": 0.9}],
        "en",
    )
    assert timings[0]["start"] == 1.0 and timings[1] is None


def test_alignment_windows_carry_spoken_text() -> None:
    project = {
        "duration_s": 10.0, "language": "en",
        "segments": [{"start": 1.0, "end": 3.0, "text": "Grok said 42."}],
    }
    assert retake.alignment_segments_for_project(project)[0]["text"] == "Grok said forty two"



# ---------------------------------------------------------------------------
# The waveform as ground truth
# ---------------------------------------------------------------------------

def wave_of(*bursts: tuple[float, float], seconds: float):
    audio = recording(*bursts, seconds=seconds)
    levels = transcription.frame_levels(audio)
    islands = transcription.audible_islands(levels, transcription.audible_threshold(levels))
    return audio, islands, transcription.smooth_levels(levels)


def test_a_word_tail_the_aligner_timed_short_is_given_back_to_the_word() -> None:
    # One word's sound runs 1.0-2.0; the aligner said 1.0-1.7.
    _audio, islands, smooth = wave_of((1.0, 2.0), (3.0, 4.0), seconds=5.0)
    words = [
        {"text": "hello", "start": 1.05, "end": 1.7},
        {"text": "there", "start": 3.2, "end": 4.0},
    ]
    transcription.snap_words_to_waveform(words, islands, smooth, 5.0)
    assert abs(words[0]["end"] - 2.0) < 0.02       # the tail is the word's
    assert abs(words[0]["start"] - 1.0) < 0.02     # and so is the onset
    assert abs(words[1]["start"] - 3.0) < 0.02
    assert transcription.unowned_wave(islands, words) == []


def test_words_in_unbroken_sound_meet_at_its_quietest_point() -> None:
    audio = recording((1.0, 3.0), seconds=4.0)
    rate = transcription.SAMPLE_RATE
    # A soft patch at 1.9-2.0 inside the sound: where a person would split.
    audio[int(1.9 * rate): int(2.0 * rate)] //= 20
    levels = transcription.frame_levels(audio)
    islands = transcription.audible_islands(levels, transcription.audible_threshold(levels))
    smooth = transcription.smooth_levels(levels)
    words = [{"text": "one", "start": 1.0, "end": 1.6}, {"text": "two", "start": 2.3, "end": 3.0}]
    transcription.snap_words_to_waveform(words, islands, smooth, 4.0)
    assert words[0]["end"] == words[1]["start"]
    assert 1.85 <= words[0]["end"] <= 2.05


def test_a_snapped_edge_never_reaches_past_a_tail_or_into_the_next_word() -> None:
    _audio, islands, smooth = wave_of((1.0, 3.0), (3.6, 4.0), seconds=5.0)
    words = [{"text": "so", "start": 1.0, "end": 1.4}, {"text": "yes", "start": 3.6, "end": 4.0}]
    transcription.snap_words_to_waveform(words, islands, smooth, 5.0)
    # 1.6 s of sound after "so" is not its tail: it moves 0.35 s at most.
    assert abs(words[0]["end"] - (1.4 + transcription.SNAP_TAIL_S)) < 0.01
    left = transcription.unowned_wave(islands, words)
    assert left and left[0][0] >= words[0]["end"] - 1e-6
    # What is left is long enough to be heard again for missing speech.
    assert transcription.stretches_to_hear(islands, words)


def test_a_word_the_aligner_could_not_place_is_laid_on_the_wave_between_its_neighbours() -> None:
    islands = [(1.0, 1.4), (1.55, 1.7), (1.8, 2.3)]
    words = [
        {"text": "for", "start": 1.0, "end": 1.4, "alignment_score": 0.9},
        {"text": "a", "start": 2.6, "end": 2.7},
        {"text": "random", "start": 1.8, "end": 2.3, "alignment_score": 0.9},
    ]
    assert retake.place_unaligned_on_waveform(words, islands) == 1
    assert (words[1]["start"], words[1]["end"]) == (1.55, 1.7)
    assert words[1]["waveform_timed"] is True
    assert retake.word_timing_is_rough(words[1]) is False


def test_coverage_counts_wave_owned_by_words_and_by_sounds() -> None:
    islands = [(0.0, 1.0), (2.0, 2.5), (3.0, 3.03)]
    words = [{"text": "hi", "start": 0.0, "end": 1.0}]
    report = transcription.wave_coverage(islands, words, [])
    assert report["owned"] < 1.0 and report["unowned_s"] == 0.5
    report = transcription.wave_coverage(islands, words, [(2.0, 2.5)])
    # The 30 ms flicker is room tone, not something to own.
    assert report["owned"] == 1.0


def test_waveform_pass_leaves_no_wave_unowned_and_every_word_timed() -> None:
    audio, islands, _smooth = wave_of((1.0, 2.0), (2.6, 2.9), (4.0, 5.0), seconds=6.0)
    draft = {
        "duration_s": 6.0, "language": "en",
        "tokens": [
            {"id": 0, "kind": "word", "text": "hello", "start": 1.05, "end": 1.7, "seg": 0, "cut": False, "alignment_score": 0.9},
            {"id": 1, "kind": "word", "text": "a", "start": 3.5, "end": 3.6, "seg": 0, "cut": False},
            {"id": 2, "kind": "word", "text": "world", "start": 4.1, "end": 4.9, "seg": 1, "cut": False, "alignment_score": 0.8},
        ],
        "segments": [],
    }
    tokens, segments, sounds, report = retake.waveform_pass(draft, islands, audio, None)
    words = [t for t in tokens if t["kind"] == "word"]
    assert [w["text"] for w in words] == ["hello", "a", "world"]
    assert report["timed_from_audio"] == 1.0
    assert report["owned"] == 1.0 and sounds == []
    assert words[1]["waveform_timed"] and abs(words[1]["start"] - 2.6) < 0.02
    assert all(b["start"] >= a["end"] - 1e-6 for a, b in zip(words, words[1:]))
    assert [s["text"] for s in segments] == ["hello a", "world"]


def test_unowned_stretches_never_bridge_across_a_word() -> None:
    islands = [(1.0, 1.9), (1.9, 2.4)]
    words = [{"text": "but", "start": 1.6, "end": 1.7}, {"text": "if", "start": 1.75, "end": 1.85}]
    stretches = transcription.stretches_to_hear(islands, words)
    assert all(not (a < 1.6 and b > 1.7) for a, b in stretches)


def test_rehearing_adds_only_words_that_are_not_already_there() -> None:
    """Whisper re-hears "Grok" early, as "Grox"; it is the next word, not a new one."""
    audio = recording((1.0, 1.6), (2.0, 2.6), (3.0, 3.6), seconds=4.0)
    words = [
        {"text": "course,", "start": 1.0, "end": 1.6},
        {"text": "Grok", "start": 3.0, "end": 3.6},
    ]
    model = ScriptedWhisper([HeardSegment([
        ("course,", 0.0, 0.6), ("Grox.", 1.1, 1.5), ("Grok", 2.0, 2.6),
    ], 0.02)])
    assert transcription.hear_in_context(model, audio, "en", (2.0, 2.6), words, 4.0) == []
    model = ScriptedWhisper([HeardSegment([
        ("course,", 0.0, 0.6), ("um,", 1.1, 1.5), ("well", 1.5, 1.6), ("Grok", 2.0, 2.6),
    ], 0.02)])
    fresh = transcription.hear_in_context(model, audio, "en", (2.0, 2.6), words, 4.0)
    assert [w["text"] for w in fresh] == ["um,", "well"]


def test_rehearing_never_adds_punctuation_as_a_word() -> None:
    audio = recording((1.0, 1.6), (2.0, 2.6), seconds=3.0)
    words = [{"text": "wait", "start": 1.0, "end": 1.6}]
    model = ScriptedWhisper([HeardSegment([("wait", 0.0, 0.6), ("-", 1.1, 1.2)], 0.02)])
    assert transcription.hear_in_context(model, audio, "en", (2.0, 2.6), words, 3.0) == []
