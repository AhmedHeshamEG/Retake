"""Transcription for Retake: measure the recording, then hear it one phrase at a time.

An improvised recording is mostly the same line said several times. Whisper,
handed a stretch that holds a line twice, treats the second take as a slip and
writes the line down once -- and it does not leave a hole where the other take
was, it draws one word of the take it kept right across it. Nothing downstream
can cut, keep or even show a take that has no words.

So the recording is measured first. Every stretch that rises above its own
noise floor is an *island*, and islands are joined into *phrases* only across
pauses shorter than a breath. Each phrase is decoded on its own, so two takes
separated by any real pause never share a decode.

Two takes said with no pause between them can still land in one phrase. That
shows up as a word drawn over far more sound than it could take to say, or as
sound with no word on it at all. The phrase is then split at the quietest
moment of that stretch -- only where the level genuinely drops -- and both
halves are decoded again in full. A half always carries real words with it,
which is what keeps Whisper from inventing a sentence for it.

What is kept is what Whisper itself says is speech. A breath or a click
decoded alone comes back as "Thank you." with a perfectly good log-probability;
its no-speech probability gives it away, and it is reported as a sound so the
editor can remove it.

The gate here is a measurement, not a judgement. It only ever sets aside audio
within a few decibels of the recording's own floor, and when it cannot find a
pause it splits at the quietest moment instead of dropping anything.
"""
from __future__ import annotations

import bisect
import difflib
import subprocess
from typing import Any, Callable, Optional

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 160                       # 10 ms
FRAMES_PER_S = SAMPLE_RATE // FRAME_SAMPLES

# Audible means this far above the quietest twentieth of the recording. The
# floor is clamped so a noise-gated phone recording, whose silence is digital
# zero, does not make every trace of room tone count as sound; the headroom
# keeps the threshold under the speech itself when the room is loud.
AUDIBLE_MARGIN_DB = 10.0
AUDIBLE_FLOOR_DB = -60.0
AUDIBLE_HEADROOM_DB = 12.0
# A dip shorter than this is the inside of a word, not a pause.
MIN_SILENCE_S = 0.15
MIN_ISLAND_S = 0.03

# A phrase is a run of islands with no pause of PHRASE_BREAK_S or more inside
# it. Retakes are almost always separated by at least that much, so each one
# reaches Whisper alone. Speech that never pauses is split at a trough so no
# phrase outgrows what Whisper decodes at once.
PHRASE_BREAK_S = 0.35
PHRASE_MAX_S = 24.0
PHRASE_PAD_S = 0.2
# Kept for the trough splitter, which works on any over-long island.
PIECE_MAX_S = PHRASE_MAX_S

# How many phrases, spread across the recording, vote on its language, and
# how long one must be to get a vote: a breath has no language.
LANGUAGE_WINDOWS = 6
LANGUAGE_MIN_S = 2.0

# How far a word's reported timing is trusted to reach past its edges.
WORD_REACH_S = 0.12
# Sound with no word on it is worth splitting a phrase for from this length.
# Sound that runs straight on from a word needs to be much longer, because the
# end of a word -- a spoken number above all -- is routinely timed short.
FREE_STRETCH_MIN_S = 0.4
ATTACHED_STRETCH_MIN_S = 1.2
STRETCH_BRIDGE_S = 0.4

# When Whisper drops a take it draws one word of the take it kept across the
# whole of the one it dropped, so a two-second "random" sits on top of a
# missing "we pick the number that looks". A word is only believed for as long
# as it could take to say, plus this slack.
WORD_BASE_S = 0.45
WORD_PER_LETTER_S = 0.07
WORD_PER_OTHER_LETTER_S = 0.2     # scripts that write a syllable or more per character
WORD_PER_DIGIT_S = 0.45           # "2017" is one token and four spoken words
WORD_SYMBOLS = "%$€£&@+=#°"       # each of these is a whole spoken word too
WORD_SLACK_S = 0.3

# Splitting a phrase to recover a hidden take. Each half keeps at least
# SPLIT_MIN_SIDE_S of audio, so neither is a scrap Whisper must invent words
# for; the split point must sit SPLIT_DIP_DB under the phrase's speech, so a
# drawn-out vowel is never cut in two; and a split is only kept when the two
# halves, heard in full, hold more words than the whole did.
SPLIT_MIN_SIDE_S = 0.5
SPLIT_SEARCH_S = 0.4
SPLIT_DIP_DB = 12.0
SPLIT_MAX_DEPTH = 3

# Whisper's no-speech probability, read directly. Breath and handling noise
# decoded alone score 0.8 and up; speech, even a clipped false start, stays
# under 0.7.
NO_SPEECH_MAX = 0.72
# Whisper's stock inventions for noise. Dropped when Whisper is even a little
# unsure the clip is speech; a clearly spoken "thank you" is kept.
PHANTOM_PHRASES = frozenset({
    "thankyou", "thanks", "thankyouverymuch", "thankyouforwatching",
    "thanksforwatching", "thankyousomuchforwatching", "pleasesubscribe",
    "subtitlesbytheamaraorgcommunity", "you", "bye", "amen", "ooh", "oh",
    "beep", "hmm", "mm", "uh", "um",
})
PHANTOM_NO_SPEECH_MIN = 0.3

# An island with no word on it is a sound worth showing from this length.
SOUND_MIN_S = 0.06
SOUND_WORD_REACH_S = 0.05

DECODE_OPTIONS: dict[str, Any] = dict(
    word_timestamps=True,
    # The VAD used to run *before* Whisper and delete anything it judged
    # non-speech, so a sentence begun softly never reached the model. It stays
    # off: what is set aside here is measured silence, nothing else.
    vad_filter=False,
    # A wider beam is the one decoding knob that buys accuracy outright.
    beam_size=10,
    # Each phrase is decoded on its own, so a take cannot inherit its
    # neighbour's wording or loop on its mistake.
    condition_on_previous_text=False,
)
# Whisper discards nothing itself: a phrase can be a single short word, and
# its own thresholds would drop it. The no-speech probability is read here
# instead, which is the one signal that separates a clipped retake from a cough.
PHRASE_OPTIONS: dict[str, Any] = dict(
    no_speech_threshold=None,
    log_prob_threshold=None,
)

Span = tuple[float, float]


def decode_audio(ffmpeg: str, media_path: str) -> np.ndarray:
    """The recording as 16 kHz mono int16, which is what Whisper hears."""
    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-nostdin", "-i", media_path, "-vn",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-",
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[-400:]
        raise RuntimeError(f"could not read the audio: {detail or 'ffmpeg failed'}")
    return np.frombuffer(result.stdout, np.int16)


def frame_levels(audio: np.ndarray) -> np.ndarray:
    """RMS level in dBFS of every 10 ms frame."""
    count = len(audio) // FRAME_SAMPLES
    frames = audio[: count * FRAME_SAMPLES].reshape(count, FRAME_SAMPLES)
    levels = np.empty(count, np.float32)
    step = 60000  # ten minutes at a time, so the float copy stays small
    for index in range(0, count, step):
        block = frames[index:index + step].astype(np.float32) / 32768.0
        levels[index:index + step] = 10.0 * np.log10((block ** 2).mean(axis=1) + 1e-10)
    return levels


def smooth_levels(levels: np.ndarray) -> np.ndarray:
    """Levels averaged over 50 ms, so a trough is a pause and not one quiet frame."""
    if not len(levels):
        return levels
    return np.convolve(levels, np.ones(5, np.float32) / 5.0, mode="same")


def audible_threshold(levels: np.ndarray) -> float:
    if not len(levels):
        return AUDIBLE_FLOOR_DB
    floor = float(np.percentile(levels, 5))
    top = float(np.percentile(levels, 95))
    return min(max(floor + AUDIBLE_MARGIN_DB, AUDIBLE_FLOOR_DB), top - AUDIBLE_HEADROOM_DB)


def audible_islands(levels: np.ndarray, threshold: float) -> list[Span]:
    """Every stretch above the threshold, in seconds, pauses excluded."""
    loud = (levels > threshold).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], loud, [0]))))
    merged: list[list[int]] = []
    for start, end in zip(edges[::2].tolist(), edges[1::2].tolist()):
        if merged and start - merged[-1][1] < MIN_SILENCE_S * FRAMES_PER_S:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [
        (start / FRAMES_PER_S, end / FRAMES_PER_S)
        for start, end in merged
        if end - start >= MIN_ISLAND_S * FRAMES_PER_S
    ]


def _split_unbroken(islands: list[Span], levels: np.ndarray) -> list[Span]:
    """Break an island too long for one decode at its quietest moments.

    Only speech with no pause at all for PHRASE_MAX_S gets here. The cut goes
    to the lowest point of the last stretch that still fits, which is the
    trough between two words rather than the middle of one.
    """
    smooth = smooth_levels(levels)
    out: list[Span] = []
    for start, end in islands:
        while end - start > PHRASE_MAX_S:
            low = int((start + PHRASE_MAX_S * 0.55) * FRAMES_PER_S)
            high = int((start + PHRASE_MAX_S) * FRAMES_PER_S)
            cut = (low + int(np.argmin(smooth[low:high]))) / FRAMES_PER_S
            out.append((start, cut))
            start = cut
        out.append((start, end))
    return out


def plan_phrases(islands: list[Span], levels: np.ndarray, total: float) -> list[Span]:
    """Join islands across pauses shorter than a breath; pad each with real silence."""
    groups: list[list[Span]] = []
    for island in _split_unbroken(islands, levels):
        current = groups[-1] if groups else None
        if (
            current is not None
            and island[0] - current[-1][1] < PHRASE_BREAK_S
            and island[1] - current[0][0] <= PHRASE_MAX_S
        ):
            current.append(island)
        else:
            groups.append([island])

    phrases: list[Span] = []
    for index, group in enumerate(groups):
        start, end = group[0][0], group[-1][1]
        before = start - (groups[index - 1][-1][1] if index else 0.0)
        after = (groups[index + 1][0][0] if index + 1 < len(groups) else total) - end
        # Padding is real audio from the pause on either side, never more than
        # half of it, so two phrases cannot both claim the same moment.
        left = min(PHRASE_PAD_S, before / 2 if index else before)
        right = min(PHRASE_PAD_S, after / 2 if index + 1 < len(groups) else after)
        phrases.append((max(0.0, start - left), min(total, end + right)))
    return phrases


def plausible_word_seconds(text: str) -> float:
    """The longest a word could reasonably take to say."""
    seconds = WORD_BASE_S
    for character in text:
        if character.isdigit() or character in WORD_SYMBOLS:
            seconds += WORD_PER_DIGIT_S
        elif character.isalpha():
            seconds += WORD_PER_LETTER_S if character.isascii() else WORD_PER_OTHER_LETTER_S
    return seconds


def audible_seconds(islands: list[Span], start: float, end: float) -> float:
    """How much of ``start``..``end`` is sound rather than pause."""
    index = bisect.bisect_right(islands, (start, float("inf"))) - 1
    total = 0.0
    for island_start, island_end in islands[max(index, 0):]:
        if island_start >= end:
            break
        total += max(0.0, min(island_end, end) - max(island_start, start))
    return total


def word_is_stretched(islands: list[Span], word: dict[str, Any]) -> bool:
    """Whether a word covers more sound than saying it could account for.

    A word drawn out across a pause is not stretched in this sense: only the
    sound under it is counted, so "okay" followed by two seconds of nothing
    is still just "okay".
    """
    return audible_seconds(islands, float(word["start"]), float(word["end"])) > (
        plausible_word_seconds(str(word["text"])) + WORD_SLACK_S
    )


def wordless_stretches(
    islands: list[Span], words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Audible stretches no word accounts for, long enough to matter.

    Pass only the words that are believed: a stretched word left out of
    ``words`` turns the sound under it into a stretch of its own.

    ``attached`` marks a stretch that runs straight on from a word inside the
    same island; ``audible`` is how much of the stretch is sound, since a free
    stretch may bridge short pauses between islands that belong together.
    """
    spans = sorted((float(word["start"]), float(word["end"])) for word in words)
    starts = [span[0] for span in spans]
    raw: list[dict[str, Any]] = []
    for island_start, island_end in islands:
        index = bisect.bisect_left(starts, island_start)
        while index > 0 and spans[index - 1][1] + WORD_REACH_S > island_start:
            index -= 1
        cursor, attached = island_start, False
        for word_start, word_end in spans[index:]:
            if word_start - WORD_REACH_S >= island_end:
                break
            if word_end + WORD_REACH_S <= cursor:
                continue
            if word_start - WORD_REACH_S > cursor:
                raw.append({
                    "start": cursor, "end": word_start - WORD_REACH_S,
                    "attached": True,
                })
            cursor = max(cursor, word_end + WORD_REACH_S)
            attached = True
        if cursor < island_end:
            raw.append({"start": cursor, "end": island_end, "attached": attached})

    stretches: list[dict[str, Any]] = []
    for hole in raw:
        hole["audible"] = hole["end"] - hole["start"]
        last = stretches[-1] if stretches else None
        if (
            last is not None
            and not last["attached"] and not hole["attached"]
            and hole["start"] - last["end"] < STRETCH_BRIDGE_S
        ):
            last["end"] = hole["end"]
            last["audible"] += hole["audible"]
        else:
            stretches.append(hole)
    return [
        stretch for stretch in stretches
        if stretch["audible"] >= (
            ATTACHED_STRETCH_MIN_S if stretch["attached"] else FREE_STRETCH_MIN_S
        )
    ]


def unaccounted_stretches(
    islands: list[Span], words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Every stretch of sound the words do not explain, in order.

    Each carries ``under``: the stretched words lying across it. A run of
    stretched words is always one stretch, whole.
    """
    believed: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    open_region = False
    for word in sorted(words, key=lambda item: item["start"]):
        if not word_is_stretched(islands, word):
            believed.append(word)
            open_region = False
        elif open_region:
            regions[-1]["end"] = max(regions[-1]["end"], word["end"])
            regions[-1]["under"].append(word)
        else:
            regions.append({"start": word["start"], "end": word["end"], "under": [word]})
            open_region = True

    stretches: list[dict[str, Any]] = []
    for stretch in wordless_stretches(islands, believed):
        region = next(
            (
                region for region in regions
                if stretch["start"] < region["end"] + WORD_REACH_S
                and stretch["end"] > region["start"] - WORD_REACH_S
            ),
            None,
        )
        if region is None:
            stretches.append({"start": stretch["start"], "end": stretch["end"], "under": []})
        else:
            region["start"] = min(region["start"], stretch["start"])
            region["end"] = max(region["end"], stretch["end"])
    stretches.extend(regions)
    stretches.sort(key=lambda stretch: stretch["start"])
    return stretches


def islands_within(islands: list[Span], span: Span) -> list[Span]:
    """The islands clipped to ``span``."""
    start, end = span
    return [
        (max(a, start), min(b, end)) for a, b in islands
        if b > start and a < end and min(b, end) > max(a, start)
    ]


def split_point(
    islands: list[Span], smooth: np.ndarray, span: Span, words: list[dict[str, Any]],
) -> Optional[float]:
    """Where to split a phrase so a take it is hiding gets heard, or None.

    A stretched word is split inside itself, at its deepest pause: that pause
    is the seam between the take Whisper kept and the one it drew over. Sound
    with no word on it is split where it begins, so the half it lands in still
    starts with words Whisper has heard, not with a scrap.
    """
    start, end = span
    low_limit, high_limit = start + SPLIT_MIN_SIDE_S, end - SPLIT_MIN_SIDE_S
    if high_limit - low_limit < 0.1:
        return None
    inside = islands_within(islands, span)
    first = int(start * FRAMES_PER_S)
    last = min(len(smooth), int(end * FRAMES_PER_S))
    if last - first < 2:
        return None
    speech = smooth[first:last]
    speech_level = float(np.percentile(speech, 80))

    best: Optional[tuple[float, float]] = None   # (level, time)
    for stretch in unaccounted_stretches(inside, words):
        if stretch["under"]:
            low, high = stretch["start"], stretch["end"]
        else:
            low, high = stretch["start"] - SPLIT_SEARCH_S, stretch["start"] + SPLIT_SEARCH_S
        low, high = max(low, low_limit), min(high, high_limit)
        a, b = int(low * FRAMES_PER_S), int(high * FRAMES_PER_S)
        if b - a < 2:
            continue
        index = a + int(np.argmin(smooth[a:b]))
        level = float(smooth[index])
        if speech_level - level < SPLIT_DIP_DB:
            continue  # no real pause here; cutting would split a word
        if best is None or level < best[0]:
            best = (level, index / FRAMES_PER_S)
    return None if best is None else best[1]


def _plain(text: str) -> str:
    return "".join(character for character in text.lower() if character.isalnum())


def same_word(one: str, other: str) -> bool:
    """Loosely: "hit" for "hits" is one word heard twice, not two words."""
    one, other = _plain(one), _plain(other)
    return one == other or (
        min(len(one), len(other)) >= 3
        and (one.startswith(other) or other.startswith(one))
    )


def is_phantom(segment: dict[str, Any]) -> bool:
    """Whisper's stock noise caption, said with any doubt that it is speech."""
    text = _plain("".join(word["text"] for word in segment["words"]))
    return text in PHANTOM_PHRASES and float(segment["no_speech_prob"]) >= PHANTOM_NO_SPEECH_MIN


def is_speech(segment: dict[str, Any]) -> bool:
    return float(segment["no_speech_prob"]) < NO_SPEECH_MAX and not is_phantom(segment)


def _decode(
    model: Any, audio: np.ndarray, pieces: list[Span], language: str,
    options: dict[str, Any],
) -> list[dict[str, Any]]:
    """Decode pieces as one clip and put every word back on the recording."""
    clips: list[np.ndarray] = []
    offsets: list[float] = []
    at = 0.0
    for start, end in pieces:
        clip = audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
        clips.append(clip)
        offsets.append(at)
        at += len(clip) / SAMPLE_RATE
    samples = np.concatenate(clips).astype(np.float32) / 32768.0
    if not len(samples):
        return []
    found, _info = model.transcribe(
        samples, language=language, **DECODE_OPTIONS, **options
    )
    segments: list[dict[str, Any]] = []
    for segment in found:
        words: list[dict[str, Any]] = []
        for word in segment.words or []:
            text = word.word.strip()
            if not text:
                continue
            # A word belongs to the piece its middle falls in, and is held
            # inside that piece: the joins between pieces are not real time.
            middle = (float(word.start) + float(word.end)) / 2.0
            piece = max(0, bisect.bisect_right(offsets, middle) - 1)
            piece_start, piece_end = pieces[piece]
            start = min(max(piece_start + float(word.start) - offsets[piece], piece_start), piece_end)
            end = min(max(piece_start + float(word.end) - offsets[piece], start), piece_end)
            if end - start < 0.02:
                # Held against the edge of its piece. A word with no length
                # cannot be cut, so it keeps a sliver of the audio it is on.
                end = min(start + 0.02, piece_end)
                start = min(start, end - 0.02)
            words.append({
                "text": text, "start": start, "end": end,
                "probability": float(word.probability),
            })
        if words:
            segments.append({
                "words": words,
                "no_speech_prob": float(segment.no_speech_prob),
            })
    return segments


def detect_language(model: Any, audio: np.ndarray, windows: list[list[Span]]) -> str:
    """Ask Whisper what is spoken, across the recording rather than at its start.

    One window is a coin toss for an accented speaker: thirty seconds of
    English came back as Arabic at 0.71, and a wrong guess here does not cost
    a few words, it has the whole recording written out in the other language.
    The same recording is English in ten of its twelve windows, so several
    spread through it are asked and their probabilities added.
    """
    if not windows:
        return "en"
    count = min(LANGUAGE_WINDOWS, len(windows))
    chosen = [windows[index * len(windows) // count] for index in range(count)]
    totals: dict[str, float] = {}
    for window in chosen:
        samples = np.concatenate([
            audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
            for start, end in window
        ]).astype(np.float32) / 32768.0
        if not len(samples):
            continue
        _language, _probability, ranked = model.detect_language(audio=samples)
        for language, probability in ranked:
            totals[language] = totals.get(language, 0.0) + float(probability)
    return max(totals, key=totals.get) if totals else "en"


def non_speech_sounds(
    islands: list[Span], words: list[dict[str, Any]],
) -> list[Span]:
    """Islands no word touches: a breath, a click, a hum, a cough.

    Each is bounded by measured silence on both sides, so removing one cannot
    reach into a word. Sound that runs straight on from a word is left alone
    for the same reason -- it may be the end of the word.
    """
    spans = sorted((float(word["start"]), float(word["end"])) for word in words)
    starts = [span[0] for span in spans]
    sounds: list[Span] = []
    for start, end in islands:
        if end - start < SOUND_MIN_S:
            continue
        index = bisect.bisect_left(starts, end + SOUND_WORD_REACH_S)
        touched = False
        while index > 0:
            index -= 1
            if spans[index][1] + SOUND_WORD_REACH_S > start:
                touched = True
                break
            if spans[index][0] < start - 30.0:
                break
        if not touched:
            sounds.append((round(start, 3), round(end, 3)))
    return sounds


# ---------------------------------------------------------------------------
# The waveform as ground truth
#
# Editing by hand, the waveform says it all: a flat line is silence, anything
# else is the speaker. Every stretch of wave must belong to something -- a word
# or a sound -- and every word must sit on its own wave. These run once the
# words carry their final, aligned timings.
# ---------------------------------------------------------------------------

# Room-tone flicker this short is not a sound anyone would cut.
FLICKER_S = 0.06
# How far a word's edge may move out to meet the end of its wave. The aligner
# times consonant tails and breath releases short; past these it is not the
# word any more.
SNAP_TAIL_S = 0.35
SNAP_ONSET_S = 0.25
# Two words with unbroken sound between them meet at its quietest point --
# if that sound is short enough to be the join between them.
SNAP_JOIN_MAX_S = 0.7
# Wave with no word on it is heard again, with the sentence around it, from
# this much sound up.
UNOWNED_HEAR_MIN_S = 0.4
UNOWNED_BRIDGE_S = 0.15
HEAR_CONTEXT_S = 2.0


def islands_between(islands: list[Span], start: float, end: float) -> list[Span]:
    return islands_within(islands, (start, end))


def snap_words_to_waveform(
    words: list[dict[str, Any]], islands: list[Span], smooth: np.ndarray, total: float,
) -> int:
    """Move each word's edges out to where its wave really starts and ends.

    ``words`` is chronological and edited in place. Sound running on from a
    word's end becomes the end of that word; sound leading into its start
    becomes its start. Where the sound between two words never drops to
    silence, they meet at its quietest point -- the split a person would
    make looking at the wave. Returns how many edges moved.
    """
    moved = 0
    for index in range(len(words) + 1):
        before = words[index - 1] if index > 0 else None
        after = words[index] if index < len(words) else None
        low = float(before["end"]) if before is not None else 0.0
        high = float(after["start"]) if after is not None else total
        if high - low <= 0.0:
            continue
        runs = islands_between(islands, low, high)
        if not runs:
            continue
        whole = runs[0][0] <= low + 0.011 and runs[-1][1] >= high - 0.011 and len(runs) == 1
        if whole and before is not None and after is not None:
            if high - low > SNAP_JOIN_MAX_S:
                continue
            a, b = int(low * FRAMES_PER_S), max(int(high * FRAMES_PER_S), int(low * FRAMES_PER_S) + 1)
            trough = (a + int(np.argmin(smooth[a:b]))) / FRAMES_PER_S if b <= len(smooth) else (low + high) / 2
            trough = min(max(trough, low), high)
            before["end"], after["start"] = round(trough, 3), round(trough, 3)
            moved += 2
            continue
        first, last = runs[0], runs[-1]
        if before is not None and first[0] <= low + 0.011:
            reach = min(first[1], low + SNAP_TAIL_S, high)
            if reach > low:
                before["end"] = round(reach, 3)
                moved += 1
        if after is not None and last[1] >= high - 0.011:
            reach = max(last[0], high - SNAP_ONSET_S, float(before["end"]) if before is not None else 0.0)
            if reach < high:
                after["start"] = round(reach, 3)
                moved += 1
    return moved


def unowned_wave(
    islands: list[Span], words: list[dict[str, Any]], sounds: list[Span] = (),
    reach: float = 0.0,
) -> list[Span]:
    """Every stretch of wave that no word and no sound accounts for."""
    owned = sorted(
        [(float(w["start"]) - reach, float(w["end"]) + reach) for w in words]
        + [(float(a), float(b)) for a, b in sounds]
    )
    out: list[Span] = []
    index = 0
    for start, end in islands:
        cursor = start
        while index < len(owned) and owned[index][1] <= start:
            index += 1
        probe = index
        while probe < len(owned) and owned[probe][0] < end:
            a, b = owned[probe]
            if a > cursor:
                out.append((cursor, min(a, end)))
            cursor = max(cursor, b)
            probe += 1
        if cursor < end:
            out.append((cursor, end))
    return [(round(a, 3), round(b, 3)) for a, b in out if b - a > 1e-6]


def stretches_to_hear(islands: list[Span], words: list[dict[str, Any]]) -> list[Span]:
    """Unowned wave long enough that it may be speech nobody wrote down.

    Runs separated by less than a breath are one stretch; what counts is how
    much sound is in it.
    """
    spans = sorted((float(w["start"]), float(w["end"])) for w in words)
    starts = [span[0] for span in spans]

    def word_between(a: float, b: float) -> bool:
        index = bisect.bisect_left(starts, a - 1e-6)
        return index < len(spans) and spans[index][0] < b - 1e-6

    merged: list[list[float]] = []
    for start, end in unowned_wave(islands, words):
        # Runs a breath apart are one stretch -- never across a word.
        if (
            merged and start - merged[-1][1] < UNOWNED_BRIDGE_S
            and not word_between(merged[-1][1], start)
        ):
            merged[-1][1] = end
            merged[-1][2] += end - start
        else:
            merged.append([start, end, end - start])
    return [(a, b) for a, b, sound in merged if sound >= UNOWNED_HEAR_MIN_S]


def wave_coverage(
    islands: list[Span], words: list[dict[str, Any]], sounds: list[Span],
) -> dict[str, float]:
    """How much of the wave belongs to a word, to a sound, and to nothing."""
    audible = sum(b - a for a, b in islands if b - a >= FLICKER_S) or 1e-9
    by_words = audible - sum(b - a for a, b in unowned_wave(islands, words))
    loose = sum(
        b - a for a, b in unowned_wave(islands, words, sounds) if b - a >= FLICKER_S
    )
    return {
        "audible_s": round(audible, 2),
        "words_s": round(max(0.0, by_words), 2),
        "unowned_s": round(loose, 2),
        "owned": round(max(0.0, 1.0 - loose / audible), 4),
    }


def hear_in_context(
    model: Any, audio: np.ndarray, language: str, stretch: Span,
    words: list[dict[str, Any]], total: float,
) -> list[dict[str, Any]]:
    """Words said inside ``stretch`` that the transcript does not have.

    The stretch is decoded with the words around it, so Whisper hears a
    sentence and not a scrap; only words it times inside the stretch, beyond
    what the transcript already holds there, are new.
    """
    start, end = stretch
    earlier = [w for w in words if float(w["end"]) <= start + 0.05]
    later = [w for w in words if float(w["start"]) >= end - 0.05]
    clip_start = max(0.0, start - HEAR_CONTEXT_S)
    if earlier and float(earlier[-1]["start"]) >= start - HEAR_CONTEXT_S:
        clip_start = float(earlier[-1]["start"])
    clip_end = min(total, end + HEAR_CONTEXT_S)
    if later and float(later[0]["end"]) <= end + HEAR_CONTEXT_S:
        clip_end = float(later[0]["end"])
    decoded = _decode(model, audio, [(clip_start, clip_end)], language, PHRASE_OPTIONS)
    heard = [
        word for segment in decoded if is_speech(segment) for word in segment["words"]
        if _plain(str(word["text"]))   # a lone "-" or "..." is not a word
    ]
    known = [
        w for w in words
        if float(w["end"]) > clip_start and float(w["start"]) < clip_end
    ]
    # Line the new hearing up against the words already there. Whisper times
    # the words it re-hears loosely, and may spell them differently ("Grox"
    # for "Grok"), so a heard word that resembles a known one is that word.
    # Only what the hearing *inserts* between known words is new.
    known_keys = [_plain(str(w["text"])) for w in known]
    heard_keys = []
    for word in heard:
        key = _plain(str(word["text"]))
        alike = max(
            known_keys, key=lambda other: difflib.SequenceMatcher(None, key, other).ratio(),
            default=None,
        )
        if alike is not None and (
            same_word(key, alike) or difflib.SequenceMatcher(None, key, alike).ratio() >= 0.75
        ):
            key = alike
        heard_keys.append(key)
    fresh: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(None, known_keys, heard_keys, autojunk=False)
    for tag, _a1, _a2, b1, b2 in matcher.get_opcodes():
        if tag != "insert":
            continue
        for word in heard[b1:b2]:
            middle = (float(word["start"]) + float(word["end"])) / 2.0
            if start - 0.1 <= middle <= end + 0.1:
                fresh.append(word)
    return fresh


def _word_count(segments: list[dict[str, Any]]) -> int:
    return sum(len(segment["words"]) for segment in segments)


def transcribe_recording(
    model: Any,
    audio: np.ndarray,
    language: Optional[str] = None,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Words for everything spoken in ``audio``, plus the map they were checked against.

    ``on_progress`` receives a fraction from 0 to 1 and a line saying what is
    happening. The returned segments each hold chronological words, and the
    segments follow one another: no two words anywhere are out of order.
    """
    def report(fraction: float, message: str) -> None:
        if on_progress is not None:
            on_progress(min(1.0, max(0.0, fraction)), message)

    total = len(audio) / SAMPLE_RATE
    levels = frame_levels(audio)
    smooth = smooth_levels(levels)
    threshold = audible_threshold(levels)
    islands = audible_islands(levels, threshold)
    phrases = plan_phrases(islands, levels, total)
    if not language:
        voters = [p for p in phrases if p[1] - p[0] >= LANGUAGE_MIN_S] or phrases
        language = detect_language(model, audio, [[phrase] for phrase in voters])

    splits: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    def hear(span: Span, depth: int) -> list[dict[str, Any]]:
        decoded = _decode(model, audio, [span], language, PHRASE_OPTIONS)
        kept = [segment for segment in decoded if is_speech(segment)]
        for segment in decoded:
            if not is_speech(segment):
                dropped.append({
                    "start": round(segment["words"][0]["start"], 2),
                    "end": round(segment["words"][-1]["end"], 2),
                    "text": " ".join(word["text"] for word in segment["words"]),
                    "no_speech": round(float(segment["no_speech_prob"]), 2),
                })
        if depth >= SPLIT_MAX_DEPTH:
            return kept
        words = [word for segment in kept for word in segment["words"]]
        at = split_point(islands, smooth, span, words)
        if at is None:
            return kept
        halves = hear((span[0], at), depth + 1) + hear((at, span[1]), depth + 1)
        better = _word_count(halves) > len(words)
        splits.append({
            "start": round(span[0], 2), "end": round(span[1], 2), "at": round(at, 2),
            "was": " ".join(word["text"] for word in words),
            "heard": " ".join(
                word["text"] for segment in halves for word in segment["words"]
            ),
            "kept": better,
        })
        return halves if better else kept

    heard_segments: list[dict[str, Any]] = []
    spoken = sum(end - start for start, end in phrases) or 1.0
    done = 0.0
    for phrase in phrases:
        heard_segments.extend(hear(phrase, 0))
        done += phrase[1] - phrase[0]
        report(0.98 * done / spoken, f"transcribing… {phrase[1]:.0f}s / {total:.0f}s")
    report(1.0, "transcription complete")

    # Phrases never overlap and are heard in order, so sorting by each
    # segment's first word only settles words Whisper timed against a padded
    # edge. Inside a segment, words keep the order they were spoken in.
    heard_segments.sort(key=lambda segment: segment["words"][0]["start"])
    segments_out: list[dict[str, Any]] = []
    previous_end = 0.0
    for segment in heard_segments:
        words = []
        for word in segment["words"]:
            start = max(float(word["start"]), previous_end)
            end = max(float(word["end"]), start + 0.02)
            words.append({"text": word["text"], "start": start, "end": end})
            previous_end = end
        segments_out.append({"words": words})
    return {
        "language": language,
        "segments": segments_out,
        "islands": islands,
        "splits": splits,
        "dropped": dropped,
        "stats": {
            "threshold_db": round(threshold, 1),
            "audible_s": round(sum(end - start for start, end in islands), 2),
            "phrases": len(phrases),
            "splits_kept": sum(1 for split in splits if split["kept"]),
            "dropped_segments": len(dropped),
            "words": _word_count(segments_out),
        },
    }
