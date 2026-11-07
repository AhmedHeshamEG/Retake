"""Where exactly a cut goes: read the waveform, the way a person zoomed in would.

The transcript says *which* pause a join belongs in -- between this kept word
and that deleted one. Word timings are estimates, off by tens of milliseconds
when forced alignment worked and by hundreds when it did not, so they are only
ever used as walls around a search. Inside those walls the join goes where the
audio is actually quiet:

* a keep that *ends* stops a short handle after its last word has died away,
  at the start of the first quiet stretch;
* a keep that *starts* begins a short handle before its first word -- and
  before any breath leading into it -- at the end of the last quiet stretch.

A breath is louder than the quiet threshold, so it is never split: it is either
inside the kept handle's side of the quiet or entirely in the cut. When there
is no quiet at all, because two words were said in one breath, the join goes to
the quietest instant between them and the export fades across it.

Everything here is pure numpy on one decoded track, so preview, export and the
self-check all measure the same thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
HOP_S = 0.005                     # one level every 5 ms
WINDOW_S = 0.010                  # each level is the RMS of 10 ms around it
HOP = int(SAMPLE_RATE * HOP_S)
HALF_WINDOW = int(SAMPLE_RATE * WINDOW_S) // 2

# The shortest quiet stretch that counts as a place to cut.
QUIET_MIN_S = 0.03
# Forced-aligned walls are trusted to about this; a search may look this far
# past them before deciding there is no quiet at all.
PRECISE_REACH_S = 0.03
# A wall from a word alignment could not confirm is a guess. The search may
# reach this far past it, but only once the walled-in pause has no quiet in it.
ROUGH_REACH_S = 0.15
# A join this far above the quiet threshold is on sound and is reported.
LOUD_MARGIN_DB = 6.0
# How much audio either side of a join decides whether it is on sound.
SPLICE_RADIUS_S = 0.010


@dataclass
class Track:
    """Levels of one recording, and what counts as quiet in it."""

    db: np.ndarray            # level at t = index * HOP_S
    duration: float
    noise_floor_db: float
    speech_level_db: float
    quiet_db: float           # below this, a dip between words is a place to cut
    silence_db: float         # below this, it is dead air worth shortening

    def index(self, t: float) -> int:
        return int(round(max(0.0, min(self.duration, t)) / HOP_S))

    def time(self, index: int) -> float:
        return min(self.duration, index * HOP_S)


def track_from_samples(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Track:
    """Build a track from mono samples (int16 or float in [-1, 1])."""
    audio = np.asarray(samples)
    if audio.dtype.kind in "iu":
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)
    if sample_rate != SAMPLE_RATE and len(audio):
        positions = np.arange(0, len(audio), sample_rate / SAMPLE_RATE)
        audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
    duration = len(audio) / SAMPLE_RATE
    if not len(audio):
        empty = np.full(1, -120.0)
        return Track(empty, 0.0, -120.0, -120.0, -60.0, -55.0)
    squared = np.concatenate([np.zeros(1, np.float64), np.cumsum(audio.astype(np.float64) ** 2)])
    centres = np.arange(0, len(audio) + 1, HOP)
    low = np.clip(centres - HALF_WINDOW, 0, len(audio))
    high = np.clip(centres + HALF_WINDOW, 0, len(audio))
    width = np.maximum(high - low, 1)
    rms = np.sqrt((squared[high] - squared[low]) / width)
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    noise_floor = float(np.percentile(db, 20))
    speech_level = float(np.percentile(db, 80))
    # The dip between two words rarely reaches the noise floor, but it sits well
    # under the speaker's own level. Whichever reference is more permissive
    # finds it on a quiet recording and on a noisy one.
    quiet = max(-60.0, min(-24.0, max(noise_floor + 6.0, speech_level - 25.0)))
    silence = max(-55.0, min(-32.0, noise_floor + 3.0))
    return Track(db, duration, noise_floor, speech_level, quiet, silence)


def quiet_runs(
    track: Track, low: float, high: float,
    threshold: Optional[float] = None, min_length: float = QUIET_MIN_S,
) -> list[tuple[float, float]]:
    """Every stretch inside [low, high] quieter than ``threshold``, in order."""
    if high <= low:
        return []
    limit = track.quiet_db if threshold is None else threshold
    first, last = track.index(low), track.index(high)
    if last <= first:
        return []
    quiet = track.db[first:last + 1] < limit
    runs: list[tuple[float, float]] = []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]])))
    for begin, end in zip(edges[::2], edges[1::2]):
        start = max(low, track.time(first + int(begin)))
        stop = min(high, track.time(first + int(end) - 1))
        if stop - start >= min_length - 1e-9:
            runs.append((start, stop))
    return runs


def quietest(track: Track, low: float, high: float) -> float:
    """The quietest instant in [low, high]; the middle of a tie."""
    first, last = track.index(low), track.index(high)
    if last <= first:
        return (low + high) / 2.0
    levels = track.db[first:last + 1]
    # A 15 ms smoothing keeps a single low sample inside a vowel from winning.
    if len(levels) >= 3:
        levels = np.convolve(levels, np.ones(3) / 3.0, mode="same")
    floor = float(levels.min())
    ties = np.flatnonzero(levels <= floor + 0.5)
    middle = ties[len(ties) // 2]
    return max(low, min(high, track.time(first + int(middle))))


@dataclass
class Placement:
    time: float
    how: str          # "quiet", "quiet-wide" (found past a rough wall), "glued"


def place_edge(
    track: Track, edge: str, low: float, high: float, handle: float,
    reach_low: float = PRECISE_REACH_S, reach_high: float = PRECISE_REACH_S,
) -> Placement:
    """Put one join inside the pause walled in by [low, high].

    ``edge`` is "end" for the end of a keep (the cut starts here) and "start"
    for the start of one (the cut ends here). ``reach_low``/``reach_high`` say
    how far past each wall the search may go when the walls themselves hold no
    quiet; a wall from an unconfirmed word gets the wide reach.
    """
    low, high = max(0.0, low), min(track.duration, high)
    if high < low:
        low = high = (low + high) / 2.0

    def from_runs(runs: list[tuple[float, float]]) -> float:
        # A handle from the speech side, but never past the middle of the
        # quiet: in a 40 ms dip the far edge is already the next word.
        if edge == "end":
            start, stop = runs[0]
            return min(start + handle, (start + stop) / 2.0)
        start, stop = runs[-1]
        return max(stop - handle, (start + stop) / 2.0)

    runs = quiet_runs(track, low, high)
    if runs:
        return Placement(from_runs(runs), "quiet")
    wide_low = max(0.0, low - reach_low)
    wide_high = min(track.duration, high + reach_high)
    runs = quiet_runs(track, wide_low, wide_high)
    if runs:
        return Placement(from_runs(runs), "quiet-wide")
    return Placement(quietest(track, wide_low, wide_high), "glued")


def splice_level(track: Track, t: float, radius: float = SPLICE_RADIUS_S) -> float:
    """The loudest level within ``radius`` of a join."""
    first, last = track.index(t - radius), track.index(t + radius)
    return float(track.db[first:last + 1].max()) if last >= first else -120.0


def is_on_sound(track: Track, t: float) -> bool:
    return splice_level(track, t) > track.quiet_db + LOUD_MARGIN_DB


def dead_air(track: Track, min_length: float) -> list[tuple[float, float]]:
    """Long stretches under the recording's own silence level."""
    return quiet_runs(track, 0.0, track.duration, track.silence_db, min_length)
