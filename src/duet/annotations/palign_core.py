# ==================================================================================================
#                          IPU FROM TOKENSALIGN (CORE)
# ==================================================================================================
"""
Core logic to derive IPU intervals from a SPPAS palign TextGrid "TokensAlign" tier.

This module contains no CLI code and does not read/write files.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# ==================================================================================================
#                                   CONSTANTS
# ==================================================================================================

DEFAULT_MIN_IPU_S: float = 0.01
DEFAULT_MIN_SILENCE_S: float = 0.20  # reasonable default; override as needed

SILENCE_LABEL: str = "#"
IPU_LABEL: str = "IPU"


# ==================================================================================================
#                                   TYPES
# ==================================================================================================

@dataclass(frozen=True)
class Interval:
    """Simple time interval with a label."""
    start: float
    end: float
    text: str

    def duration(self) -> float:
        return float(self.end - self.start)


# ==================================================================================================
#                                   HELPERS
# ==================================================================================================

def _is_token_in_ipu(
    token: str,
    *,
    include_laughter: bool,
    include_noise: bool,
    include_filled_pause: bool,
) -> bool:
    """
    Decide whether a token label counts as speech (part of an IPU).

    Rules:
    - Any "valid token" counts as IPU by default.
    - '@' counts only if include_laughter=True
    - '*' counts only if include_noise=True
    - 'fp' counts only if include_filled_pause=True
    - '#' is treated as silence (never in IPU)

    Notes
    -----
    This is intentionally conservative about silence: '#' is always silence.
    Everything else is considered a token unless explicitly excluded by flags.

    Usage example
    -------------
        ok = _is_token_in_ipu("hello",
                              include_laughter=False,
                              include_noise=False,
                              include_filled_pause=True)
    """
    t = token.strip()

    if t == "":
        # Empty labels (often appear in TextGrid gaps) are treated as silence.
        return False
    if t == SILENCE_LABEL:
        return False
    if t == "@":
        return bool(include_laughter)
    if t == "*":
        return bool(include_noise)
    if t.lower() == "fp":
        return bool(include_filled_pause)

    # Any other token is considered part of IPU
    return True


# ==================================================================================================
#                                   CORE LOGIC
# ==================================================================================================

def build_ipu_segments_from_tokens(
    tokens: Sequence[Interval],
    *,
    include_laughter: bool = False,
    include_noise: bool = False,
    include_filled_pause: bool = True,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
) -> List[Tuple[float, float]]:
    """
    Build IPU segments (start, end) from token intervals (TokensAlign).

    Parameters
    ----------
    tokens
        Token-level intervals (e.g., from TokensAlign). Assumed time-ordered.
    include_laughter
        If True, '@' is treated as speech and included in IPUs.
    include_noise
        If True, '*' is treated as speech and included in IPUs.
    include_filled_pause
        If True, 'fp' is treated as speech and included in IPUs.
    min_silence_s
        Adjacent IPUs separated by silence shorter than this are merged.

    Returns
    -------
    segments
        List of (start, end) for each IPU segment.

    Usage example
    -------------
        segments = build_ipu_segments_from_tokens(
            tokens,
            include_laughter=False,
            include_noise=False,
            include_filled_pause=True,
            min_silence_s=0.2,
        )
    """
    if min_silence_s < 0:
        raise ValueError("min_silence_s must be >= 0")

    # First pass: contiguous "in-IPU" runs based purely on token membership.
    raw_segments: List[Tuple[float, float]] = []
    current_start: Optional[float] = None
    current_end: Optional[float] = None

    for itv in tokens:
        # Token membership only decides whether we are "inside speech" right
        # now; it does not enforce minimum IPU or silence durations yet.
        in_ipu = _is_token_in_ipu(
            itv.text,
            include_laughter=include_laughter,
            include_noise=include_noise,
            include_filled_pause=include_filled_pause,
        )

        if in_ipu:
            if current_start is None:
                # Start a new run at the first speech-like token.
                current_start = itv.start
                current_end = itv.end
            else:
                # Extend current run; max() is robust to tiny timestamp overlap.
                current_end = max(current_end, itv.end)
        else:
            if current_start is not None and current_end is not None:
                # Speech run ended by a silence/non-speech token.
                raw_segments.append((current_start, current_end))
                current_start = None
                current_end = None

    if current_start is not None and current_end is not None:
        # Flush a trailing run if file ends while still inside speech.
        raw_segments.append((current_start, current_end))

    if not raw_segments:
        return []

    # Second pass: merge segments separated by < min_silence_s.
    merged: List[Tuple[float, float]] = [raw_segments[0]]
    for start, end in raw_segments[1:]:
        prev_start, prev_end = merged[-1]
        gap = float(start - prev_end)
        if gap < min_silence_s:
            # Gap too short to count as a real pause; merge into one IPU.
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            # Gap long enough to keep as distinct conversational turns/chunks.
            merged.append((start, end))

    return merged


def apply_min_ipu_and_render_full_tier(
    *,
    t_start: float,
    t_end: float,
    ipu_segments: Sequence[Tuple[float, float]],
    min_ipu_s: float = DEFAULT_MIN_IPU_S,
    ipu_label: str = IPU_LABEL,
    silence_label: str = SILENCE_LABEL,
) -> List[Interval]:
    """
    Enforce min IPU duration and render a full-coverage interval tier.

    Steps
    -----
    1) Drop any IPU segments shorter than min_ipu_s (treat them as silence).
    2) Create a full tier from t_start..t_end with alternating silence/IPU labels.

    Parameters
    ----------
    t_start, t_end
        Total timeline bounds for the output tier.
    ipu_segments
        Candidate IPU segments (start, end), typically from build_ipu_segments_from_tokens.
    min_ipu_s
        IPUs shorter than this are replaced by silence.
    ipu_label, silence_label
        Labels to use in the output tier.

    Returns
    -------
    intervals
        Full-coverage list of Interval objects.

    Usage example
    -------------
        tier = apply_min_ipu_and_render_full_tier(
            t_start=0.0,
            t_end=10.0,
            ipu_segments=[(1.0, 2.0), (2.1, 2.105)],
            min_ipu_s=0.01,
        )
    """
    if t_end < t_start:
        raise ValueError("t_end must be >= t_start")
    if min_ipu_s < 0:
        raise ValueError("min_ipu_s must be >= 0")

    kept: List[Tuple[float, float]] = []
    for s, e in ipu_segments:
        # Drop malformed ranges defensively.
        if e <= s:
            continue
        # Enforce minimum duration before rendering final tier.
        if (e - s) >= min_ipu_s:
            # Clamp to timeline boundaries to avoid out-of-range intervals.
            kept.append((max(t_start, s), min(t_end, e)))

    # Keep only strictly positive durations after clamping.
    kept = [(s, e) for s, e in kept if e > s]

    # Render full tier
    out: List[Interval] = []
    cursor = float(t_start)

    for s, e in kept:
        if s > cursor:
            # Fill leading gap with silence to preserve full coverage.
            out.append(Interval(start=cursor, end=s, text=silence_label))
        out.append(Interval(start=s, end=e, text=ipu_label))
        cursor = e

    if cursor < t_end:
        # Fill trailing silence after the last IPU.
        out.append(Interval(start=cursor, end=t_end, text=silence_label))

    # Remove zero/negative duration artifacts
    out = [itv for itv in out if itv.end > itv.start]
    return out
