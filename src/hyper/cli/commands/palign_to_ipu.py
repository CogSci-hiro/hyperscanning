# ==================================================================================================
#                         CLI: palign-to-ipu
# ==================================================================================================
#
# Thin command:
# - parse args
# - read TokensAlign tier from palign TextGrid
# - derive IPU segments using core logic
# - write an output TextGrid with a single tier "IPU"
#
# ==================================================================================================

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

from hyper.annotations.palign_core import (
    DEFAULT_MIN_IPU_S,
    DEFAULT_MIN_SILENCE_S,
    Interval,
    apply_min_ipu_and_render_full_tier,
    build_ipu_segments_from_tokens,
)


@dataclass(frozen=True, slots=True)
class PalignToIpuCliConfig:
    """
    CLI configuration for `palign-to-ipu`.

    Usage example
    -------------
        cfg = PalignToIpuCliConfig(
            in_textgrid=Path("sub-001_run-01_palign.TextGrid"),
            out_textgrid=Path("sub-001_run-01_ipu.TextGrid"),
            tokens_tier="TokensAlign",
            include_laughter=False,
            include_noise=False,
            include_filled_pause=True,
            min_ipu_s=0.01,
            min_silence_s=0.2,
        )
    """

    in_textgrid: Path
    out_textgrid: Path
    tokens_tier: str

    include_laughter: bool
    include_noise: bool
    include_filled_pause: bool

    min_ipu_s: float
    min_silence_s: float


# ==================================================================================================
# Subparser registration
# ==================================================================================================

def add_subparser(subparsers: Any) -> None:
    """
    Register the palign-to-ipu subcommand.

    Usage example
    -------------
        yourtool palign-to-ipu in.TextGrid out.TextGrid --min-silence 0.2
    """
    p = subparsers.add_parser(
        "palign-to-ipu",
        help="Create an IPU tier from SPPAS palign TokensAlign tier.",
    )

    p.add_argument("in_textgrid", type=Path, help="Input palign TextGrid.")
    p.add_argument("out_textgrid", type=Path, help="Output TextGrid (single tier: IPU).")

    p.add_argument("--tokens-tier", type=str, default="TokensAlign", help="Tier name containing tokens.")

    p.add_argument("--include-laughter", action="store_true", help="Include '@' tokens in IPUs.")
    p.add_argument("--include-noise", action="store_true", help="Include '*' tokens in IPUs.")
    p.add_argument(
        "--include-filled-pause",
        type=str,
        default="true",
        help="Include 'fp' tokens in IPUs (true/false).",
    )

    p.add_argument(
        "--min-ipu",
        type=float,
        default=DEFAULT_MIN_IPU_S,
        help="IPUs shorter than this become silence (seconds).",
    )
    p.add_argument(
        "--min-silence",
        type=float,
        default=DEFAULT_MIN_SILENCE_S,
        help="Silences shorter than this merge adjacent IPUs (seconds).",
    )


# ==================================================================================================
# Run
# ==================================================================================================

def run(args: argparse.Namespace) -> int:
    """
    Run the palign-to-ipu command.

    Parameters
    ----------
    args
        Parsed CLI args.

    Returns
    -------
    int
        Exit code (0 on success).
    """
    # Normalize and validate CLI arguments once at the boundary.
    cfg = _parse_args(args)

    # Read token-level alignment intervals from the requested tier.
    t_start, t_end, tokens = _read_tokensalign_intervals(
        tg_path=cfg.in_textgrid,
        tier_name=cfg.tokens_tier,
    )

    # Convert token stream to coarse IPU spans using configurable rules
    # (e.g., whether laughter/noise/fillers count as speech).
    segments = build_ipu_segments_from_tokens(
        tokens,
        include_laughter=cfg.include_laughter,
        include_noise=cfg.include_noise,
        include_filled_pause=cfg.include_filled_pause,
        min_silence_s=cfg.min_silence_s,
    )

    # Apply duration threshold and render a full-coverage IPU/silence tier.
    ipu_tier = apply_min_ipu_and_render_full_tier(
        t_start=t_start,
        t_end=t_end,
        ipu_segments=segments,
        min_ipu_s=cfg.min_ipu_s,
    )

    # Persist resulting tier as a single-tier TextGrid.
    _write_ipu_textgrid(
        out_path=cfg.out_textgrid,
        t_start=t_start,
        t_end=t_end,
        ipu_intervals=ipu_tier,
    )

    return 0


# ==================================================================================================
# Helpers
# ==================================================================================================

def _parse_bool(text: str) -> bool:
    t = str(text).strip().lower()
    if t in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if t in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {text!r} (expected true/false)")


def _parse_args(args: argparse.Namespace) -> PalignToIpuCliConfig:
    # Resolve to absolute paths early so subsequent error messages are explicit.
    in_textgrid = Path(args.in_textgrid).expanduser().resolve()
    out_textgrid = Path(args.out_textgrid).expanduser().resolve()

    if not in_textgrid.exists():
        raise FileNotFoundError(f"Input TextGrid not found: {in_textgrid}")

    return PalignToIpuCliConfig(
        in_textgrid=in_textgrid,
        out_textgrid=out_textgrid,
        tokens_tier=str(args.tokens_tier).strip(),
        include_laughter=bool(args.include_laughter),
        include_noise=bool(args.include_noise),
        include_filled_pause=_parse_bool(args.include_filled_pause),
        min_ipu_s=float(args.min_ipu),
        min_silence_s=float(args.min_silence),
    )


def _read_tokensalign_intervals(*, tg_path: Path, tier_name: str) -> Tuple[float, float, List[Interval]]:
    """
    Read a TextGrid and return (t_start, t_end, tokens) from the given interval tier.

    Usage example
    -------------
        t0, t1, toks = _read_tokensalign_intervals(tg_path=Path("x.TextGrid"),
                                                   tier_name="TokensAlign")
    """
    try:
        from praatio import textgrid as praatio_textgrid  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'praatio'. Install with: pip install praatio") from e

    tg = praatio_textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=True)

    if tier_name not in tg.tierNames:
        raise ValueError(f"Tier '{tier_name}' not found. Available tiers: {tg.tierNames}")

    tier = tg.getTier(tier_name)
    entries = tier.entries  # (start, end, label)

    t_start = float(tg.minTimestamp)
    t_end = float(tg.maxTimestamp)

    # Cast everything to plain Python numeric/string types to keep downstream
    # core logic independent of `praatio` object internals.
    tokens: List[Interval] = [Interval(float(s), float(e), str(lbl)) for (s, e, lbl) in entries]
    tokens.sort(key=lambda x: (x.start, x.end))
    return t_start, t_end, tokens


def _write_ipu_textgrid(*, out_path: Path, t_start: float, t_end: float, ipu_intervals: List[Interval]) -> None:
    """
    Write a single-tier TextGrid named 'IPU'.

    Usage example
    -------------
        _write_ipu_textgrid(out_path=Path("out.TextGrid"),
                            t_start=0.0, t_end=12.3,
                            ipu_intervals=intervals)
    """
    try:
        from praatio import textgrid as praatio_textgrid  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency 'praatio'. Install with: pip install praatio") from e

    # Create a new standalone TextGrid containing only the derived IPU tier.
    tg_out = praatio_textgrid.Textgrid(minTimestamp=t_start, maxTimestamp=t_end)
    entries = [(itv.start, itv.end, itv.text) for itv in ipu_intervals]
    ipu_tier = praatio_textgrid.IntervalTier(name="IPU", entries=entries, minT=t_start, maxT=t_end)
    tg_out.addTier(ipu_tier)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tg_out.save(str(out_path), format="long_textgrid", includeBlankSpaces=True)
