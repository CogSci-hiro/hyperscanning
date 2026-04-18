"""CLI command for vowel-centered F1/F2 event extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.features.acoustic.formants import FormantEventExtractionConfig
from hyper.features.pipelines.acoustics import run_vowel_formant_pipeline


def add_subparser(subparsers: Any) -> None:
    """Register the `acoustic-formants` subcommand."""
    parser = subparsers.add_parser(
        "acoustic-formants",
        help="Extract vowel-centered F1/F2 median event tables from alignment intervals.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--audio", type=Path, required=True, help="Input speech WAV file.")
    parser.add_argument(
        "--alignment",
        "--textgrid",
        dest="alignment",
        type=Path,
        required=True,
        help="Alignment file containing vowel source intervals (.TextGrid or palign .csv).",
    )
    parser.add_argument("--tier", type=str, required=True, help="Tier name containing phoneme/vowel intervals.")
    parser.add_argument("--out-tsv", type=Path, required=True, help="Output TSV event table path.")
    parser.add_argument("--out-sidecar", type=Path, required=True, help="Output JSON sidecar path.")
    parser.add_argument("--feature-name", type=str, default=None, help="Optional feature name stored in metadata.")
    parser.add_argument("--speaker", type=str, default=None, help="Optional speaker label to store in the output table.")
    parser.add_argument("--source-subject", type=str, default=None, help="Optional source subject stored in metadata.")
    parser.add_argument("--source-role", type=str, default=None, help="Optional source role stored in metadata.")
    parser.add_argument("--language", type=str, default=None, help="Optional VoxAtlas phonology language code.")
    parser.add_argument("--resource-root", type=str, default=None, help="Optional VoxAtlas phonology resource root.")
    parser.add_argument("--frame-length", type=float, default=None, help="Formant frame length in seconds.")
    parser.add_argument("--frame-step", type=float, default=None, help="Formant frame step in seconds.")
    parser.add_argument("--lpc-order", type=int, default=None, help="Fallback LPC order for formant tracking.")
    parser.add_argument("--max-formant", type=float, default=None, help="Maximum formant frequency in Hertz.")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        help="Minimum vowel interval duration in seconds required for extraction.",
    )


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `acoustic-formants` command."""
    feature_cfg = getattr(cfg, "raw", {}).get("features", {}).get("formants", {})
    defaults = FormantEventExtractionConfig()
    run_vowel_formant_pipeline(
        audio_path=args.audio,
        alignment_path=args.alignment,
        tier_name=str(args.tier),
        output_tsv_path=args.out_tsv,
        output_sidecar_path=args.out_sidecar,
        speaker=args.speaker,
        feature_name=args.feature_name,
        source_subject=args.source_subject,
        source_role=args.source_role,
        config=FormantEventExtractionConfig(
            language=args.language if args.language is not None else feature_cfg.get("language", defaults.language),
            resource_root=(
                args.resource_root
                if args.resource_root is not None
                else feature_cfg.get("resource_root", defaults.resource_root)
            ),
            frame_length_seconds=float(
                args.frame_length if args.frame_length is not None else feature_cfg.get(
                    "frame_length_seconds",
                    defaults.frame_length_seconds,
                )
            ),
            frame_step_seconds=float(
                args.frame_step if args.frame_step is not None else feature_cfg.get(
                    "frame_step_seconds",
                    defaults.frame_step_seconds,
                )
            ),
            lpc_order=int(
                args.lpc_order if args.lpc_order is not None else feature_cfg.get(
                    "lpc_order",
                    defaults.lpc_order,
                )
            ),
            max_formant_hz=float(
                args.max_formant if args.max_formant is not None else feature_cfg.get(
                    "max_formant_hz",
                    defaults.max_formant_hz,
                )
            ),
            min_interval_duration_seconds=float(
                args.min_duration if args.min_duration is not None else feature_cfg.get(
                    "min_interval_duration_seconds",
                    defaults.min_interval_duration_seconds,
                )
            ),
            use_parselmouth=bool(feature_cfg.get("use_parselmouth", defaults.use_parselmouth)),
        ),
    )
