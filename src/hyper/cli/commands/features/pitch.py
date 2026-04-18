"""CLI command for VoxAtlas-backed F0 extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.features.acoustic.pitch import PitchExtractionConfig
from hyper.features.pipelines.acoustics import run_pitch_pipeline


def add_subparser(subparsers: Any) -> None:
    """Register the `acoustic-pitch` subcommand."""
    parser = subparsers.add_parser(
        "acoustic-pitch",
        help="Extract a VoxAtlas F0 contour aligned to EEG samples.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--audio", type=Path, required=True, help="Input speech WAV file.")
    parser.add_argument("--eeg-sfreq", type=float, required=True, help="Target EEG sampling rate in Hertz.")
    parser.add_argument("--eeg-samples", type=int, required=True, help="Target EEG sample count.")
    parser.add_argument("--out", type=Path, required=True, help="Output NumPy array path.")
    parser.add_argument("--out-sidecar", type=Path, required=True, help="Output JSON sidecar path.")
    parser.add_argument("--fmin", type=float, default=75.0, help="Minimum admissible F0 in Hertz.")
    parser.add_argument("--fmax", type=float, default=500.0, help="Maximum admissible F0 in Hertz.")
    parser.add_argument("--frame-length", type=float, default=0.040, help="Pitch frame length in seconds.")
    parser.add_argument("--frame-step", type=float, default=0.010, help="Pitch frame step in seconds.")
    parser.add_argument(
        "--fill-strategy",
        choices=["nan", "zero", "linear", "forward_fill"],
        default="linear",
        help="How to prepare unvoiced regions for EEG-aligned TRF regressors.",
    )


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `acoustic-pitch` command."""
    del cfg
    run_pitch_pipeline(
        audio_path=args.audio,
        eeg_sampling_rate_hz=float(args.eeg_sfreq),
        eeg_sample_count=int(args.eeg_samples),
        output_values_path=args.out,
        output_sidecar_path=args.out_sidecar,
        config=PitchExtractionConfig(
            fmin_hz=float(args.fmin),
            fmax_hz=float(args.fmax),
            frame_length_seconds=float(args.frame_length),
            frame_step_seconds=float(args.frame_step),
            fill_strategy=str(args.fill_strategy),
        ),
    )
