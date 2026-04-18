"""CLI command for VoxAtlas-backed acoustic envelope extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hyper.features.acoustic.envelope import EnvelopeExtractionConfig
from hyper.features.pipelines.acoustics import run_envelope_pipeline


def add_subparser(subparsers: Any) -> None:
    """Register the `acoustic-envelope` subcommand."""
    parser = subparsers.add_parser(
        "acoustic-envelope",
        help="Extract a VoxAtlas Oganian-style envelope aligned to EEG samples.",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML.")
    parser.add_argument("--audio", type=Path, required=True, help="Input speech WAV file.")
    parser.add_argument("--eeg-sfreq", type=float, required=True, help="Target EEG sampling rate in Hertz.")
    parser.add_argument("--eeg-samples", type=int, required=True, help="Target EEG sample count.")
    parser.add_argument("--out", type=Path, required=True, help="Output NumPy array path.")
    parser.add_argument("--out-sidecar", type=Path, required=True, help="Output JSON sidecar path.")
    parser.add_argument("--frame-length", type=float, default=0.025, help="Envelope frame length in seconds.")
    parser.add_argument("--frame-step", type=float, default=0.010, help="Envelope frame step in seconds.")
    parser.add_argument("--smoothing", type=int, default=7, help="Envelope smoothing window in frames.")
    parser.add_argument("--peak-threshold", type=float, default=0.1, help="VoxAtlas Oganian peak threshold.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `acoustic-envelope` command."""
    del cfg
    run_envelope_pipeline(
        audio_path=args.audio,
        eeg_sampling_rate_hz=float(args.eeg_sfreq),
        eeg_sample_count=int(args.eeg_samples),
        output_values_path=args.out,
        output_sidecar_path=args.out_sidecar,
        config=EnvelopeExtractionConfig(
            frame_length_seconds=float(args.frame_length),
            frame_step_seconds=float(args.frame_step),
            smoothing_frames=int(args.smoothing),
            peak_threshold=float(args.peak_threshold),
        ),
    )
