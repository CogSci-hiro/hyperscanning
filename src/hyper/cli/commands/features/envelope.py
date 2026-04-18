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
    parser.add_argument("--eeg-samples", type=int, required=True, help="Target EEG sample count.")
    parser.add_argument("--out", type=Path, required=True, help="Output NumPy array path.")
    parser.add_argument("--out-sidecar", type=Path, required=True, help="Output JSON sidecar path.")
    parser.add_argument("--frame-length", type=float, default=None, help="Envelope frame length in seconds.")
    parser.add_argument("--frame-step", type=float, default=None, help="Envelope frame step in seconds.")
    parser.add_argument("--smoothing", type=int, default=None, help="Envelope smoothing window in frames.")
    parser.add_argument("--peak-threshold", type=float, default=None, help="VoxAtlas Oganian peak threshold.")


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `acoustic-envelope` command."""
    features_cfg = getattr(cfg, "raw", {}).get("features", {})
    feature_cfg = features_cfg.get("envelope", {})
    continuous_cfg = features_cfg.get("continuous", {})
    defaults = EnvelopeExtractionConfig()
    eeg_sfreq_hz = continuous_cfg.get("sfreq_hz")
    if eeg_sfreq_hz is None:
        raise ValueError("Continuous feature EEG sampling rate must be set in features.continuous.sfreq_hz.")
    run_envelope_pipeline(
        audio_path=args.audio,
        eeg_sampling_rate_hz=float(eeg_sfreq_hz),
        eeg_sample_count=int(args.eeg_samples),
        output_values_path=args.out,
        output_sidecar_path=args.out_sidecar,
        config=EnvelopeExtractionConfig(
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
            smoothing_frames=int(
                args.smoothing if args.smoothing is not None else feature_cfg.get(
                    "smoothing_frames",
                    defaults.smoothing_frames,
                )
            ),
            peak_threshold=float(
                args.peak_threshold if args.peak_threshold is not None else feature_cfg.get(
                    "peak_threshold",
                    defaults.peak_threshold,
                )
            ),
        ),
    )
