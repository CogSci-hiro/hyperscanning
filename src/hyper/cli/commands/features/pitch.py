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
    parser.add_argument(
        "--eeg-samples",
        type=int,
        default=None,
        help="Target EEG sample count. Optional when --raw is provided.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Optional raw FIF file from which to infer the EEG sample count.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output NumPy array path.")
    parser.add_argument("--out-sidecar", type=Path, required=True, help="Output JSON sidecar path.")
    parser.add_argument("--feature-name", type=str, default=None, help="Optional feature name stored in metadata.")
    parser.add_argument("--source-subject", type=str, default=None, help="Optional source subject stored in metadata.")
    parser.add_argument("--source-role", type=str, default=None, help="Optional source role stored in metadata.")
    parser.add_argument("--fmin", type=float, default=None, help="Minimum admissible F0 in Hertz.")
    parser.add_argument("--fmax", type=float, default=None, help="Maximum admissible F0 in Hertz.")
    parser.add_argument("--frame-length", type=float, default=None, help="Pitch frame length in seconds.")
    parser.add_argument("--frame-step", type=float, default=None, help="Pitch frame step in seconds.")
    parser.add_argument(
        "--fill-strategy",
        choices=["nan", "zero", "linear", "forward_fill"],
        default=None,
        help="How to prepare unvoiced regions for EEG-aligned TRF regressors.",
    )


def run(args: argparse.Namespace, cfg) -> None:
    """Execute the `acoustic-pitch` command."""
    features_cfg = getattr(cfg, "raw", {}).get("features", {})
    feature_cfg = features_cfg.get("pitch", {})
    continuous_cfg = features_cfg.get("continuous", {})
    defaults = PitchExtractionConfig()
    eeg_sfreq_hz = continuous_cfg.get("sfreq_hz", getattr(args, "eeg_sfreq", None))
    if eeg_sfreq_hz is None:
        raise ValueError("Continuous feature EEG sampling rate must be set in features.continuous.sfreq_hz.")
    eeg_sample_count = args.eeg_samples
    if eeg_sample_count is None:
        if args.raw is None:
            raise ValueError("Either --eeg-samples or --raw must be provided.")
        import mne

        eeg_sample_count = int(mne.io.read_raw_fif(args.raw, preload=False, verbose="ERROR").n_times)
    run_pitch_pipeline(
        audio_path=args.audio,
        eeg_sampling_rate_hz=float(eeg_sfreq_hz),
        eeg_sample_count=int(eeg_sample_count),
        output_values_path=args.out,
        output_sidecar_path=args.out_sidecar,
        feature_name=args.feature_name,
        source_subject=args.source_subject,
        source_role=args.source_role,
        config=PitchExtractionConfig(
            fmin_hz=float(args.fmin if args.fmin is not None else feature_cfg.get("fmin_hz", defaults.fmin_hz)),
            fmax_hz=float(args.fmax if args.fmax is not None else feature_cfg.get("fmax_hz", defaults.fmax_hz)),
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
            fill_strategy=str(
                args.fill_strategy if args.fill_strategy is not None else feature_cfg.get(
                    "fill_strategy",
                    defaults.fill_strategy,
                )
            ),
        ),
    )
