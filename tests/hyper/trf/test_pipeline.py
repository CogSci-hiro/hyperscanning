"""Unit tests for the TRF run/segment pipeline helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hyper.config import ProjectConfig
from hyper.paths import ProjectPaths
from hyper.trf.config import TrfConfig
from hyper.trf import pipeline as mod
from hyper.trf.pipeline import (
    TrfRunInput,
    TrfSegment,
    build_circular_shifted_eeg_qc_null_run_inputs,
    build_reduced_predictor_list,
    compute_score_delta,
    _predictor_path,
    build_lagged_segment_design,
    load_trf_run_inputs,
    run_trf_qc_score_tables,
    prepare_group_kfold,
    split_run_into_segments,
)


def _make_project_config(tmp_path: Path, *, trf_overrides: dict | None = None) -> ProjectConfig:
    """Build a minimal project config for TRF tests."""
    trf_cfg = {
        "enabled": True,
        "predictors": ["speech_envelope"],
        "target_sfreq": 10.0,
        "lags": {"tmin_seconds": -0.1, "tmax_seconds": 0.2},
        "conversation": {"duration_seconds": 4.0, "start_source": "metadata"},
        "segmentation": {
            "method": "blockwise_within_run",
            "n_blocks_per_run": 4,
            "drop_remainder": False,
            "min_block_duration_seconds": 0.2,
        },
        "cv": {
            "outer": {"splitter": "group_kfold", "n_splits": 5, "group_by": "run_id"},
            "inner": {"splitter": "group_kfold", "n_splits": 4, "group_by": "segment_id"},
        },
        "hyperparameters": {"alpha": {"scale": "logspace", "start_exp": -1, "stop_exp": 1, "num": 3}},
        "model": {"estimator": "ridge", "fit_intercept": False, "standardize_x": True, "standardize_y": False},
        "scoring": {"primary": "pearsonr"},
        "outputs": {
            "save_fold_scores": True,
            "save_selected_alpha_per_fold": True,
            "save_coefficients": True,
            "save_design_info": True,
        },
    }
    if trf_overrides:
        trf_cfg.update(trf_overrides)
    return ProjectConfig(
        raw={
            "paths": {
                "raw_root": str(tmp_path / "raw"),
                "out_dir": str(tmp_path / "derived"),
                "derived_root": str(tmp_path / "derived"),
                "lm_feature_root": str(tmp_path / "lm-derived"),
                "results_root": str(tmp_path / "results"),
                "reports_root": str(tmp_path / "reports"),
            },
            "subjects": {"missing_runs": {}},
            "runs": {"include": {"conversation": [1, 2]}},
            "trf": trf_cfg,
        }
    )


def _make_run_input(sample_count: int, *, run_id: str = "1") -> TrfRunInput:
    """Create a tiny cropped run input for segmentation tests."""
    predictor_values = np.arange(sample_count, dtype=np.float32)[:, np.newaxis]
    target_values = np.arange(sample_count * 2, dtype=np.float32).reshape(sample_count, 2)
    return TrfRunInput(
        subject_id="sub-001",
        task="conversation",
        run_id=run_id,
        predictor_names=("speech_envelope",),
        predictor_values=predictor_values,
        target_values=target_values,
        sampling_rate_hz=10.0,
        conversation_start_seconds=0.0,
        cropped_start_sample=0,
        cropped_stop_sample=sample_count,
        source_duration_seconds=float(sample_count / 10.0),
    )


def _write_test_run(paths: ProjectPaths, *, subject_id: str, run_id: str, sample_count: int) -> None:
    """Create minimal raw/predictor/timing artifacts for run-loading tests."""
    import mne

    stem = f"{subject_id}_task-conversation_run-{run_id}"
    raw_path = paths.out_dir / "eeg" / "filtered" / f"{stem}_raw_filt.fif"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["Cz", "Pz"], sfreq=10.0, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(np.zeros((2, sample_count), dtype=np.float32), info, verbose="ERROR")
    raw.save(raw_path, overwrite=True)

    for descriptor in ("self_envelope", "other_envelope"):
        predictor_path = paths.out_dir / "features" / "continuous" / "envelope" / f"{stem}_desc-{descriptor}_feature.npy"
        predictor_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(predictor_path, np.linspace(0.0, 1.0, sample_count, dtype=np.float32))

    timing_path = paths.out_dir / "eeg" / "downsampled" / f"{stem}_raw_ds_timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(json.dumps({"conversation_start_seconds": 0.0}), encoding="utf-8")


def _write_test_lm_event_feature(
    paths: ProjectPaths,
    *,
    subject_id: str,
    run_id: str,
    descriptor: str,
    dirname: str,
    value_column: str,
    rows: list[dict[str, float | str]],
) -> None:
    """Create a minimal LM event-feature TSV for one subject/run."""
    stem = f"{subject_id}_task-conversation_run-{run_id}"
    event_path = paths.lm_feature_root / "features" / "events" / dirname / f"{stem}_desc-{descriptor}_features.tsv"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    table = []
    for index, row in enumerate(rows, start=1):
        table.append(
            {
                "onset": row["onset"],
                "duration": row.get("duration", 0.1),
                "speaker": row.get("speaker", "A"),
                "word": row.get("word", f"tok{index}"),
                "normalized_word": row.get("normalized_word", row.get("word", f"tok{index}")),
                "word_index": index,
                value_column: row["value"],
                "run": str(run_id),
                "dyad_id": row.get("dyad_id", "dyad-001"),
                "lm_token_id": row.get("lm_token_id", f"{subject_id}_run-{run_id}_tok-{index}"),
                "source_interval_id": row.get("source_interval_id", f"{subject_id}_run-{run_id}_ann-{index}"),
                "alignment_status": "ok",
            }
        )
    pd.DataFrame(table).to_csv(event_path, sep="\t", index=False)


def _write_test_out_event_feature(
    paths: ProjectPaths,
    *,
    subject_id: str,
    run_id: str,
    descriptor: str,
    dirname: str,
    rows: list[dict[str, float | str]],
) -> None:
    """Create a minimal out-dir event-feature TSV for one subject/run."""
    stem = f"{subject_id}_task-conversation_run-{run_id}"
    event_path = paths.out_dir / "features" / "events" / dirname / f"{stem}_desc-{descriptor}_features.tsv"
    event_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(event_path, sep="\t", index=False)


def _write_test_run_with_timing(
    paths: ProjectPaths,
    *,
    subject_id: str,
    run_id: str,
    predictor_sample_count: int,
    eeg_sample_count: int,
    conversation_start_seconds: float,
) -> None:
    """Create a run where predictors are conversation-aligned but EEG spans the full run."""
    import mne

    stem = f"{subject_id}_task-conversation_run-{run_id}"
    raw_path = paths.out_dir / "eeg" / "filtered" / f"{stem}_raw_filt.fif"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    info = mne.create_info(["Cz", "Pz"], sfreq=10.0, ch_types=["eeg", "eeg"])
    raw_values = np.arange(eeg_sample_count * 2, dtype=np.float32).reshape(2, eeg_sample_count)
    raw = mne.io.RawArray(raw_values, info, verbose="ERROR")
    raw.save(raw_path, overwrite=True)

    predictor_values = np.arange(predictor_sample_count, dtype=np.float32)
    for descriptor in ("self_envelope", "other_envelope"):
        predictor_path = paths.out_dir / "features" / "continuous" / "envelope" / f"{stem}_desc-{descriptor}_feature.npy"
        predictor_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(predictor_path, predictor_values)

    timing_path = paths.out_dir / "eeg" / "downsampled" / f"{stem}_raw_ds_timing.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timing_path.write_text(
        json.dumps({"conversation_start_seconds": conversation_start_seconds}),
        encoding="utf-8",
    )


def test_build_circular_shifted_eeg_qc_null_run_inputs_preserves_length_and_is_deterministic() -> None:
    """EEG QC nulls should circularly shift run-local envelope predictors without changing shape."""
    run_input = TrfRunInput(
        subject_id="sub-001",
        task="conversation",
        run_id="1",
        predictor_names=("self_speech_envelope",),
        predictor_values=np.arange(12, dtype=np.float32)[:, np.newaxis],
        target_values=np.zeros((12, 2), dtype=np.float32),
        sampling_rate_hz=10.0,
        conversation_start_seconds=0.0,
        cropped_start_sample=0,
        cropped_stop_sample=12,
        source_duration_seconds=1.2,
    )

    shifted_one, shifts_one = build_circular_shifted_eeg_qc_null_run_inputs([run_input], seed=7)
    shifted_two, shifts_two = build_circular_shifted_eeg_qc_null_run_inputs([run_input], seed=7)

    assert shifted_one[0].predictor_values.shape == run_input.predictor_values.shape
    assert shifts_one == shifts_two
    assert shifts_one["1"] != 0
    np.testing.assert_array_equal(shifted_one[0].predictor_values, shifted_two[0].predictor_values)
    assert not np.array_equal(shifted_one[0].predictor_values, run_input.predictor_values)


def test_build_reduced_predictor_list_removes_one_target_group() -> None:
    """Feature QC ablations should drop exactly the requested configured predictor."""
    reduced = build_reduced_predictor_list(
        ("self_speech_envelope", "self_surprisal", "other_surprisal"),
        ablation_target="self_surprisal",
    )

    assert reduced == ("self_speech_envelope", "other_surprisal")


def test_build_reduced_predictor_list_supports_grouped_word_class_ablation() -> None:
    """Grouped ablations should remove both function and content word onsets together."""
    reduced = build_reduced_predictor_list(
        (
            "other_speech_envelope",
            "other_function_word_onsets",
            "other_content_word_onsets",
            "other_surprisal",
        ),
        ablation_target="other_word_class_onsets",
    )

    assert reduced == ("other_speech_envelope", "other_surprisal")


def test_build_reduced_predictor_list_supports_grouped_word_class_alias_predictor() -> None:
    """Grouped ablations should also remove a grouped alias when it is the configured predictor."""
    reduced = build_reduced_predictor_list(
        ("other_speech_envelope", "other_word_class_onsets", "other_surprisal"),
        ablation_target="other_word_class_onsets",
    )

    assert reduced == ("other_speech_envelope", "other_surprisal")


def test_compute_score_delta_subtracts_reference_score() -> None:
    """Delta scores should preserve the intended model-a minus model-b direction."""
    assert compute_score_delta(0.42, 0.17) == pytest.approx(0.25)


def test_split_run_into_segments_uses_requested_block_count() -> None:
    """A run should be split into the configured number of contiguous blocks."""
    cfg = TrfConfig.from_mapping(
        {
            "enabled": True,
            "predictors": ["speech_envelope"],
            "target_sfreq": 10.0,
            "lags": {"tmin_seconds": -0.1, "tmax_seconds": 0.2},
            "conversation": {"duration_seconds": 4.0, "start_source": "metadata"},
            "segmentation": {
                "method": "blockwise_within_run",
                "n_blocks_per_run": 4,
                "drop_remainder": False,
                "min_block_duration_seconds": 0.2,
            },
            "cv": {
                "outer": {"splitter": "group_kfold", "n_splits": 5, "group_by": "run_id"},
                "inner": {"splitter": "group_kfold", "n_splits": 4, "group_by": "segment_id"},
            },
            "hyperparameters": {"alpha": {"scale": "logspace", "start_exp": -1, "stop_exp": 1, "num": 3}},
            "model": {"estimator": "ridge", "fit_intercept": False, "standardize_x": True, "standardize_y": False},
            "scoring": {"primary": "pearsonr"},
            "outputs": {},
        }
    )
    run_input = _make_run_input(40)

    segments, skipped = split_run_into_segments(run_input, cfg)

    assert skipped == []
    assert len(segments) == 4
    assert [segment.start_sample for segment in segments] == [0, 10, 20, 30]
    assert [segment.stop_sample for segment in segments] == [10, 20, 30, 40]


def test_split_run_into_segments_distributes_remainder_without_gaps() -> None:
    """Remainder samples should stay inside contiguous blocks when drop_remainder is false."""
    cfg = TrfConfig.from_mapping(
        {
            "enabled": True,
            "predictors": ["speech_envelope"],
            "target_sfreq": 10.0,
            "lags": {"tmin_seconds": -0.1, "tmax_seconds": 0.2},
            "conversation": {"duration_seconds": 4.0, "start_source": "metadata"},
            "segmentation": {
                "method": "blockwise_within_run",
                "n_blocks_per_run": 4,
                "drop_remainder": False,
                "min_block_duration_seconds": 0.2,
            },
            "cv": {
                "outer": {"splitter": "group_kfold", "n_splits": 5, "group_by": "run_id"},
                "inner": {"splitter": "group_kfold", "n_splits": 4, "group_by": "segment_id"},
            },
            "hyperparameters": {"alpha": {"scale": "logspace", "start_exp": -1, "stop_exp": 1, "num": 3}},
            "model": {"estimator": "ridge", "fit_intercept": False, "standardize_x": True, "standardize_y": False},
            "scoring": {"primary": "pearsonr"},
            "outputs": {},
        }
    )
    run_input = _make_run_input(14)

    segments, skipped = split_run_into_segments(run_input, cfg)

    assert skipped == []
    assert [segment.stop_sample - segment.start_sample for segment in segments] == [4, 4, 3, 3]
    assert segments[-1].stop_sample == 14


def test_load_trf_run_inputs_skips_missing_runs_cleanly(tmp_path: Path) -> None:
    """Missing runs should be skipped rather than crashing the subject loader."""
    cfg = _make_project_config(tmp_path)
    paths = ProjectPaths.from_config(cfg)
    _write_test_run(paths, subject_id="sub-001", run_id="1", sample_count=40)

    run_inputs, skipped_runs = load_trf_run_inputs(
        cfg=cfg,
        paths=paths,
        subject_id="sub-001",
        task="conversation",
        run_ids=["1", "2"],
    )

    assert [run_input.run_id for run_input in run_inputs] == ["1"]
    assert skipped_runs == ["2"]


def test_load_trf_run_inputs_supports_self_and_other_envelope_predictors(tmp_path: Path) -> None:
    """TRF loading should accept both self and partner envelope predictors."""
    cfg = _make_project_config(
        tmp_path,
        trf_overrides={"predictors": ["self_speech_envelope", "other_speech_envelope"]},
    )
    paths = ProjectPaths.from_config(cfg)
    _write_test_run(paths, subject_id="sub-001", run_id="1", sample_count=40)

    run_inputs, skipped_runs = load_trf_run_inputs(
        cfg=cfg,
        paths=paths,
        subject_id="sub-001",
        task="conversation",
        run_ids=["1"],
    )

    assert skipped_runs == []
    assert len(run_inputs) == 1
    assert run_inputs[0].predictor_names == ("self_speech_envelope", "other_speech_envelope")
    assert run_inputs[0].predictor_values.shape[1] == 2


def test_load_trf_run_inputs_crops_only_eeg_target_when_predictors_start_at_conversation_onset(tmp_path: Path) -> None:
    """Predictors should keep their conversation-aligned start while EEG is cropped from run onset."""
    cfg = _make_project_config(tmp_path)
    paths = ProjectPaths.from_config(cfg)
    _write_test_run_with_timing(
        paths,
        subject_id="sub-001",
        run_id="1",
        predictor_sample_count=40,
        eeg_sample_count=50,
        conversation_start_seconds=1.0,
    )

    run_inputs, skipped_runs = load_trf_run_inputs(
        cfg=cfg,
        paths=paths,
        subject_id="sub-001",
        task="conversation",
        run_ids=["1"],
    )

    assert skipped_runs == []
    assert len(run_inputs) == 1
    assert run_inputs[0].predictor_values.shape == (40, 1)
    assert run_inputs[0].target_values.shape == (40, 2)
    assert np.allclose(run_inputs[0].predictor_values[:, 0], np.arange(40, dtype=np.float32))
    assert np.allclose(run_inputs[0].target_values[:, 0], np.arange(10, 50, dtype=np.float32))


def test_load_trf_run_inputs_supports_self_and_other_lm_predictors(tmp_path: Path) -> None:
    """TRF loading should rasterize self/other LM event predictors from the LM feature root."""
    cfg = _make_project_config(
        tmp_path,
        trf_overrides={"predictors": ["self_surprisal", "other_entropy"], "target_sfreq": 10.0},
    )
    paths = ProjectPaths.from_config(cfg)
    _write_test_run(paths, subject_id="sub-001", run_id="1", sample_count=40)
    _write_test_lm_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="lmSurprisal",
        dirname="lm_surprisal",
        value_column="surprisal",
        rows=[
            {"onset": 0.0, "value": 1.5, "speaker": "A", "word": "self0"},
            {"onset": 1.0, "value": 2.5, "speaker": "A", "word": "self1"},
        ],
    )
    _write_test_lm_event_feature(
        paths,
        subject_id="sub-002",
        run_id="1",
        descriptor="lmShannonEntropy",
        dirname="lm_shannon_entropy",
        value_column="entropy",
        rows=[
            {"onset": 0.2, "value": 0.7, "speaker": "B", "word": "other0"},
            {"onset": 1.2, "value": 0.9, "speaker": "B", "word": "other1"},
        ],
    )

    run_inputs, skipped_runs = load_trf_run_inputs(
        cfg=cfg,
        paths=paths,
        subject_id="sub-001",
        task="conversation",
        run_ids=["1"],
    )

    assert skipped_runs == []
    assert len(run_inputs) == 1
    assert run_inputs[0].predictor_names == ("self_surprisal", "other_entropy")
    assert run_inputs[0].predictor_values.shape == (40, 2)
    assert np.isclose(run_inputs[0].predictor_values[0, 0], 1.5)
    assert np.isclose(run_inputs[0].predictor_values[10, 0], 2.5)
    assert np.isclose(run_inputs[0].predictor_values[2, 1], 0.7)
    assert np.isclose(run_inputs[0].predictor_values[12, 1], 0.9)


def test_load_trf_run_inputs_rejects_mismatched_lm_speaker_encoding(tmp_path: Path) -> None:
    """LM predictor files should match the canonical speaker implied by their subject stem."""
    cfg = _make_project_config(
        tmp_path,
        trf_overrides={"predictors": ["self_surprisal"], "target_sfreq": 10.0},
    )
    paths = ProjectPaths.from_config(cfg)
    _write_test_run(paths, subject_id="sub-001", run_id="1", sample_count=40)
    _write_test_lm_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="lmSurprisal",
        dirname="lm_surprisal",
        value_column="surprisal",
        rows=[
            {"onset": 0.0, "value": 1.5, "speaker": "B", "word": "wrong-speaker"},
        ],
    )

    with pytest.raises(ValueError, match="LM event predictor speaker mismatch"):
        load_trf_run_inputs(
            cfg=cfg,
            paths=paths,
            subject_id="sub-001",
            task="conversation",
            run_ids=["1"],
        )


def test_load_trf_run_inputs_keeps_lm_events_on_conversation_timebase(tmp_path: Path) -> None:
    """LM event predictors should remain on their native conversation-aligned timebase."""
    cfg = _make_project_config(
        tmp_path,
        trf_overrides={"predictors": ["self_entropy"], "target_sfreq": 10.0},
    )
    paths = ProjectPaths.from_config(cfg)
    _write_test_run_with_timing(
        paths,
        subject_id="sub-001",
        run_id="1",
        predictor_sample_count=40,
        eeg_sample_count=50,
        conversation_start_seconds=1.0,
    )
    _write_test_lm_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="lmShannonEntropy",
        dirname="lm_shannon_entropy",
        value_column="entropy",
        rows=[
            {"onset": 1.0, "value": 0.5, "speaker": "A", "word": "start"},
            {"onset": 2.0, "value": 1.0, "speaker": "A", "word": "later"},
        ],
    )

    run_inputs, skipped_runs = load_trf_run_inputs(
        cfg=cfg,
        paths=paths,
        subject_id="sub-001",
        task="conversation",
        run_ids=["1"],
    )

    assert skipped_runs == []
    assert len(run_inputs) == 1
    assert run_inputs[0].predictor_values.shape == (40, 1)
    assert np.isclose(run_inputs[0].predictor_values[10, 0], 0.5)
    assert np.isclose(run_inputs[0].predictor_values[20, 0], 1.0)


def test_load_trf_run_inputs_supports_all_feature_families_simultaneously(tmp_path: Path) -> None:
    """TRF loading should combine continuous, impulse, LM, and expanded formant predictors."""
    cfg = _make_project_config(
        tmp_path,
        trf_overrides={
            "predictors": [
                "self_speech_envelope",
                "other_speech_envelope",
                "self_f0",
                "other_f0",
                "self_f1_f2",
                "other_f1_f2",
                "self_phoneme_onsets",
                "other_phoneme_onsets",
                "self_syllable_onsets",
                "other_syllable_onsets",
                "self_token_onsets",
                "other_token_onsets",
                "self_surprisal",
                "other_surprisal",
                "self_entropy",
                "other_entropy",
            ],
            "target_sfreq": 10.0,
        },
    )
    paths = ProjectPaths.from_config(cfg)
    _write_test_run_with_timing(
        paths,
        subject_id="sub-001",
        run_id="1",
        predictor_sample_count=50,
        eeg_sample_count=50,
        conversation_start_seconds=1.0,
    )
    stem = "sub-001_task-conversation_run-1"
    for descriptor, values in (
        ("self_f0", np.linspace(1.0, 2.0, 50, dtype=np.float32)),
        ("other_f0", np.linspace(2.0, 3.0, 50, dtype=np.float32)),
    ):
        predictor_path = paths.out_dir / "features" / "continuous" / "f0" / f"{stem}_desc-{descriptor}_feature.npy"
        predictor_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(predictor_path, values)

    _write_test_out_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="self_vowels",
        dirname="vowels",
        rows=[
            {"onset_seconds": 0.3, "f1_median_hz": 500.0, "f2_median_hz": 1500.0},
            {"onset_seconds": 1.0, "f1_median_hz": 550.0, "f2_median_hz": 1550.0},
        ],
    )
    _write_test_out_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="other_vowels",
        dirname="vowels",
        rows=[
            {"onset_seconds": 0.4, "f1_median_hz": 600.0, "f2_median_hz": 1600.0},
        ],
    )
    for subject_id, descriptor_prefix in (("sub-001", "self"), ("sub-001", "other")):
        _write_test_out_event_feature(
            paths,
            subject_id=subject_id,
            run_id="1",
            descriptor=f"{descriptor_prefix}_phonemes",
            dirname="phonemes",
            rows=[{"onset_seconds": 0.2, "duration_seconds": 0.05, "label": "p"}],
        )
        _write_test_out_event_feature(
            paths,
            subject_id=subject_id,
            run_id="1",
            descriptor=f"{descriptor_prefix}_syllables",
            dirname="syllables",
            rows=[{"onset_seconds": 0.5, "duration_seconds": 0.10, "label": "ba"}],
        )
        _write_test_out_event_feature(
            paths,
            subject_id=subject_id,
            run_id="1",
            descriptor=f"{descriptor_prefix}_tokens",
            dirname="tokens",
            rows=[{"onset_seconds": 0.8, "duration_seconds": 0.10, "label": "word"}],
        )
    _write_test_lm_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="lmSurprisal",
        dirname="lm_surprisal",
        value_column="surprisal",
        rows=[{"onset": 1.6, "value": 1.2}],
    )
    _write_test_lm_event_feature(
        paths,
        subject_id="sub-002",
        run_id="1",
        descriptor="lmSurprisal",
        dirname="lm_surprisal",
        value_column="surprisal",
        rows=[{"onset": 1.7, "value": 2.2, "speaker": "B"}],
    )
    _write_test_lm_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="lmShannonEntropy",
        dirname="lm_shannon_entropy",
        value_column="entropy",
        rows=[{"onset": 1.9, "value": 0.4}],
    )
    _write_test_lm_event_feature(
        paths,
        subject_id="sub-002",
        run_id="1",
        descriptor="lmShannonEntropy",
        dirname="lm_shannon_entropy",
        value_column="entropy",
        rows=[{"onset": 2.1, "value": 0.8, "speaker": "B"}],
    )

    run_inputs, skipped_runs = load_trf_run_inputs(
        cfg=cfg,
        paths=paths,
        subject_id="sub-001",
        task="conversation",
        run_ids=["1"],
    )

    assert skipped_runs == []
    assert len(run_inputs) == 1
    assert run_inputs[0].target_values.shape == (40, 2)
    assert run_inputs[0].predictor_names == (
        "self_speech_envelope",
        "other_speech_envelope",
        "self_f0",
        "other_f0",
        "self_f1",
        "self_f2",
        "other_f1",
        "other_f2",
        "self_phoneme_onsets",
        "other_phoneme_onsets",
        "self_syllable_onsets",
        "other_syllable_onsets",
        "self_token_onsets",
        "other_token_onsets",
        "self_surprisal",
        "other_surprisal",
        "self_entropy",
        "other_entropy",
    )
    assert run_inputs[0].predictor_values.shape == (40, 18)
    assert np.isclose(run_inputs[0].predictor_values[0, 0], 0.0)
    assert np.isclose(run_inputs[0].predictor_values[-1, 0], 39.0)
    assert np.isclose(run_inputs[0].predictor_values[3, 4], 500.0)
    assert np.isclose(run_inputs[0].predictor_values[4, 6], 600.0)
    assert np.isclose(run_inputs[0].predictor_values[2, 8], 1.0)
    assert np.isclose(run_inputs[0].predictor_values[5, 10], 1.0)
    assert np.isclose(run_inputs[0].predictor_values[8, 12], 1.0)
    assert np.isclose(run_inputs[0].predictor_values[16, 14], 1.2)
    assert np.isclose(run_inputs[0].predictor_values[17, 15], 2.2)
    assert np.isclose(run_inputs[0].predictor_values[19, 16], 0.4)
    assert np.isclose(run_inputs[0].predictor_values[21, 17], 0.8)


def test_other_event_predictors_resolve_from_current_subject_outputs(tmp_path: Path) -> None:
    """Self/other event predictors should differ by descriptor, not by subject stem."""
    cfg = _make_project_config(
        tmp_path,
        trf_overrides={"predictors": ["self_phoneme_onsets", "other_phoneme_onsets"]},
    )
    paths = ProjectPaths.from_config(cfg)

    self_path = _predictor_path(
        paths,
        subject_id="sub-007",
        task="conversation",
        run_id="1",
        predictor_name="self_phoneme_onsets",
    )
    other_path = _predictor_path(
        paths,
        subject_id="sub-007",
        task="conversation",
        run_id="1",
        predictor_name="other_phoneme_onsets",
    )

    assert self_path.name == "sub-007_task-conversation_run-1_desc-self_phonemes_features.tsv"
    assert other_path.name == "sub-007_task-conversation_run-1_desc-other_phonemes_features.tsv"


def test_load_trf_run_inputs_supports_word_class_event_predictors(tmp_path: Path) -> None:
    """TRF loading should rasterize function/content word onset predictors from event tables."""
    cfg = _make_project_config(
        tmp_path,
        trf_overrides={
            "predictors": ["self_function_word_onsets", "other_content_word_onsets"],
            "target_sfreq": 10.0,
        },
    )
    paths = ProjectPaths.from_config(cfg)
    _write_test_run_with_timing(
        paths,
        subject_id="sub-001",
        run_id="1",
        predictor_sample_count=50,
        eeg_sample_count=50,
        conversation_start_seconds=1.0,
    )
    _write_test_out_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="self_function_words",
        dirname="function_words",
        rows=[
            {"onset_seconds": 0.5, "duration_seconds": 0.1, "label": "le"},
            {"onset_seconds": 1.4, "duration_seconds": 0.1, "label": "de"},
        ],
    )
    _write_test_out_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="other_content_words",
        dirname="content_words",
        rows=[
            {"onset_seconds": 0.7, "duration_seconds": 0.1, "label": "chat"},
            {"onset_seconds": 1.8, "duration_seconds": 0.1, "label": "mange"},
        ],
    )

    run_inputs, skipped_runs = load_trf_run_inputs(
        cfg=cfg,
        paths=paths,
        subject_id="sub-001",
        task="conversation",
        run_ids=["1"],
    )

    assert skipped_runs == []
    assert len(run_inputs) == 1
    assert run_inputs[0].predictor_names == ("self_function_word_onsets", "other_content_word_onsets")
    assert run_inputs[0].predictor_values.shape == (40, 2)
    assert np.isclose(run_inputs[0].predictor_values[5, 0], 1.0)
    assert np.isclose(run_inputs[0].predictor_values[14, 0], 1.0)
    assert np.isclose(run_inputs[0].predictor_values[7, 1], 1.0)
    assert np.isclose(run_inputs[0].predictor_values[18, 1], 1.0)


def test_load_trf_run_inputs_combines_grouped_word_class_predictor_into_one_feature(tmp_path: Path) -> None:
    """Grouped word-class predictors should sum their component event rasters into one column."""
    cfg = _make_project_config(
        tmp_path,
        trf_overrides={
            "predictors": ["other_word_class_onsets"],
            "target_sfreq": 10.0,
        },
    )
    paths = ProjectPaths.from_config(cfg)
    _write_test_run_with_timing(
        paths,
        subject_id="sub-001",
        run_id="1",
        predictor_sample_count=50,
        eeg_sample_count=50,
        conversation_start_seconds=1.0,
    )
    _write_test_out_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="other_function_words",
        dirname="function_words",
        rows=[
            {"onset_seconds": 0.7, "duration_seconds": 0.1, "label": "le"},
            {"onset_seconds": 1.3, "duration_seconds": 0.1, "label": "de"},
        ],
    )
    _write_test_out_event_feature(
        paths,
        subject_id="sub-001",
        run_id="1",
        descriptor="other_content_words",
        dirname="content_words",
        rows=[
            {"onset_seconds": 0.7, "duration_seconds": 0.1, "label": "chat"},
            {"onset_seconds": 1.8, "duration_seconds": 0.1, "label": "mange"},
        ],
    )

    run_inputs, skipped_runs = load_trf_run_inputs(
        cfg=cfg,
        paths=paths,
        subject_id="sub-001",
        task="conversation",
        run_ids=["1"],
    )

    assert skipped_runs == []
    assert len(run_inputs) == 1
    assert run_inputs[0].predictor_names == ("other_word_class_onsets",)
    assert run_inputs[0].predictor_values.shape == (40, 1)
    assert np.isclose(run_inputs[0].predictor_values[7, 0], 2.0)
    assert np.isclose(run_inputs[0].predictor_values[13, 0], 1.0)
    assert np.isclose(run_inputs[0].predictor_values[18, 0], 1.0)


def test_prepare_group_kfold_downgrades_when_too_few_groups() -> None:
    """Requested grouped CV splits should downgrade to the available group count."""
    lag_samples = np.array([1, 0], dtype=int)
    segment_designs = [
        build_lagged_segment_design(
            TrfSegment(
                subject_id="sub-001",
                task="conversation",
                run_id=str(index),
                segment_id=f"segment-{index}",
                predictor_names=("speech_envelope",),
                predictor_values=np.arange(8, dtype=np.float32)[:, np.newaxis],
                target_values=np.arange(16, dtype=np.float32).reshape(8, 2),
                sampling_rate_hz=10.0,
                start_sample=0,
                stop_sample=8,
                start_seconds=0.0,
                stop_seconds=0.8,
            ),
            lag_samples,
        )
        for index in range(1, 4)
    ]

    splitter, groups, actual_splits = prepare_group_kfold(
        segment_designs,
        requested_splits=5,
        group_by="run_id",
        context="outer CV",
    )

    assert actual_splits == 3
    assert np.unique(groups).size == 3
    assert splitter.n_splits == 3


def test_build_lagged_segment_design_prevents_cross_segment_leakage() -> None:
    """Lagged design rows should be built independently per segment."""
    lag_samples = np.array([1, 0], dtype=int)
    first_segment = TrfSegment(
        subject_id="sub-001",
        task="conversation",
        run_id="1",
        segment_id="segment-1",
        predictor_names=("speech_envelope",),
        predictor_values=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        target_values=np.zeros((3, 1), dtype=np.float32),
        sampling_rate_hz=1.0,
        start_sample=0,
        stop_sample=3,
        start_seconds=0.0,
        stop_seconds=3.0,
    )
    second_segment = TrfSegment(
        subject_id="sub-001",
        task="conversation",
        run_id="2",
        segment_id="segment-2",
        predictor_names=("speech_envelope",),
        predictor_values=np.array([[100.0], [200.0], [300.0]], dtype=np.float32),
        target_values=np.zeros((3, 1), dtype=np.float32),
        sampling_rate_hz=1.0,
        start_sample=0,
        stop_sample=3,
        start_seconds=0.0,
        stop_seconds=3.0,
    )

    first_design = build_lagged_segment_design(first_segment, lag_samples)
    second_design = build_lagged_segment_design(second_segment, lag_samples)

    assert np.allclose(first_design.design_matrix, np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32))
    assert np.allclose(second_design.design_matrix[0], np.array([100.0, 200.0], dtype=np.float32))
    assert not np.isclose(second_design.design_matrix[0, 0], 3.0)


def test_prepare_group_kfold_requires_at_least_two_groups() -> None:
    """Grouped CV should fail clearly when the data cannot support it."""
    lag_samples = np.array([0], dtype=int)
    segment_design = build_lagged_segment_design(
        TrfSegment(
            subject_id="sub-001",
            task="conversation",
            run_id="1",
            segment_id="segment-1",
            predictor_names=("speech_envelope",),
            predictor_values=np.arange(4, dtype=np.float32)[:, np.newaxis],
            target_values=np.arange(4, dtype=np.float32)[:, np.newaxis],
            sampling_rate_hz=1.0,
            start_sample=0,
            stop_sample=4,
            start_seconds=0.0,
            stop_seconds=4.0,
        ),
        lag_samples,
    )

    with pytest.raises(ValueError, match="at least 2 unique groups"):
        prepare_group_kfold([segment_design], requested_splits=5, group_by="run_id", context="outer CV")


def test_run_trf_qc_score_tables_writes_eeg_and_feature_delta_tables(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """QC score-table generation should emit machine-readable EEG and feature delta rows."""
    cfg = _make_project_config(
        tmp_path,
        trf_overrides={
            "predictors": ["self_speech_envelope", "other_speech_envelope"],
            "qc_predictors": ["self_speech_envelope"],
            "ablation_targets": ["other_speech_envelope"],
        },
    )
    raw_root = Path(cfg.raw["paths"]["raw_root"])
    (raw_root / "sub-001" / "eeg").mkdir(parents=True, exist_ok=True)
    paths = ProjectPaths.from_config(cfg)
    _write_test_run(paths, subject_id="sub-001", run_id="1", sample_count=40)

    def fake_fit_subject_trf_score(*, run_inputs, config, subject_id, task, progress_label=None):  # noqa: ANN001
        del config, progress_label
        score = float(len(run_inputs[0].predictor_names)) + float(np.mean(run_inputs[0].predictor_values[:2, 0])) / 100.0
        return mod.TrfQcScoreSummary(
            subject_id=subject_id,
            score=score,
            score_name="pearsonr",
            predictor_names=run_inputs[0].predictor_names,
        )

    monkeypatch.setattr(mod, "fit_subject_trf_score", fake_fit_subject_trf_score)

    eeg_output_path = tmp_path / "derived" / "trf_qc" / "task-conversation" / "eeg_scores.tsv"
    feature_output_path = tmp_path / "derived" / "trf_qc" / "task-conversation" / "feature_scores.tsv"
    summary = run_trf_qc_score_tables(
        cfg=cfg,
        task="conversation",
        eeg_output_path=eeg_output_path,
        feature_output_path=feature_output_path,
    )

    eeg_table = pd.read_csv(eeg_output_path, sep="\t")
    feature_table = pd.read_csv(feature_output_path, sep="\t")

    assert summary["eeg_row_count"] == 1
    assert summary["feature_row_count"] == 1
    assert list(eeg_table.columns) == ["subject", "real_score", "null_score", "delta", "predictor_set", "score_name"]
    assert list(feature_table.columns) == ["subject", "target", "full_score", "reduced_score", "delta", "score_name"]
    assert eeg_table.loc[0, "subject"] == "sub-001"
    assert eeg_table.loc[0, "delta"] != 0.0
    assert feature_table.loc[0, "target"] == "other_speech_envelope"
    assert feature_table.loc[0, "delta"] > 0.0
