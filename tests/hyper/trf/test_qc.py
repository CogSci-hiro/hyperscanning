"""Tests for TRF QC figure aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hyper.config import ProjectConfig
from hyper.trf import qc as mod


def _make_project_config(tmp_path: Path) -> ProjectConfig:
    raw_root = tmp_path / "bids"
    for subject_id in ("sub-001", "sub-002"):
        (raw_root / subject_id).mkdir(parents=True, exist_ok=True)
    return ProjectConfig(
        raw={
            "paths": {
                "raw_root": str(raw_root),
                "out_dir": str(tmp_path / "derived"),
                "results_root": str(tmp_path / "results"),
                "reports_root": str(tmp_path / "reports"),
            },
            "subjects": {"pattern": "sub-*", "exclude": [], "missing_runs": {}},
            "debug": {"enabled": False, "subjects": []},
        }
    )


def _write_subject_trf_outputs(
    out_dir: Path,
    *,
    subject_id: str,
    predictors: list[str],
    lag_seconds: np.ndarray,
    fold_kernels: list[np.ndarray],
) -> None:
    subject_dir = out_dir / "trf" / subject_id / "task-conversation"
    subject_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        subject_dir / "coefficients.npz",
        lag_seconds=np.asarray(lag_seconds, dtype=np.float32),
        **{f"outer_fold_{index}": np.asarray(kernel, dtype=np.float32) for index, kernel in enumerate(fold_kernels, start=1)},
    )
    (subject_dir / "design_info.json").write_text(
        json.dumps({"predictors": predictors, "available_runs": ["1"]}),
        encoding="utf-8",
    )


def test_build_group_average_trf_kernel_manifest_plots_one_map_per_predictor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _make_project_config(tmp_path)
    lag_seconds = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    predictors = ["self_speech_envelope", "other_speech_envelope"]

    subject_one_fold_one = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=np.float32,
    )[..., np.newaxis]
    subject_one_fold_two = np.array(
        [
            [[2.0, 20.0], [4.0, 40.0]],
            [[6.0, 60.0], [8.0, 80.0]],
            [[10.0, 100.0], [12.0, 120.0]],
        ],
        dtype=np.float32,
    )[..., np.newaxis]
    subject_two_fold_one = np.array(
        [
            [[5.0, 50.0], [6.0, 60.0]],
            [[7.0, 70.0], [8.0, 80.0]],
            [[9.0, 90.0], [10.0, 100.0]],
        ],
        dtype=np.float32,
    )[..., np.newaxis]
    subject_two_fold_two = np.array(
        [
            [[7.0, 70.0], [8.0, 80.0]],
            [[9.0, 90.0], [10.0, 100.0]],
            [[11.0, 110.0], [12.0, 120.0]],
        ],
        dtype=np.float32,
    )[..., np.newaxis]

    _write_subject_trf_outputs(
        Path(cfg.raw["paths"]["out_dir"]),
        subject_id="sub-001",
        predictors=predictors,
        lag_seconds=lag_seconds,
        fold_kernels=[subject_one_fold_one, subject_one_fold_two],
    )
    _write_subject_trf_outputs(
        Path(cfg.raw["paths"]["out_dir"]),
        subject_id="sub-002",
        predictors=predictors,
        lag_seconds=lag_seconds,
        fold_kernels=[subject_two_fold_one, subject_two_fold_two],
    )

    monkeypatch.setattr(mod, "_load_channel_names_for_subject", lambda *args, **kwargs: ["Cz", "Pz"])
    plot_calls: list[dict[str, object]] = []

    def fake_plot_joint_map(
        kernel_map: np.ndarray,
        *,
        times: np.ndarray,
        channel_names: list[str],
        output_stem: Path,
        title: str | None = None,
        formats=("png", "pdf"),
        dpi: int = 300,
        line_width: float = 2.5,
        joint_times="peaks",
        significance_mask=None,
    ) -> list[Path]:
        plot_calls.append(
            {
                "kernel_map": np.asarray(kernel_map, dtype=float),
                "times": np.asarray(times, dtype=float),
                "channel_names": list(channel_names),
                "output_stem": output_stem,
                "title": title,
                "formats": tuple(formats),
                "dpi": dpi,
                "line_width": line_width,
                "joint_times": joint_times,
                "significance_mask": significance_mask,
            }
        )
        return [output_stem.with_suffix(".png")]

    monkeypatch.setattr(mod, "plot_joint_map", fake_plot_joint_map)

    manifest_path = Path(cfg.raw["paths"]["reports_root"]) / "figures" / "trf_kernels" / "manifest.json"
    manifest = mod.build_group_average_trf_kernel_manifest(
        cfg=cfg,
        task="conversation",
        manifest_path=manifest_path,
        formats=("png",),
    )

    assert manifest["status"] == "ok"
    assert manifest["subject_count"] == 2
    assert manifest["plot_count"] == 2
    assert [plot["predictor"] for plot in manifest["plots"]] == predictors
    assert manifest_path.exists()
    assert len(plot_calls) == 2

    expected_subject_one = np.mean(np.stack([subject_one_fold_one[..., 0], subject_one_fold_two[..., 0]], axis=0), axis=0)
    expected_subject_two = np.mean(np.stack([subject_two_fold_one[..., 0], subject_two_fold_two[..., 0]], axis=0), axis=0)
    expected_group_mean = np.mean(np.stack([expected_subject_one, expected_subject_two], axis=0), axis=0)

    np.testing.assert_allclose(plot_calls[0]["kernel_map"], expected_group_mean[:, 0, :].T)
    np.testing.assert_allclose(plot_calls[1]["kernel_map"], expected_group_mean[:, 1, :].T)
    np.testing.assert_allclose(plot_calls[0]["times"], lag_seconds)
    assert plot_calls[0]["channel_names"] == ["Cz", "Pz"]
    assert plot_calls[0]["formats"] == ("png",)


def test_build_subject_alpha_qc_manifest_writes_one_subject_plot_with_feature_lines(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg = _make_project_config(tmp_path)
    paths_out = Path(cfg.raw["paths"]["out_dir"])
    subject_dir = paths_out / "trf" / "sub-001" / "task-conversation"
    subject_dir.mkdir(parents=True, exist_ok=True)
    (subject_dir / "design_info.json").write_text(
        json.dumps({"predictors": ["self_speech_envelope", "other_speech_envelope"], "available_runs": ["1"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        mod,
        "_compute_subject_alpha_curve",
        lambda cfg, paths, subject_id, task, predictor_name: mod.SubjectAlphaCurve(
            subject_id=subject_id,
            predictor_name=predictor_name,
            alpha_values=np.array([0.1, 1.0, 10.0], dtype=float),
            mean_scores=np.array([0.2, 0.3, 0.25], dtype=float),
            se_scores=np.array([0.01, 0.02, 0.01], dtype=float),
            fold_scores=np.array([[0.2, 0.3, 0.25], [0.21, 0.32, 0.24]], dtype=float),
        ),
    )

    plot_calls: list[dict[str, object]] = []

    def fake_plot_subject_alpha_curves(curves, *, output_stem, task, formats=("png", "pdf"), dpi=300):
        plot_calls.append(
            {
                "curves": [curve.predictor_name for curve in curves],
                "output_stem": output_stem,
                "task": task,
                "formats": tuple(formats),
                "dpi": dpi,
            }
        )
        return [output_stem.with_suffix(".png")]

    monkeypatch.setattr(mod, "_plot_subject_alpha_curves", fake_plot_subject_alpha_curves)

    manifest_path = paths_out / "figures" / "trf_alpha_scores" / "manifest.json"
    manifest = mod.build_subject_alpha_qc_manifest(
        cfg=cfg,
        task="conversation",
        manifest_path=manifest_path,
        formats=("png",),
    )

    assert manifest["status"] == "ok"
    assert manifest["plot_count"] == 1
    assert manifest["plots"][0]["subject_id"] == "sub-001"
    assert manifest["plots"][0]["predictors"] == ["self_speech_envelope", "other_speech_envelope"]
    assert manifest_path.exists()
    assert plot_calls == [
        {
            "curves": ["self_speech_envelope", "other_speech_envelope"],
            "output_stem": paths_out / "figures" / "trf_alpha_scores" / "subjects" / "sub-001",
            "task": "conversation",
            "formats": ("png",),
            "dpi": 300,
        }
    ]
