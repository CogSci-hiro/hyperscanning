"""Tests for the TRF main-figure builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hyper.config import ProjectConfig
from hyper.viz import trf_main_figure as mod


def test_build_trf_main_figure_renders_configured_feature_grid(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The main figure should render one joint panel per configured feature."""
    cfg = ProjectConfig(
        raw={
            "paths": {
                "raw_root": str(tmp_path / "raw"),
                "out_dir": str(tmp_path / "derived"),
                "reports_root": str(tmp_path / "reports"),
                "results_root": str(tmp_path / "results"),
            },
            "viz": {
                "trf_main_figure": {
                    "task": "conversation",
                    "dpi": 120,
                    "panel_dpi": 100,
                    "figsize": {"width": 8.0, "height": 4.0},
                    "layout": {"rows": 1, "cols": 2},
                    "joint_times_seconds": [0.1, 0.2, 0.3],
                    "joint_plot": {
                        "line_width": 4.0,
                        "font_scale": 3.0,
                        "ylabel": "A.U.",
                        "show_colorbar": False,
                        "compact_vertical": True,
                    },
                    "features": [
                        {"predictor": "feat_a", "label": "Feature A"},
                        {"predictors": ["feat_b", "feat_c"], "label": "Feature BC"},
                    ],
                }
            },
        }
    )

    fake_group_data = mod.GroupAverageKernelData(
        predictor_names=("feat_a", "feat_b", "feat_c"),
        lag_seconds=np.array([0.0, 0.1, 0.2], dtype=float),
        channel_names=("Cz", "Pz"),
        group_mean_kernel_lag_feature_channel=np.array(
            [
                [[1.0, 0.5], [0.4, 0.2], [0.2, 0.1]],
                [[2.0, 1.0], [0.8, 0.3], [0.4, 0.2]],
                [[0.5, 0.2], [0.2, 0.1], [0.1, 0.05]],
            ],
            dtype=float,
        ),
        subject_ids=("sub-001", "sub-002"),
    )
    monkeypatch.setattr(mod, "_load_group_average_kernel_data", lambda cfg, task: fake_group_data)

    calls: list[dict[str, object]] = []

    def fake_plot_joint_map(
        beta_map: np.ndarray,
        *,
        times: np.ndarray,
        channel_names: list[str],
        output_stem: Path,
        title: str | None = None,
        formats=("png",),
        dpi: int = 300,
        line_width: float = 2.5,
        joint_times="peaks",
        significance_mask=None,
        font_scale: float = 1.0,
        ylabel: str | None = None,
        show_colorbar: bool = True,
        compact_vertical: bool = False,
    ) -> list[Path]:
        del significance_mask
        calls.append(
            {
                "beta_map": np.asarray(beta_map, dtype=float),
                "times": np.asarray(times, dtype=float),
                "channel_names": list(channel_names),
                "title": title,
                "formats": tuple(formats),
                "dpi": dpi,
                "line_width": line_width,
                "joint_times": np.asarray(joint_times, dtype=float),
                "font_scale": font_scale,
                "ylabel": ylabel,
                "show_colorbar": show_colorbar,
                "compact_vertical": compact_vertical,
            }
        )
        image_path = output_stem.with_suffix(".png")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mod.plt.imsave(image_path, np.ones((8, 8, 3), dtype=float))
        return [image_path]

    monkeypatch.setattr(mod, "plot_joint_map", fake_plot_joint_map)

    output_path = tmp_path / "reports" / "figures" / "trf_main.png"
    written = mod.build_trf_main_figure(cfg=cfg, output_path=output_path)

    assert written == output_path
    assert output_path.exists()
    assert [call["title"] for call in calls] == [None, None]
    assert all(call["formats"] == ("png",) for call in calls)
    assert all(call["channel_names"] == ["Cz", "Pz"] for call in calls)
    assert all(call["dpi"] == 100 for call in calls)
    assert all(call["line_width"] == 4.0 for call in calls)
    assert all(call["font_scale"] == 3.0 for call in calls)
    assert all(call["ylabel"] == "A.U." for call in calls)
    assert all(call["show_colorbar"] is False for call in calls)
    assert all(call["compact_vertical"] is True for call in calls)
    np.testing.assert_allclose(calls[0]["joint_times"], np.array([0.1, 0.2, 0.3], dtype=float))
