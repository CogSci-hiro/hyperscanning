"""Tests for TRF score QC figure helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hyper.config import ProjectConfig
from hyper.viz import trf_score_qc as mod


def test_load_inputs_reads_expected_tables(tmp_path: Path) -> None:
    """Input loader should read both TSV tables without altering columns."""
    eeg_path = tmp_path / "eeg.tsv"
    feature_path = tmp_path / "feature.tsv"
    pd.DataFrame(
        [{"subject": "sub-001", "real_score": 0.2, "null_score": 0.1, "delta": 0.1, "predictor_set": "a", "score_name": "pearsonr"}]
    ).to_csv(eeg_path, sep="\t", index=False)
    pd.DataFrame(
        [{"subject": "sub-001", "target": "self_speech_envelope", "full_score": 0.2, "reduced_score": 0.1, "delta": 0.1, "score_name": "pearsonr"}]
    ).to_csv(feature_path, sep="\t", index=False)

    loaded = mod._load_inputs(eeg_table_path=eeg_path, feature_table_path=feature_path)

    assert list(loaded.eeg_table.columns) == ["subject", "real_score", "null_score", "delta", "predictor_set", "score_name"]
    assert list(loaded.feature_table.columns) == ["subject", "target", "full_score", "reduced_score", "delta", "score_name"]


def test_plotted_feature_labels_reads_optional_alias_mapping() -> None:
    """Feature label aliases should come from viz.trf_score.plotted_features."""
    cfg = ProjectConfig(
        raw={
            "viz": {
                "trf_score": {
                    "plotted_features": {
                        "other_speech_envelope": "envelope",
                    }
                }
            }
        }
    )

    assert mod._plotted_feature_labels(cfg) == {"other_speech_envelope": "envelope"}


def test_build_trf_score_qc_figure_writes_expected_path(monkeypatch, tmp_path: Path) -> None:
    """Figure builder should save the requested TRF-score figure output."""
    cfg = ProjectConfig(
        raw={"viz": {"trf_score": {"dpi": 123, "figsize": {"width": 8.0, "height": 3.0}, "x_tick_label_scale": 0.7}}}
    )
    eeg_path = tmp_path / "eeg.tsv"
    feature_path = tmp_path / "feature.tsv"
    pd.DataFrame(
        [{"subject": "sub-001", "real_score": 0.2, "null_score": 0.1, "delta": 0.1, "predictor_set": "a", "score_name": "pearsonr"}]
    ).to_csv(eeg_path, sep="\t", index=False)
    pd.DataFrame(
        [{"subject": "sub-001", "target": "self_speech_envelope", "full_score": 0.2, "reduced_score": 0.1, "delta": 0.1, "score_name": "pearsonr"}]
    ).to_csv(feature_path, sep="\t", index=False)

    saved = {}

    class DummyText:
        def __init__(self, fontsize: float) -> None:
            self._fontsize = fontsize

        def get_fontsize(self) -> float:
            return self._fontsize

        def set_fontsize(self, value: float) -> None:
            self._fontsize = value

    class DummyFigure:
        def __init__(self) -> None:
            self.texts = [DummyText(10.0), DummyText(14.0)]

        def findobj(self, cls):
            return self.texts

        def tight_layout(self) -> None:
            saved["tight_layout"] = True

        def savefig(self, path, dpi, bbox_inches) -> None:
            saved["savefig"] = (Path(path), dpi, bbox_inches)

    class DummyAxis:
        def __init__(self) -> None:
            self._ticks = [DummyText(12.0), DummyText(16.0)]

        def get_xticklabels(self):
            return self._ticks

    figure = DummyFigure()
    axes = [DummyAxis(), DummyAxis()]

    def _subplots(*args, **kwargs):
        saved["subplots_kwargs"] = kwargs
        return figure, axes

    monkeypatch.setattr(mod.plt, "subplots", _subplots)
    monkeypatch.setattr(mod, "_plot_eeg_quality", lambda axis, table: None)
    monkeypatch.setattr(mod, "_plot_feature_quality", lambda axis, table, plotted_feature_labels: None)
    monkeypatch.setattr(mod.plt, "close", lambda fig: saved.setdefault("closed", fig))

    output_path = tmp_path / "reports" / "trf_score_summary.png"
    written = mod.build_trf_score_qc_figure(
        cfg=cfg,
        eeg_table_path=eeg_path,
        feature_table_path=feature_path,
        output_path=output_path,
    )

    assert written == output_path
    assert saved["subplots_kwargs"]["gridspec_kw"] == {"width_ratios": [1, 3]}
    assert saved["savefig"] == (output_path, 123, "tight")
    assert saved["closed"] is figure
    assert [text.get_fontsize() for text in figure.texts] == [20.0, 28.0]
    assert np.allclose([tick.get_fontsize() for tick in axes[0].get_xticklabels()], [8.4, 11.2])
    assert np.allclose([tick.get_fontsize() for tick in axes[1].get_xticklabels()], [8.4, 11.2])


def test_pvalue_to_stars_uses_conventional_thresholds() -> None:
    """P-value helper should map to standard star strings."""
    assert mod._pvalue_to_stars(0.2) == ""
    assert mod._pvalue_to_stars(0.04) == "*"
    assert mod._pvalue_to_stars(0.009) == "**"
    assert mod._pvalue_to_stars(0.0009) == "***"


def test_annotate_one_sample_stars_marks_only_significant_groups() -> None:
    """Only groups with significant non-zero means should receive stars."""
    captured = []

    class DummyAxis:
        def get_ylim(self):
            return (0.0, 1.0)

        def text(self, x, y, label, ha, va, transform=None):
            captured.append((x, y, label, ha, va, transform))

    mod._annotate_one_sample_stars(
        DummyAxis(),
        values_by_group={
            "a": np.array([0.3, 0.31, 0.29, 0.32]),
            "b": np.array([0.01, -0.01, 0.0, 0.0]),
        },
        order=["a", "b"],
    )

    assert len(captured) == 1
    assert captured[0][0] == 0
    assert captured[0][2] in {"*", "**", "***"}


def test_annotate_one_sample_stars_supports_explicit_positions_and_axes_fraction() -> None:
    """Star annotation helper should support fixed x positions and axes-fraction heights."""
    captured = {}

    class DummyAxis:
        def get_ylim(self):
            return (0.0, 1.0)

        def get_xaxis_transform(self):
            return "xaxis-transform"

        def text(self, x, y, label, ha, va, transform=None):
            captured["text"] = (x, y, label, ha, va, transform)

    mod._annotate_one_sample_stars(
        DummyAxis(),
        values_by_group={"delta": np.array([0.4, 0.42, 0.41, 0.43])},
        order=["delta"],
        positions={"delta": 2.0},
        axes_fraction_y=0.9,
    )

    assert captured["text"] == (2.0, 0.9, "***", "center", "bottom", "xaxis-transform")


@pytest.mark.parametrize("plotter_name, expected_y", [("_plot_eeg_quality", "pearson_r"), ("_plot_feature_quality", "delta")])
def test_stripplot_uses_updated_jitter_styling(monkeypatch, plotter_name: str, expected_y: str) -> None:
    """Jittered points should use the updated alpha, size, and outline width."""
    table = pd.DataFrame(
        [{"subject": "sub-001", "real_score": 0.2, "null_score": 0.1, "delta": 0.1, "predictor_set": "a", "score_name": "pearsonr", "target": "self"}]
    )
    captured = {}

    monkeypatch.setattr(mod.sns, "violinplot", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.sns, "despine", lambda **kwargs: None)

    def _capture_stripplot(*args, **kwargs):
        captured["stripplot"] = kwargs

    monkeypatch.setattr(mod.sns, "stripplot", _capture_stripplot)

    class DummyAxis:
        def set_ylabel(self, value): pass
        def set_title(self, value): pass
        def set_xlabel(self, value): pass
        def grid(self, *args, **kwargs): pass
        def set_axisbelow(self, value): pass
        def set_xticks(self, *args, **kwargs): pass

    if plotter_name == "_plot_feature_quality":
        getattr(mod, plotter_name)(DummyAxis(), table, plotted_feature_labels={})
    else:
        getattr(mod, plotter_name)(DummyAxis(), table)

    assert captured["stripplot"]["y"] == expected_y
    assert captured["stripplot"]["alpha"] == 1.0
    assert captured["stripplot"]["size"] == 4.65
    assert captured["stripplot"]["linewidth"] == 0.675


def test_plot_feature_quality_uses_alias_labels_and_adds_groupwise_stars(monkeypatch) -> None:
    """Feature panel should use configured alias labels and annotate all groups."""
    table = pd.DataFrame(
        [
            {"target": "self", "delta": 0.2},
            {"target": "self", "delta": 0.22},
            {"target": "other", "delta": 0.15},
            {"target": "other", "delta": 0.18},
        ]
    )
    captured = {}

    monkeypatch.setattr(mod.sns, "violinplot", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.sns, "stripplot", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.sns, "despine", lambda **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_annotate_one_sample_stars",
        lambda axis, values_by_group, order, positions=None, axes_fraction_y=None: captured.setdefault(
            "stars", (values_by_group, order, positions, axes_fraction_y)
        ),
    )

    class DummyAxis:
        def set_xticks(self, positions, labels, rotation=None, ha=None):
            captured["xticks"] = (positions, labels, rotation, ha)

        def set_ylabel(self, value): pass
        def set_title(self, value): pass
        def set_xlabel(self, value): pass
        def grid(self, *args, **kwargs): pass
        def set_axisbelow(self, value): pass

    axis = DummyAxis()
    mod._plot_feature_quality(axis, table, plotted_feature_labels={"self": "env", "other": "pitch"})

    assert captured["xticks"][2] is None
    assert captured["xticks"][3] is None
    assert list(captured["xticks"][1]) == ["env", "pitch"]
    assert captured["stars"][1] == ["self", "other"]
    assert captured["stars"][2] == {"self": 0.0, "other": 1.0}
    assert captured["stars"][3] == 0.9


def test_plot_feature_quality_filters_to_mapped_features(monkeypatch) -> None:
    """A non-empty plotted_features mapping should select only those targets."""
    table = pd.DataFrame(
        [
            {"target": "self_speech_envelope", "delta": 0.2},
            {"target": "other_speech_envelope", "delta": 0.22},
            {"target": "other_speech_envelope", "delta": 0.21},
        ]
    )
    captured = {}

    monkeypatch.setattr(mod.sns, "violinplot", lambda *args, **kwargs: captured.setdefault("violinplot", kwargs))
    monkeypatch.setattr(mod.sns, "stripplot", lambda *args, **kwargs: captured.setdefault("stripplot", kwargs))
    monkeypatch.setattr(mod.sns, "despine", lambda **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_annotate_one_sample_stars",
        lambda axis, values_by_group, order, positions=None, axes_fraction_y=None: captured.setdefault(
            "stars", (values_by_group, order, positions, axes_fraction_y)
        ),
    )

    class DummyAxis:
        def set_xticks(self, positions, labels, rotation=None, ha=None):
            captured["xticks"] = (positions, labels)

        def set_ylabel(self, value): pass
        def set_title(self, value): pass
        def set_xlabel(self, value): pass
        def grid(self, *args, **kwargs): pass
        def set_axisbelow(self, value): pass

    mod._plot_feature_quality(
        DummyAxis(),
        table,
        plotted_feature_labels={"other_speech_envelope": "envelope"},
    )

    assert list(captured["violinplot"]["data"]["target"]) == ["other_speech_envelope", "other_speech_envelope"]
    assert list(captured["xticks"][1]) == ["envelope"]
    assert captured["stars"][1] == ["other_speech_envelope"]


def test_plot_eeg_quality_adds_stars_only_for_delta(monkeypatch) -> None:
    """EEG panel should request significance stars only for the delta group."""
    table = pd.DataFrame(
        [
            {"null_score": 0.0, "real_score": 0.1, "delta": 0.1},
            {"null_score": 0.0, "real_score": 0.12, "delta": 0.12},
        ]
    )
    captured = {}

    monkeypatch.setattr(mod.sns, "violinplot", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.sns, "stripplot", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod.sns, "despine", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_annotate_one_sample_stars", lambda axis, values_by_group, order, positions=None, axes_fraction_y=None: captured.setdefault("stars", (values_by_group, order, positions, axes_fraction_y)))

    class DummyAxis:
        def set_ylabel(self, value): pass
        def set_title(self, value): pass
        def set_xlabel(self, value): pass
        def grid(self, *args, **kwargs): pass
        def set_axisbelow(self, value): pass

    mod._plot_eeg_quality(DummyAxis(), table)

    assert list(captured["stars"][0]) == [r"$\Delta R$"]
    assert captured["stars"][1] == [r"$\Delta R$"]
    assert captured["stars"][2] == {r"$\Delta R$": 2.0}
    assert captured["stars"][3] == 0.9
