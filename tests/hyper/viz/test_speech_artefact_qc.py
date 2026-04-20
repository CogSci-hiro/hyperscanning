"""Tests for speech artefact QC helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hyper.config import ProjectConfig
from hyper.viz import speech_artefact_qc as mod


def test_load_component_count_summary_uses_exclude_and_component_total(monkeypatch, tmp_path: Path) -> None:
    """Included counts should be derived from total-minus-excluded components."""

    class DummyIca:
        def __init__(self, total: int, excluded: list[int]) -> None:
            self.n_components_ = total
            self.exclude = excluded

    monkeypatch.setattr(
        mod.mne.preprocessing,
        "read_ica",
        lambda path: DummyIca(10, [0, 2]) if Path(path).name == "a.fif" else DummyIca(8, [1]),
    )

    summary = mod._load_component_count_summary([tmp_path / "a.fif", tmp_path / "b.fif"])

    assert summary.subject_ids == ("a.fif", "b.fif")
    np.testing.assert_array_equal(summary.total_counts, np.array([10.0, 8.0]))
    np.testing.assert_array_equal(summary.included_counts, np.array([8.0, 7.0]))
    np.testing.assert_array_equal(summary.excluded_counts, np.array([2.0, 1.0]))
    np.testing.assert_array_equal(summary.bad_channel_counts, np.array([0.0, 0.0]))


def test_load_component_count_summary_deduplicates_excluded_components(monkeypatch, tmp_path: Path) -> None:
    """Repeated component ids in `ica.exclude` should only count once."""

    class DummyIca:
        def __init__(self) -> None:
            self.n_components_ = 10
            self.exclude = [1, 1, 3, 3]

    monkeypatch.setattr(mod.mne.preprocessing, "read_ica", lambda path: DummyIca())

    summary = mod._load_component_count_summary([tmp_path / "a.fif"])

    np.testing.assert_array_equal(summary.included_counts, np.array([8.0]))
    np.testing.assert_array_equal(summary.excluded_counts, np.array([2.0]))


def test_build_speech_artefact_summary_figure_writes_expected_path(monkeypatch, tmp_path: Path) -> None:
    """Figure builder should use configured output settings and write the requested path."""
    cfg = ProjectConfig(
        raw={
            "viz": {
                "speech_artefact": {
                    "dpi": 123,
                    "figsize": {"width": 9.0, "height": 3.0},
                    "psd": {"method": "welch", "fmin_hz": 1.0, "fmax_hz": 30.0, "n_fft": 128},
                }
            }
        }
    )

    monkeypatch.setattr(
        mod,
        "_load_average_psd",
        lambda *args, **kwargs: mod.PsdSummary(
            info={"chs": []},
            frequencies_hz=np.array([1.0, 2.0, 3.0]),
            channel_power=np.array([[0.4, 0.3, 0.2], [0.6, 0.2, 0.05]]),
        ),
    )
    monkeypatch.setattr(
        mod,
        "_load_component_count_summary",
        lambda *args, **kwargs: mod.ComponentCountSummary(
            subject_ids=("sub-001", "sub-002"),
            total_counts=np.array([6.0, 8.0]),
            included_counts=np.array([5.0, 6.0]),
            excluded_counts=np.array([1.0, 2.0]),
            bad_channel_counts=np.array([0.0, 0.0]),
        ),
    )

    saved = {}

    class DummyFigure:
        def suptitle(self, *args, **kwargs) -> None:
            saved["title"] = (args, kwargs)

        def tight_layout(self) -> None:
            saved["tight_layout"] = True

        def savefig(self, path, dpi, bbox_inches) -> None:
            saved["savefig"] = (Path(path), dpi, bbox_inches)

    figure = DummyFigure()
    axes = [object(), object(), object()]
    monkeypatch.setattr(mod.plt, "subplots", lambda *args, **kwargs: (figure, axes))
    monkeypatch.setattr(mod, "_plot_psd", lambda axis, summary, title: saved.setdefault("psd_titles", []).append(title))
    monkeypatch.setattr(mod, "_remove_second_panel_y_axis", lambda axis: saved.setdefault("remove_second_panel_y_axis", axis))
    monkeypatch.setattr(mod, "_plot_component_counts", lambda axis, summary: None)
    monkeypatch.setattr(mod, "_materialize_prebandpass_inputs", lambda run_paths, cfg, scratch_root, apply_ica: list(run_paths))
    monkeypatch.setattr(mod, "_add_bad_channel_counts", lambda summary, cfg, run_paths: summary)
    monkeypatch.setattr(mod, "_sort_component_summary", lambda summary: summary)
    monkeypatch.setattr(mod, "_reduce_first_second_gap", lambda axes, delta: saved.setdefault("gap_adjust", (axes, delta)))
    monkeypatch.setattr(mod, "_reduce_second_third_gap", lambda axes, delta: saved.setdefault("gap_adjust_2", (axes, delta)))
    monkeypatch.setattr(mod.plt, "close", lambda fig: saved.setdefault("closed", fig))

    output_path = tmp_path / "reports" / "speech_artefact_summary.png"
    written = mod.build_speech_artefact_summary_figure(
        cfg=cfg,
        filtered_noica_paths=[tmp_path / "noica.fif"],
        filtered_paths=[tmp_path / "filtered.fif"],
        ica_paths=[tmp_path / "ica.fif"],
        output_path=output_path,
    )

    assert written == output_path
    assert saved["savefig"] == (output_path, 123, "tight")
    assert saved["psd_titles"] == ["PSD: no ICA", "PSD: with ICA"]
    assert saved["remove_second_panel_y_axis"] is axes[1]
    assert saved["gap_adjust"] == (axes, mod.FIRST_SECOND_GAP_REDUCTION)
    assert saved["gap_adjust_2"] == (axes, mod.SECOND_THIRD_GAP_REDUCTION)
    assert saved["closed"] is figure
    assert "title" not in saved


def test_load_average_psd_uses_requested_method_and_per_channel_average(monkeypatch, tmp_path: Path) -> None:
    """PSD aggregation should use MNE per-channel spectra and honor the configured method."""

    class DummySpectrum:
        def __init__(self, data: np.ndarray, freqs: np.ndarray) -> None:
            self._data = data
            self.freqs = freqs

        def get_data(self) -> np.ndarray:
            return self._data

    class DummyRaw:
        def __init__(self, spectra: np.ndarray, ch_names: list[str]) -> None:
            self.spectra = spectra
            self.ch_names = ch_names
            self.info = {"ch_names": ch_names}
            self.calls: list[dict[str, object]] = []

        def compute_psd(self, **kwargs):
            self.calls.append(kwargs)
            return DummySpectrum(self.spectra, np.array([1.0, 2.0, 3.0]))

        def copy(self):
            return self

        def pick(self, picks):
            self.info = {"ch_names": [self.ch_names[index] for index in picks]}
            return self

    raws = {
        "a_raw.fif": DummyRaw(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), ["Cz", "Pz"]),
        "b_raw.fif": DummyRaw(np.array([[5.0, 6.0, 7.0], [7.0, 8.0, 9.0]]), ["Cz", "Pz"]),
    }
    monkeypatch.setattr(mod.mne.io, "read_raw_fif", lambda path, **kwargs: raws[Path(path).name])
    monkeypatch.setattr(mod.mne, "pick_types", lambda info, eeg, meg, exclude=(): [0, 1])

    summary = mod._load_average_psd(
        [tmp_path / "a_raw.fif", tmp_path / "b_raw.fif"],
        method="multitaper",
        fmin_hz=1.0,
        fmax_hz=30.0,
        n_fft=128,
    )

    assert raws["a_raw.fif"].calls == [{"method": "multitaper", "fmin": 1.0, "fmax": 30.0, "verbose": "ERROR"}]
    assert raws["b_raw.fif"].calls == [{"method": "multitaper", "fmin": 1.0, "fmax": 30.0, "verbose": "ERROR"}]
    assert summary.info == {"ch_names": ["Cz", "Pz"]}
    np.testing.assert_allclose(summary.frequencies_hz, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(summary.channel_power, np.array([[3.0, 4.0, 5.0], [5.0, 6.0, 7.0]]))
def test_plot_psd_uses_mne_spectrum_plot(monkeypatch) -> None:
    """PSD panel rendering should delegate to MNE's default Spectrum plot."""
    summary = mod.PsdSummary(
        info={"ch_names": ["Cz", "Pz"]},
        frequencies_hz=np.array([1.0, 2.0, 3.0]),
        channel_power=np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
    )

    captured = {}

    class DummySpectrum:
        def plot(self, **kwargs):
            captured["plot_kwargs"] = kwargs

    monkeypatch.setattr(mod.mne.time_frequency, "SpectrumArray", lambda data, info, freqs: captured.update({"data": data, "info": info, "freqs": freqs}) or DummySpectrum())

    class DummyLine:
        def set_linewidth(self, value):
            captured.setdefault("linewidths", []).append(value)

    class DummyAxis:
        def __init__(self):
            self.lines = [DummyLine(), DummyLine()]

        def set_title(self, value):
            captured["title"] = value

    axis = DummyAxis()
    mod._plot_psd(axis, summary, title="Filtered no ICA")

    np.testing.assert_allclose(captured["data"], summary.channel_power)
    np.testing.assert_allclose(captured["freqs"], summary.frequencies_hz)
    assert captured["info"] is summary.info
    assert captured["plot_kwargs"]["axes"] is axis
    assert captured["plot_kwargs"]["show"] is False
    assert captured["plot_kwargs"]["average"] is False
    assert captured["linewidths"] == [1.8, 1.8]
    assert captured["title"] == "Filtered no ICA"


def test_remove_second_panel_y_axis_hides_ticks_and_label() -> None:
    """Second PSD panel should suppress y-axis ticks and label."""
    captured = {}

    class DummyAxis:
        def set_ylabel(self, value):
            captured["ylabel"] = value

        def set_yticks(self, value):
            captured["yticks"] = value

    mod._remove_second_panel_y_axis(DummyAxis())

    assert captured["ylabel"] == ""
    assert captured["yticks"] == []


def test_reduce_first_second_gap_moves_second_axis_left() -> None:
    """Gap reduction helper should shift the second axis left by the requested amount."""

    class DummyBox:
        def __init__(self, x0, x1, y0=0.1, height=0.3):
            self.x0 = x0
            self.x1 = x1
            self.y0 = y0
            self.height = height
            self.width = x1 - x0

    class DummyAxis:
        def __init__(self, box):
            self._box = box
            self.updated = None

        def get_position(self):
            return self._box

        def set_position(self, value):
            self.updated = value

    first = DummyAxis(DummyBox(0.1, 0.4))
    second = DummyAxis(DummyBox(0.5, 0.8))

    mod._reduce_first_second_gap([first, second], delta=0.025)

    assert second.updated == [0.475, 0.1, 0.30000000000000004, 0.3]


def test_reduce_second_third_gap_moves_third_axis_left() -> None:
    """Second-third gap reduction helper should shift the third axis left only."""

    class DummyBox:
        def __init__(self, x0, x1, y0=0.1, height=0.3):
            self.x0 = x0
            self.x1 = x1
            self.y0 = y0
            self.height = height
            self.width = x1 - x0

    class DummyAxis:
        def __init__(self, box):
            self._box = box
            self.updated = None

        def get_position(self):
            return self._box

        def set_position(self, value):
            self.updated = value

    second = DummyAxis(DummyBox(0.5, 0.8))
    third = DummyAxis(DummyBox(0.9, 1.2))

    mod._reduce_second_third_gap([object(), second, third], delta=0.04)

    assert third.updated == [0.86, 0.1, 0.29999999999999993, 0.3]


def test_add_bad_channel_counts_uses_max_bads_per_subject(monkeypatch, tmp_path: Path) -> None:
    """Bad-channel counts should come from the subject's channels.tsv files."""
    cfg = ProjectConfig(
        raw={
            "paths": {
                "bids_root": str(tmp_path),
                "out_dir": str(tmp_path / "derived"),
                "reports_root": str(tmp_path / "reports"),
            }
        }
    )
    summary = mod.ComponentCountSummary(
        subject_ids=("sub-001", "sub-002"),
        total_counts=np.array([62.0, 63.0]),
        included_counts=np.array([60.0, 61.0]),
        excluded_counts=np.array([2.0, 2.0]),
        bad_channel_counts=np.array([0.0, 0.0]),
    )

    monkeypatch.setattr(
        mod,
        "_count_bad_channels",
        lambda path: 3 if "sub-001" in str(path) else 1,
    )

    updated = mod._add_bad_channel_counts(
        summary,
        cfg=cfg,
        run_paths=[
            tmp_path / "sub-001_task-conversation_run-1_raw_interp_noica.fif",
            tmp_path / "sub-001_task-conversation_run-2_raw_interp_noica.fif",
            tmp_path / "sub-002_task-conversation_run-1_raw_interp_noica.fif",
        ],
    )

    np.testing.assert_array_equal(updated.bad_channel_counts, np.array([2.0, 1.0]))
    np.testing.assert_array_equal(updated.total_counts, np.array([64.0, 64.0]))


def test_sort_component_summary_orders_subjects_by_included_counts() -> None:
    """Subjects should be ordered by ascending included-component counts."""
    summary = mod.ComponentCountSummary(
        subject_ids=("sub-003", "sub-001", "sub-002"),
        total_counts=np.array([64.0, 64.0, 64.0]),
        included_counts=np.array([10.0, 8.0, 9.0]),
        excluded_counts=np.array([50.0, 54.0, 53.0]),
        bad_channel_counts=np.array([4.0, 2.0, 2.0]),
    )

    sorted_summary = mod._sort_component_summary(summary)

    assert sorted_summary.subject_ids == ("sub-001", "sub-002", "sub-003")
    np.testing.assert_array_equal(sorted_summary.included_counts, np.array([8.0, 9.0, 10.0]))
    np.testing.assert_array_equal(sorted_summary.bad_channel_counts, np.array([2.0, 2.0, 4.0]))


def test_materialize_prebandpass_input_rebuilds_missing_run(monkeypatch, tmp_path: Path) -> None:
    """Missing pre-bandpass inputs should be rebuilt in scratch space without Snakemake."""
    cfg = ProjectConfig(
        raw={
            "paths": {
                "bids_root": str(tmp_path / "bids"),
                "out_dir": str(tmp_path / "derived"),
                "reports_root": str(tmp_path / "reports"),
                "precomputed_ica_root": str(tmp_path / "ica"),
            },
            "preprocessing": {
                "interpolation": {"method": "nearest"},
            },
        }
    )
    run_path = tmp_path / "derived" / "eeg" / "interpolated" / "sub-001_task-conversation_run-1_raw_interp.fif"
    calls = {}

    monkeypatch.setattr(mod, "_ensure_downsampled_input", lambda run_path, cfg, scratch_root: tmp_path / "scratch" / "raw_ds.fif")
    monkeypatch.setattr(mod, "_resolve_channels_tsv_path", lambda paths, run_path: tmp_path / "bids" / "sub-001_channels.tsv")
    monkeypatch.setattr(mod, "_resolve_ica_path", lambda cfg, run_path: tmp_path / "ica" / "sub-001_task-conversation-ica.fif")

    def _stub_reref(**kwargs):
        calls["reref"] = kwargs
        kwargs["output_fif_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_fif_path"].write_text("reref")

    def _stub_ica(**kwargs):
        calls["ica"] = kwargs
        kwargs["output_fif_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_fif_path"].write_text("ica")

    def _stub_interp(**kwargs):
        calls["interp"] = kwargs
        kwargs["output_fif_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["output_fif_path"].write_text("interp")

    monkeypatch.setattr(mod, "rereference_fif_to_fif", _stub_reref)
    monkeypatch.setattr(mod, "apply_ica_fif_to_fif", _stub_ica)
    monkeypatch.setattr(mod, "interpolate_bads_fif_to_fif", _stub_interp)

    rebuilt = mod._materialize_prebandpass_input(
        run_path,
        cfg=cfg,
        scratch_root=tmp_path / "scratch",
        apply_ica=True,
    )

    assert rebuilt.name == "sub-001_task-conversation_run-1_raw_interp.fif"
    assert rebuilt.exists()
    assert calls["reref"]["input_fif_path"] == tmp_path / "scratch" / "raw_ds.fif"
    assert calls["ica"]["ica_path"] == tmp_path / "ica" / "sub-001_task-conversation-ica.fif"
    assert calls["interp"]["method"] == "nearest"


def test_plot_component_counts_sets_barplot_headroom_and_tilt() -> None:
    """The component-count panel should reserve legend space and tilt subject labels."""
    summary = mod.ComponentCountSummary(
        subject_ids=("sub-001", "sub-002"),
        total_counts=np.array([64.0, 64.0]),
        included_counts=np.array([8.0, 12.0]),
        excluded_counts=np.array([54.0, 50.0]),
        bad_channel_counts=np.array([2.0, 2.0]),
    )

    captured = {}

    class DummyAxis:
        def bar(self, *args, **kwargs):
            captured.setdefault("bars", []).append((args, kwargs))

        def set_xticks(self, positions, labels, rotation=None, ha=None):
            captured["xticks"] = (positions, labels, rotation, ha)

        def set_xlabel(self, value):
            captured["xlabel"] = value

        def set_ylabel(self, value):
            captured["ylabel"] = value

        def set_title(self, value):
            captured["title"] = value

        def set_ylim(self, ymin, ymax):
            captured["ylim"] = (ymin, ymax)

        def legend(self, **kwargs):
            captured["legend"] = kwargs

        def grid(self, *args, **kwargs):
            captured["grid"] = (args, kwargs)

    axis = DummyAxis()
    mod._plot_component_counts(axis, summary)

    assert captured["xticks"][2] == 75
    assert captured["xticks"][3] == "right"
    assert captured["ylabel"] == "Number of components"
    assert captured["title"] == "ICA component counts"
    assert captured["ylim"] == (0.0, 68.0)
    assert captured["legend"]["loc"] == "upper left"
    assert captured["legend"]["bbox_to_anchor"] == (1.02, 1.0)


def test_scale_figure_fonts_multiplies_text_sizes() -> None:
    """Figure-wide font scaling should multiply text sizes uniformly."""

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

    figure = DummyFigure()
    mod._scale_figure_fonts(figure, scale=2.0)

    assert [text.get_fontsize() for text in figure.texts] == [20.0, 28.0]


def test_tune_speech_artefact_fonts_applies_panel_specific_scales() -> None:
    """Speech artefact figure should reduce y labels, legend text, and third-panel x ticks."""

    class DummyText:
        def __init__(self, fontsize: float) -> None:
            self._fontsize = fontsize

        def get_fontsize(self) -> float:
            return self._fontsize

        def set_fontsize(self, value: float) -> None:
            self._fontsize = value

    class DummyLegend:
        def __init__(self) -> None:
            self._texts = [DummyText(20.0), DummyText(24.0)]

        def get_texts(self):
            return self._texts

    class DummyAxis:
        def __init__(self, *, with_legend: bool = False, tick_sizes: list[float] | None = None) -> None:
            self.yaxis = type("YAxis", (), {"label": DummyText(30.0)})()
            self._ticks = [DummyText(size) for size in (tick_sizes or [])]
            self._legend = DummyLegend() if with_legend else None

        def get_xticklabels(self):
            return self._ticks

        def get_legend(self):
            return self._legend

    axes = [
        DummyAxis(),
        DummyAxis(),
        DummyAxis(with_legend=True, tick_sizes=[18.0, 22.0]),
    ]

    mod._tune_speech_artefact_fonts(axes)

    assert axes[0].yaxis.label.get_fontsize() == 21.0
    assert axes[1].yaxis.label.get_fontsize() == 21.0
    assert axes[2].yaxis.label.get_fontsize() == 21.0
    assert [tick.get_fontsize() for tick in axes[2].get_xticklabels()] == [9.0, 11.0]
    assert [text.get_fontsize() for text in axes[2].get_legend().get_texts()] == [10.0, 12.0]
