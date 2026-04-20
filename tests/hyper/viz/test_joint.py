"""Tests for joint-plot helpers."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest

from hyper.viz import joint as mod


def test_resolve_joint_times_prefers_significant_peaks() -> None:
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    beta_map = np.array(
        [
            [0.1, 0.2, 9.0, 0.3, 6.0],
            [0.0, 0.2, 8.0, 0.1, 4.0],
        ],
        dtype=float,
    )
    significance_mask = np.array(
        [
            [False, False, False, False, True],
            [False, False, False, False, False],
        ],
        dtype=bool,
    )

    resolved = mod.resolve_joint_times(beta_map, times, significance_mask=significance_mask)

    assert resolved.tolist() == [0.4]


def test_contiguous_true_spans_returns_half_step_expanded_ranges() -> None:
    times = np.array([0.0, 0.2, 0.4, 0.6], dtype=float)
    mask = np.array([False, True, True, False], dtype=bool)

    spans = mod.contiguous_true_spans(mask, times)

    assert spans == [(0.1, 0.5)]


def test_plot_joint_map_retries_without_overlapping_channels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    create_info_calls: list[tuple[list[str], float, list[str]]] = []

    class FakeInfo:
        def set_montage(self, montage: str, on_missing: str = "ignore") -> None:
            assert montage == "standard_1020"
            assert on_missing == "ignore"

    class FakeLine:
        def __init__(self) -> None:
            self.widths: list[float] = []

        def set_linewidth(self, value: float) -> None:
            self.widths.append(value)

    class FakeAxis:
        def __init__(self) -> None:
            self.lines = [FakeLine()]
            self.spans: list[tuple[float, float, str, float, int]] = []
            self.ylabel: str | None = None
            self.yticks: list[float] | None = None
            self._title = "0.100 s"
            self.title_y: float | None = None
            self.title_fontsize: float = 10.0
            self.title = types.SimpleNamespace(
                set_y=lambda value: setattr(self, "title_y", value),
                get_fontsize=lambda: self.title_fontsize,
                set_fontsize=lambda value: setattr(self, "title_fontsize", value),
            )

        def axvspan(self, start: float, end: float, *, color: str, alpha: float, zorder: int) -> None:
            self.spans.append((start, end, color, alpha, zorder))

        def set_ylabel(self, value: str) -> None:
            self.ylabel = value

        def set_yticks(self, ticks) -> None:  # noqa: ANN001
            self.yticks = list(ticks)

        def get_title(self) -> str:
            return self._title

    class FakeFigure:
        def __init__(self) -> None:
            self.axes = [FakeAxis()]
            self.saved: list[tuple[Path, int, str]] = []

        def savefig(self, path: Path, dpi: int, bbox_inches: str) -> None:
            self.saved.append((Path(path), dpi, bbox_inches))

        def findobj(self, cls):  # noqa: ANN001
            del cls
            return [
                types.SimpleNamespace(
                    get_text=lambda: "Nave = 12",
                    set_text=lambda value: setattr(self, "nave_text", value),
                    get_fontsize=lambda: 10.0,
                    set_fontsize=lambda value: setattr(self, "scaled_fontsize", value),
                )
            ]

        nave_text: str | None = None
        scaled_fontsize: float | None = None

    class FakeEvoked:
        def __init__(self, beta_map: np.ndarray, info: FakeInfo, tmin: float, nave: int, comment: str) -> None:
            self.beta_map = beta_map
            self.info = info
            self.tmin = tmin
            self.nave = nave
            self.comment = comment
            self.ch_names = ["Cz", "Pz", "Oz"]
            self.pick_history: list[list[str]] = []
            self.plot_calls: list[dict[str, object]] = []
            self.should_raise_overlap = True
            self.figure = FakeFigure()

        def plot_joint(self, *, times: np.ndarray, title: str | None, show: bool, topomap_args: dict[str, object]):
            self.plot_calls.append(
                {
                    "times": np.asarray(times, dtype=float),
                    "title": title,
                    "show": show,
                    "topomap_args": topomap_args,
                    "ch_names": list(self.ch_names),
                }
            )
            if self.should_raise_overlap:
                self.should_raise_overlap = False
                raise ValueError("overlapping positions\nPz")
            return self.figure

        def copy(self) -> "FakeEvoked":
            duplicate = FakeEvoked(self.beta_map, self.info, self.tmin, self.nave, self.comment)
            duplicate.ch_names = list(self.ch_names)
            duplicate.pick_history = self.pick_history
            duplicate.plot_calls = self.plot_calls
            duplicate.should_raise_overlap = self.should_raise_overlap
            duplicate.figure = self.figure
            return duplicate

        def pick(self, keep_channels: list[str]) -> "FakeEvoked":
            self.pick_history.append(list(keep_channels))
            self.ch_names = [name for name in self.ch_names if name in keep_channels]
            return self

    fake_evokeds: list[FakeEvoked] = []

    def fake_create_info(ch_names: list[str], sfreq: float, ch_types: list[str]) -> FakeInfo:
        create_info_calls.append((list(ch_names), sfreq, list(ch_types)))
        return FakeInfo()

    def fake_evoked_array(beta_map: np.ndarray, info: FakeInfo, tmin: float, nave: int, comment: str) -> FakeEvoked:
        evoked = FakeEvoked(beta_map, info, tmin, nave, comment)
        fake_evokeds.append(evoked)
        return evoked

    fake_mne = types.SimpleNamespace(create_info=fake_create_info, EvokedArray=fake_evoked_array)
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    close_calls: list[FakeFigure] = []
    monkeypatch.setattr(mod.plt, "close", lambda figure: close_calls.append(figure))

    beta_map = np.array(
        [
            [1.0, 2.0, 0.5, 0.0],
            [0.5, 1.0, 0.5, 0.0],
            [0.1, 0.8, 0.2, 0.0],
        ],
        dtype=float,
    )
    significance_mask = np.array(
        [
            [False, True, True, False],
            [False, False, True, False],
            [False, False, False, False],
        ],
        dtype=bool,
    )

    written = mod.plot_joint_map(
        beta_map,
        times=np.array([0.0, 0.1, 0.2, 0.3], dtype=float),
        channel_names=["Cz", "Pz", "Oz"],
        output_stem=tmp_path / "joint" / "effect",
        title="demo",
        formats=("png",),
        line_width=3.5,
        significance_mask=significance_mask,
        ylabel="A.U.",
        show_colorbar=False,
    )

    assert written == [tmp_path / "joint" / "effect.png"]
    assert create_info_calls == [(["Cz", "Pz", "Oz"], 10.0, ["eeg", "eeg", "eeg"])]
    assert len(fake_evokeds) == 1
    evoked = fake_evokeds[0]
    assert evoked.pick_history == [["Cz", "Oz"]]
    assert len(evoked.plot_calls) == 2
    initial_call, retry_call = evoked.plot_calls
    assert initial_call["title"] == "demo"
    assert initial_call["show"] is False
    np.testing.assert_allclose(initial_call["times"], np.array([0.1, 0.2], dtype=float))
    np.testing.assert_array_equal(initial_call["topomap_args"]["mask"], significance_mask)
    np.testing.assert_array_equal(
        retry_call["topomap_args"]["mask"],
        significance_mask[[0, 2]],
    )
    figure = evoked.figure
    assert figure.saved == [(tmp_path / "joint" / "effect.png", 300, "tight")]
    assert figure.axes[0].spans == [(0.05, 0.25, "0.85", 0.7, 0)]
    assert figure.axes[0].lines[0].widths == [3.5]
    assert figure.axes[0].ylabel == "A.U."
    assert figure.axes[0].yticks == []
    assert figure.nave_text == ""
    assert figure.axes[0].title_y == pytest.approx(1.14)
    assert figure.axes[0].title_fontsize == pytest.approx(8.0)
    assert close_calls == [figure]


def test_no_colorbar_topomap_args_reports_false_without_storing_key() -> None:
    """The colorbar-suppression wrapper should hide the key from kwargs expansion."""
    wrapped = mod._NoColorbarTopomapArgs({"mask": "demo"})

    assert wrapped.get("colorbar", True) is False
    assert "colorbar" not in wrapped
    assert isinstance(wrapped.copy(), mod._NoColorbarTopomapArgs)
