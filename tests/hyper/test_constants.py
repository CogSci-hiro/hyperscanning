"""Sanity checks for canonical EEG band constants."""

from hyper.constants import ALPHA_BAND_HZ, BETA_BAND_HZ, THETA_BAND_HZ


def test_frequency_bands_are_ordered_and_non_overlapping() -> None:
    """Band boundaries should be increasing and adjacent at shared boundaries."""
    assert THETA_BAND_HZ == (4.0, 7.0)
    assert ALPHA_BAND_HZ == (7.0, 13.0)
    assert BETA_BAND_HZ == (13.0, 30.0)
    assert THETA_BAND_HZ[1] <= ALPHA_BAND_HZ[0]
    assert ALPHA_BAND_HZ[1] <= BETA_BAND_HZ[0]
