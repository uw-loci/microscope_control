"""``autofocus`` keeps the peak validation of the Z it committed.

The acquisition's focus surface weighs a result by how clear its peak was (a clear
peak must not be overruled by a surface built from faint ones), and on success
``autofocus`` returns only the Z, so the validation is kept on the hardware object.
"""

import numpy as np

from tests.test_autofocus_edge_retry import _make_dense_hardware


def _metric(img):
    return float(np.mean(img))


def test_a_committed_autofocus_keeps_its_peak_validation():
    def score_fn(z):
        return 1000.0 * np.exp(-0.5 * (z / 3.0) ** 2) + 10.0

    hw, _ = _make_dense_hardware(score_fn)
    z = hw.autofocus(
        n_steps=15, search_range=30.0, score_metric=_metric, move_stage_to_estimate=False
    )

    assert isinstance(z, float)
    validation = hw.last_autofocus_validation
    assert validation is not None
    assert validation["is_valid"]
    assert validation["peak_prominence"] > 0.5


def test_an_autofocus_that_commits_nothing_clears_it():
    def peak(z):
        return 1000.0 * np.exp(-0.5 * (z / 3.0) ** 2) + 10.0

    hw, _ = _make_dense_hardware(peak)
    hw.autofocus(n_steps=15, search_range=30.0, score_metric=_metric, move_stage_to_estimate=False)
    assert hw.last_autofocus_validation is not None

    hw2, _ = _make_dense_hardware(lambda z: 500.0)
    hw2.last_autofocus_validation = hw.last_autofocus_validation
    result = hw2.autofocus(
        n_steps=15, search_range=30.0, score_metric=_metric, move_stage_to_estimate=False
    )

    assert isinstance(result, dict) and not result["success"]
    assert hw2.last_autofocus_validation is None
