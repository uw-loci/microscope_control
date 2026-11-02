"""One focus-score table for standard autofocus, with the canonical Brenner.

The acquisition workflow's per-tile autofocus, the autofocus test and the benchmark each
kept their own name-to-function table. The test's sent ``p98_p2`` to laplacian variance,
which reads flat on bright PPM backgrounds: the fall-through the acquisition workflow had
already been fixed for. Brenner had three formulas across the packages.
"""

import logging

import numpy as np
from microscope_imageprocessing.focus import resolve_metric

from microscope_control.autofocus.core import AutofocusUtils
from microscope_control.autofocus.test import _load_autofocus_settings


def test_p98_p2_maps_to_p98_p2():
    assert AutofocusUtils.score_metric("p98_p2") is AutofocusUtils.autofocus_profile_p98_p2


def test_unmapped_name_falls_back_to_laplacian_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        fn = AutofocusUtils.score_metric("normalized_variance")
    assert fn is AutofocusUtils.autofocus_profile_laplacian_variance
    assert "normalized_variance" in caplog.text


def test_autofocus_test_loads_p98_p2_from_the_yaml(tmp_path):
    # The loader is given the microscope config and reads autofocus_<name>.yml beside it.
    (tmp_path / "autofocus_Test.yml").write_text(
        "autofocus_settings:\n  - objective: OBJ_20X\n    score_metric: p98_p2\n", encoding="ascii"
    )
    settings = _load_autofocus_settings(
        str(tmp_path / "config_Test.yml"), "OBJ_20X", logging.getLogger("t")
    )
    assert settings["score_metric_name"] == "p98_p2"
    assert settings["score_metric"] is AutofocusUtils.autofocus_profile_p98_p2


def test_brenner_is_the_canonical_lag2_form():
    rng = np.random.default_rng(3)
    image = rng.uniform(0, 4000, size=(48, 64))
    assert AutofocusUtils.autofocus_profile_brenner_gradient(image) == resolve_metric(
        "brenner_gradient"
    )(image)
    # Structure only down the rows: no horizontal lag-2 difference, so zero. The old
    # np.gradient form scored it on the vertical gradient.
    rows_only = np.repeat(rng.uniform(50, 200, size=48)[:, None], 64, axis=1)
    assert AutofocusUtils.autofocus_profile_brenner_gradient(rows_only) == 0.0
    assert AutofocusUtils.autofocus_profile_hybrid_sharpness_metric(rows_only) == 0.0
