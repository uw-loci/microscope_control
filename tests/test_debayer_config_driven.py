"""Debayering is decided per detector, from config, or not at all.

Before 2026-08-07 the decision was `self._name == "MicroPublisher6"` -- a
hardcoded device name -- while the detector block for that same camera
declares `device: QCamera` and `requires_debayering: true`. Whether
debayering happened therefore depended on what Micro-Manager called the
device at runtime, and the config key that stated the intent was read only
by the Java disk-space estimator, never by the code that debayers.

The mosaic order and bit depth were then defaulted to 'GRBG' and 14. Both
are physical properties of a sensor: a wrong mosaic swaps red and blue in
every frame, a wrong bit depth mis-scales the whole image, and neither
raises.
"""

import numpy as np
import pytest

from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera


class FakeCore:
    def __init__(self):
        self.props = {}

    def get_property(self, device, prop):
        return self.props.get((device, prop), "Cam")

    def set_property(self, device, prop, value):
        self.props[(device, prop)] = value


def _cam(**detector_config):
    detector_config.setdefault("device", "Cam")
    return PycromanagerCamera(FakeCore(), None, detector_config)


# --- The gate ------------------------------------------------------------


def test_a_detector_with_no_bayer_keys_does_not_debayer():
    # The OWS3 Hamamatsu and every laser-scanning detector look like this.
    assert _cam()._should_debayer() is False


def test_requires_debayering_false_wins_even_over_a_stray_pattern():
    # JAI's 3-CCD prism declares false explicitly; an inherited or copy-paste
    # bayer_pattern must not silently switch debayering back on.
    cam = _cam(requires_debayering=False, bayer_pattern="GRBG")
    assert cam._should_debayer() is False


def test_requires_debayering_true_turns_it_on():
    cam = _cam(requires_debayering=True, bayer_pattern="GRBG", bit_depth=14)
    assert cam._should_debayer() is True


def test_a_bayer_pattern_alone_is_enough_when_the_key_is_absent():
    assert _cam(bayer_pattern="RGGB", bit_depth=12)._should_debayer() is True


def test_the_device_name_no_longer_decides():
    # The old rule. A camera named MicroPublisher6 with no Bayer keys must
    # not debayer, and one named anything else with keys must.
    assert _cam(device="MicroPublisher6")._should_debayer() is False
    assert (
        _cam(device="SomethingElse", bayer_pattern="GRBG", bit_depth=14)._should_debayer() is True
    )


# --- Refusing to guess ---------------------------------------------------


def test_debayering_without_a_pattern_raises_rather_than_assuming_grbg():
    cam = _cam(requires_debayering=True)
    with pytest.raises(ValueError, match="bayer_pattern"):
        cam._debayer_image(np.zeros((4, 4), dtype=np.uint16))


def test_debayering_without_a_bit_depth_raises_rather_than_assuming_14():
    cam = _cam(requires_debayering=True, bayer_pattern="GRBG")
    with pytest.raises(ValueError, match="bit_depth"):
        cam._debayer_image(np.zeros((4, 4), dtype=np.uint16))


def test_a_fully_configured_detector_debayers():
    cam = _cam(requires_debayering=True, bayer_pattern="GRBG", bit_depth=14)
    out = cam._debayer_image(np.zeros((8, 8), dtype=np.uint16))
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint16


def test_the_configured_pattern_is_the_one_used():
    # Not just "it ran" -- the R and B planes must actually differ between
    # two mosaics, which is the failure a default would hide.
    img = np.zeros((4, 4), dtype=np.uint16)
    img[0, 0] = 1000  # position that is R in RGGB and G in GRBG
    grbg = _cam(bayer_pattern="GRBG", bit_depth=14)._debayer_image(img)
    rggb = _cam(bayer_pattern="RGGB", bit_depth=14)._debayer_image(img)
    assert not np.array_equal(grbg, rggb)


# --- Raw-mode toggle, also per detector ----------------------------------


def test_raw_mode_is_a_no_op_when_the_detector_does_not_declare_one():
    cam = _cam(bayer_pattern="GRBG", bit_depth=14)
    cam._set_debayer_mode_for_snap(color_off=True)
    assert cam._core.props == {}


def test_raw_mode_writes_the_configured_property_and_values():
    cam = _cam(
        device="QCamera",
        bayer_pattern="GRBG",
        bit_depth=14,
        raw_mode={"property": "Color", "raw_value": "OFF", "normal_value": "ON"},
    )
    cam._set_debayer_mode_for_snap(color_off=True)
    assert cam._core.props[("QCamera", "Color")] == "OFF"
    cam._set_debayer_mode_for_snap(color_off=False)
    assert cam._core.props[("QCamera", "Color")] == "ON"


def test_an_incomplete_raw_mode_block_raises():
    cam = _cam(raw_mode={"property": "Color"})
    with pytest.raises(ValueError, match="raw_value"):
        cam._set_debayer_mode_for_snap(color_off=True)
