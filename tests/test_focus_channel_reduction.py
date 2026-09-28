"""Standard AF honours ``channel_reduction`` from autofocus_<scope>.yml.

The key existed from 2026-09-12 but was read only by streaming AF, so on the scan path
that actually runs during an acquisition the JAI kept scoring an equal-weighted mean --
the reduction under which achromatic debris outranks stained tissue. A 2026-09-24 PPM run
focused on dust with ``channel_reduction: green`` sitting in the config.

The default must stay the camera's own ``extract_green_channel``: that method is three
different behaviours (Bayer takes green, LSM passes mono through, JAI means across RGB),
and collapsing them would move every camera at once.
"""

import numpy as np
import pytest

from microscope_control.hardware.pycromanager import PycromanagerHardware


class _Camera:
    """Stands in for the JAI: equal mean across RGB, as its real method does."""

    def __init__(self):
        self.calls = 0

    def extract_green_channel(self, img):
        self.calls += 1
        if img.ndim == 3:
            return np.mean(img, axis=2).astype(np.float32)
        return img.astype(np.float32)


class _Hardware:
    """Minimal stand-in: _reduce_for_focus only touches self.camera."""

    _focus_reduction_logged = False

    def __init__(self):
        self.camera = _Camera()


def _reduce(hw, img, reduction):
    return PycromanagerHardware._reduce_for_focus(hw, img, reduction)


def _bgra():
    """One frame where the three colour planes differ, so the reduction is visible."""
    img = np.zeros((4, 4, 4), dtype=np.uint8)
    img[:, :, 0] = 10  # B
    img[:, :, 1] = 200  # G
    img[:, :, 2] = 30  # R
    img[:, :, 3] = 255  # A, must be ignored either way
    return img


@pytest.mark.parametrize("reduction", [None, "equal_mean"])
def test_default_and_equal_mean_keep_the_cameras_own_reduction(reduction):
    hw = _Hardware()
    out = _reduce(hw, _bgra(), reduction)

    assert hw.camera.calls == 1, "the camera method must still be the default path"
    # mean of B,G,R,A as the JAI stub computes it -- the point is that it is NOT 200.
    assert not np.allclose(out, 200.0)


def test_green_takes_the_green_plane_instead():
    hw = _Hardware()
    out = _reduce(hw, _bgra(), "green")

    assert hw.camera.calls == 0, "an explicit reduction must bypass the camera method"
    assert np.allclose(out, 200.0), "green is index 1 in both BGRA and RGB"


def test_green_on_a_mono_frame_is_a_passthrough_not_an_error():
    # A mono camera with green configured must not crash; to_gray needs >=3 channels
    # before it can take a plane, and falls back otherwise.
    hw = _Hardware()
    mono = np.full((4, 4), 42, dtype=np.uint8)
    out = _reduce(hw, mono, "green")

    assert out.shape == (4, 4)
    assert np.allclose(out, 42.0)


def test_the_reduction_is_announced_only_once_per_instance(caplog):
    # Named once per hardware instance, not once per Z step -- a 100-step scan would
    # otherwise emit 100 identical lines.
    hw = _Hardware()
    with caplog.at_level("INFO"):
        for _ in range(5):
            _reduce(hw, _bgra(), "green")

    announcements = [r for r in caplog.records if "channel_reduction" in r.getMessage()]
    assert len(announcements) == 1
