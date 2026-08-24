"""Live-frame reshape while streaming autofocus holds a cropped ROI.

Streaming autofocus centres a crop ROI on the camera for the duration of a scan.
While that crop is held, the camera delivers frames at the cropped size (a JAI
rounds a requested 1032x772 down to 1024x772) but MM Core can still report the
full sensor. Reshaping by Core's numbers then paints a full-sensor image whose
leading rows are the new frame and whose trailing rows are whatever was in the
buffer before -- the live contamination bar. In the 2026-08-14 session this state
was present for 12 of 19 autofocus runs.

These tests pin the rule that fixes it: believe the camera's per-frame tags over
Core's dimensions, and never display pixels the camera did not just write.
"""

import numpy as np
import pytest

from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera

FULL_W, FULL_H = 2064, 1544
CROP_W, CROP_H = 1024, 772


class _TaggedImage:
    def __init__(self, pix, width, height):
        self.pix = pix
        self.tags = {"Width": width, "Height": height}


class _FakeCore:
    """Minimal MM Core stand-in for the live pop path."""

    def __init__(self, tagged, core_w, core_h, ncomponents=1, remaining=1):
        self._tagged = tagged
        self._core_w = core_w
        self._core_h = core_h
        self._nc = ncomponents
        self._remaining = remaining

    def get_remaining_image_count(self):
        return self._remaining

    def get_last_tagged_image(self):
        return self._tagged

    def get_image_width(self):
        return self._core_w

    def get_image_height(self):
        return self._core_h

    def get_number_of_components(self):
        return self._nc

    def get_image_buffer_size(self):
        return self._core_w * self._core_h * self._nc


def _camera(core):
    cam = PycromanagerCamera.__new__(PycromanagerCamera)
    cam._core = core
    cam._studio = None
    # Skip debayering/channel juggling: this is a geometry test.
    cam._process_raw_image = lambda pixels, w, h, nch, debayering="auto": np.asarray(
        pixels
    ).reshape(h, w)
    return cam


def test_agreeing_dimensions_are_reshaped_normally():
    pixels = np.arange(FULL_W * FULL_H, dtype=np.uint16)
    core = _FakeCore(_TaggedImage(pixels, FULL_W, FULL_H), FULL_W, FULL_H)

    frame, meta = _camera(core).get_live_frame()

    assert frame is not None
    assert frame.shape == (FULL_H, FULL_W)
    assert meta["width"] == FULL_W and meta["height"] == FULL_H


def test_cropped_buffer_is_reshaped_by_tags_not_core():
    """The camera delivered exactly the cropped frame; Core still says full."""
    pixels = np.arange(CROP_W * CROP_H, dtype=np.uint16)
    core = _FakeCore(_TaggedImage(pixels, CROP_W, CROP_H), FULL_W, FULL_H)

    frame, meta = _camera(core).get_live_frame()

    assert frame is not None, "a valid cropped frame must not be dropped"
    assert frame.shape == (CROP_H, CROP_W)
    assert meta["width"] == CROP_W and meta["height"] == CROP_H


def test_full_sized_buffer_with_small_frame_drops_the_stale_remainder():
    """The contamination case: full-sensor buffer, only the leading region fresh.

    Sizes match Core here, so the old pixel-count guard passed and the stale
    tail was rendered. Only the freshly written pixels may survive.
    """
    # Flat values, not a ramp: CROP_W*CROP_H overflows uint16, and a wrapped
    # ramp would coincidentally contain the sentinel.
    fresh = np.full(CROP_W * CROP_H, 111, dtype=np.uint16)
    stale = np.full(FULL_W * FULL_H - fresh.size, 9999, dtype=np.uint16)
    pixels = np.concatenate([fresh, stale])
    core = _FakeCore(_TaggedImage(pixels, CROP_W, CROP_H), FULL_W, FULL_H)

    frame, meta = _camera(core).get_live_frame()

    assert frame is not None
    assert frame.shape == (CROP_H, CROP_W)
    assert meta["width"] == CROP_W and meta["height"] == CROP_H
    assert 9999 not in frame, "stale buffer content leaked into the displayed frame"
    np.testing.assert_array_equal(frame.reshape(-1), fresh)


def test_unexplained_size_is_skipped_rather_than_guessed():
    """A buffer matching neither geometry is a torn frame; drop it."""
    pixels = np.zeros(12345, dtype=np.uint16)
    core = _FakeCore(_TaggedImage(pixels, CROP_W, CROP_H), FULL_W, FULL_H)

    frame, meta = _camera(core).get_live_frame()

    assert frame is None and meta is None


def test_empty_buffer_returns_nothing():
    core = _FakeCore(None, FULL_W, FULL_H, remaining=0)
    frame, meta = _camera(core).get_live_frame()
    assert frame is None and meta is None


@pytest.mark.parametrize("ncomponents", [1, 4])
def test_multi_component_frames_use_tag_geometry(ncomponents):
    """RGBA live frames must slice on the component-aware size, not pixel count."""
    fresh = np.full(CROP_W * CROP_H * ncomponents, 111, dtype=np.uint16)
    stale = np.full(FULL_W * FULL_H * ncomponents - fresh.size, 9999, dtype=np.uint16)
    pixels = np.concatenate([fresh, stale])
    core = _FakeCore(_TaggedImage(pixels, CROP_W, CROP_H), FULL_W, FULL_H, ncomponents=ncomponents)
    cam = _camera(core)
    cam._process_raw_image = lambda p, w, h, nch, debayering="auto": np.asarray(p).reshape(
        h, w, nch
    )

    frame, _ = cam.get_live_frame()

    assert frame is not None
    assert frame.size == CROP_W * CROP_H * ncomponents
    assert 9999 not in frame
