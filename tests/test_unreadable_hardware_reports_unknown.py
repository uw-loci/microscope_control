"""A hardware getter that cannot read its device must say so, not guess.

Every case here is a real fabricated default that shipped. The pattern was
always the same: `except Exception: return <plausible number>`, which turns
"I could not read the device" into a confident false claim about the
hardware. See claude-reports/design/2026-08-05_llm-prone-bug-classes.md,
bug class 1.

The tests deliberately use a core whose every read raises, because that is
the state the old code handled wrongly and the state no manual test ever
reproduces on a working rig.
"""

import pytest

from microscope_control.hardware.camera.base import Camera
from microscope_control.hardware.camera.laser_scanning_camera import LaserScanningCamera
from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera
from microscope_control.hardware.detector import DCUDetector, PMTDetector
from microscope_control.hardware.illumination import (
    AnalogIllumination,
    DevicePropertyIllumination,
)
from microscope_control.jai.properties import JAICameraProperties


class DeadCore:
    """A Micro-Manager core where every property read fails.

    Models an unplugged device, a renamed property, or a driver that has
    dropped its connection -- all of which reach the getters as an exception.
    """

    def __init__(self):
        self.writes = []

    def get_property(self, device, prop):
        raise RuntimeError(f"device {device} is not responding")

    def set_property(self, device, prop, value):
        self.writes.append((device, prop, value))

    def get_allowed_property_values(self, device, prop):
        raise RuntimeError("no allowed values")

    def wait_for_device(self, device):
        pass

    def wait_for_system(self):
        pass


class _BareCamera(Camera):
    """Minimal concrete Camera, to exercise the ABC's own defaults.

    Stands in for a camera type that implements the required abstract
    methods but none of the optional binning / gain capabilities.
    """

    def snap_image(self, **kwargs):
        raise NotImplementedError

    def get_name(self) -> str:
        return "BareCamera"

    def set_exposure(self, exposure_ms: float) -> None:
        pass

    def get_exposure(self) -> float:
        return 10.0

    def get_fov_pixels(self):
        return (100, 100)

    def get_pixel_size_um(self) -> float:
        return 1.0

    def extract_green_channel(self, img):
        return img


class LiveCore:
    """A core that answers reads from a dict, for the happy-path checks."""

    def __init__(self, values):
        self._values = values
        self.writes = []

    def get_property(self, device, prop):
        return self._values[(device, prop)]

    def set_property(self, device, prop, value):
        self.writes.append((device, prop, value))
        self._values[(device, prop)] = value

    def wait_for_device(self, device):
        pass

    def wait_for_system(self):
        pass


# --- Illumination: 0.0 / off is a safety claim, not a fallback -------------


def test_analog_illumination_power_is_none_when_unreadable():
    illum = AnalogIllumination(DeadCore(), "LED-Dev1ao0")
    assert illum.get_power() is None


def test_analog_illumination_is_on_is_none_when_power_unreadable():
    # The dangerous case: False would mean "this source is dark", which is
    # what a caller checks before opening a shutter or exposing a detector.
    illum = AnalogIllumination(DeadCore(), "LED-Dev1ao0")
    assert illum.is_on() is None


def test_analog_illumination_still_reports_a_real_reading():
    core = LiveCore({("LED-Dev1ao0", "Voltage"): "2.5"})
    illum = AnalogIllumination(core, "LED-Dev1ao0")
    assert illum.get_power() == pytest.approx(2.5)
    assert illum.is_on() is True


def test_analog_illumination_reports_a_genuine_zero_as_off():
    # None and 0.0 must stay distinguishable: a real reading of zero IS off.
    core = LiveCore({("LED-Dev1ao0", "Voltage"): "0.0"})
    illum = AnalogIllumination(core, "LED-Dev1ao0")
    assert illum.get_power() == pytest.approx(0.0)
    assert illum.is_on() is False


def test_device_property_illumination_power_and_state_are_none_when_unreadable():
    illum = DevicePropertyIllumination(
        DeadCore(),
        device_name="LightEngine",
        state_property="State",
        intensity_property="Intensity",
    )
    assert illum.get_power() is None
    assert illum.is_on() is None


# --- Detector: gain 0.0 reads as "safe to expose to light" -----------------


def test_pmt_gain_is_none_when_unreadable():
    det = PMTDetector(DeadCore(), device_name="DCC100")
    assert det.get_gain() is None


def test_pmt_enabled_state_is_none_when_unreadable():
    det = PMTDetector(DeadCore(), device_name="DCC100")
    assert det.is_enabled() is None


def test_pmt_reports_a_real_gain():
    core = LiveCore({("DCC100", "Connector1GainHV_Percent"): "42.0"})
    det = PMTDetector(core, device_name="DCC100")
    assert det.get_gain() == pytest.approx(0.42)


def test_dcu_channel_gain_and_state_are_none_when_unreadable():
    det = DCUDetector(DeadCore(), device_name="DCU", channel=1)
    assert det.get_gain() is None
    assert det.is_enabled() is None


# --- Binning: 1 means "not binning", which corrupts pixel size -------------


def test_pycromanager_binning_is_none_when_unreadable():
    cam = PycromanagerCamera(DeadCore(), None, {"device": "Cam"})
    assert cam.get_binning() is None


def test_pycromanager_binning_parses_both_encodings():
    core = LiveCore({("Cam", "Binning"): "2x2"})
    cam = PycromanagerCamera(core, None, {"device": "Cam"})
    assert cam.get_binning() == 2
    core._values[("Cam", "Binning")] = "4"
    assert cam.get_binning() == 4


def test_base_camera_binning_of_one_is_a_true_statement_not_a_guess():
    # The base class also reports available_binnings == [1], so 1 is the only
    # possible value. This is the line between a fabricated default and a
    # defensible one, and it is worth pinning so a future audit does not
    # "fix" it into None.
    cam = _BareCamera()
    assert cam.get_available_binnings() == [1]
    assert cam.get_binning() == 1


# --- Unified gain: 1.0 means "no amplification" ---------------------------


def test_jai_unified_gain_is_none_when_unreadable():
    props = JAICameraProperties(DeadCore())
    assert props.get_unified_gain() is None


def test_jai_white_balance_mode_is_none_when_unreadable():
    # "Off" is the claim that no white balance is being applied, and
    # correction paths branch on exactly that.
    props = JAICameraProperties(DeadCore())
    assert props.get_white_balance_mode() is None


def test_base_camera_unified_gain_is_none_not_unity():
    # A camera with no unified-gain concept has no gain to report. Matches
    # get_gain_range(), which already returned None for the same reason.
    cam = _BareCamera()
    assert cam.get_unified_gain() is None
    assert cam.get_gain_range() is None


def test_unreadable_gain_does_not_pass_set_verification():
    """The subtle one: writing gain 1.0 used to "verify" against a fabricated
    1.0 read-back, so a completely unreadable camera reported success."""
    core = DeadCore()
    props = JAICameraProperties(core)
    props.set_unified_gain(1.0, max_retries=1)
    # The write is still attempted; what must not happen is a silent pass.
    # get_unified_gain() reporting None is what makes the failure visible.
    assert props.get_unified_gain() is None


# --- Pixel rate / dwell time: fabricated timing lands in metadata ----------


def _lsm(core):
    cfg = {
        "device": "OSc-LSM",
        "base_pixel_size_um": 0.5,
        "valid_resolutions": [256, 512],
    }
    return LaserScanningCamera(core, None, cfg)


def test_lsm_pixel_rate_is_none_when_unreadable():
    core = LiveCore({("OSc-LSM", "LSM-Resolution"): "512"})
    cam = _lsm(core)
    # Resolution reads fine; the rate property is missing.
    with pytest.raises(KeyError):
        core.get_property("OSc-LSM", "LSM-PixelRateHz")
    assert cam.get_pixel_rate_hz() is None


def test_lsm_dwell_time_is_none_rather_than_zero():
    core = LiveCore({("OSc-LSM", "LSM-Resolution"): "512"})
    cam = _lsm(core)
    assert cam.get_dwell_time_us() is None


def test_lsm_zero_rate_is_undefined_not_zero_dwell():
    core = LiveCore({("OSc-LSM", "LSM-Resolution"): "512", ("OSc-LSM", "LSM-PixelRateHz"): "0"})
    cam = _lsm(core)
    assert cam.get_dwell_time_us() is None


def test_lsm_exposure_raises_rather_than_reporting_a_made_up_zero():
    # get_exposure() is declared -> float on the ABC and every caller treats
    # it as a measurement, so the honest failure here is an exception.
    core = LiveCore({("OSc-LSM", "LSM-Resolution"): "512"})
    cam = _lsm(core)
    with pytest.raises(RuntimeError, match="pixel rate"):
        cam.get_exposure()


def test_lsm_reports_real_dwell_time():
    core = LiveCore(
        {("OSc-LSM", "LSM-Resolution"): "512", ("OSc-LSM", "LSM-PixelRateHz"): "250000"}
    )
    cam = _lsm(core)
    assert cam.get_dwell_time_us() == pytest.approx(4.0)
    assert cam.get_exposure() == pytest.approx(0.004)


# --- Config-supplied physical constants -----------------------------------


def test_lsm_refuses_to_construct_without_a_pixel_size():
    # Pixel size is a measured property of the optical path. The old code
    # substituted 1.0 um behind a warning that called itself "obviously
    # wrong" -- but a warning does not stop an acquisition, and the images
    # then carry a wrong scale that looks measured.
    core = LiveCore({("OSc-LSM", "LSM-Resolution"): "512"})
    with pytest.raises(ValueError, match="base_pixel_size_um"):
        LaserScanningCamera(core, None, {"device": "OSc-LSM", "valid_resolutions": [256, 512]})


def test_lsm_refuses_to_construct_when_resolution_is_unreadable():
    # Resolution drives the reshape of the pixel buffer in snap_image, so a
    # guessed value either throws later or silently misreads the image.
    with pytest.raises(RuntimeError, match="resolution"):
        LaserScanningCamera(DeadCore(), None, {"device": "OSc-LSM", "base_pixel_size_um": 0.5})
