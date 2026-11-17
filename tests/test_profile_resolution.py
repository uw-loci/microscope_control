"""Picking the acquisition profile for a modality, objective and camera.

QPSC's MicroscopeConfigManager.resolveProfileKey applies the same rule; a case that
changes here must change there too.
"""

import logging
from types import SimpleNamespace

from microscope_control.config import resolve_profile_key, sibling_for_detector
from microscope_control.hardware.pycromanager import PycromanagerHardware

ORYX = "LCPS_DETECTOR_FLIR_ORYX_001"
HAM = "LCPS_DETECTOR_HAMAMATSU_FLASH_001"
OBJ20 = "LCPS_OBJECTIVE_20X_POL_001"
OBJ10 = "LCPS_OBJECTIVE_10X_001"

# The LC-PolScope layout: one profile set per camera, the Oryx keys suffixed.
LC_PROFILES = {
    "lcpolscope_20x": {"modality": "lcpolscope", "detector": HAM, "objective": OBJ20},
    "brightfield_20x": {"modality": "brightfield", "detector": HAM, "objective": OBJ20},
    "lcpolscope_10x": {"modality": "lcpolscope", "detector": HAM, "objective": OBJ10},
    "lcpolscope_20x_oryx": {"modality": "lcpolscope", "detector": ORYX, "objective": OBJ20},
    "brightfield_20x_oryx": {"modality": "brightfield", "detector": ORYX, "objective": OBJ20},
    "lcpolscope_10x_oryx": {"modality": "lcpolscope", "detector": ORYX, "objective": OBJ10},
}


def test_detector_picks_its_own_profile_set():
    assert resolve_profile_key(LC_PROFILES, "lcpolscope", OBJ20, HAM) == "lcpolscope_20x"
    assert resolve_profile_key(LC_PROFILES, "lcpolscope", OBJ20, ORYX) == "lcpolscope_20x_oryx"
    assert resolve_profile_key(LC_PROFILES, "lcpolscope", OBJ10, ORYX) == "lcpolscope_10x_oryx"


def test_modality_matches_ignoring_case_objective_exactly():
    assert resolve_profile_key(LC_PROFILES, "BrightField", OBJ20, ORYX) == "brightfield_20x_oryx"
    # The objective field is compared exactly; a different case falls to the name rules.
    assert resolve_profile_key(LC_PROFILES, "brightfield", OBJ20.lower(), ORYX) is None


def test_no_detector_takes_the_first_in_file_order():
    assert resolve_profile_key(LC_PROFILES, "lcpolscope", OBJ20, None) == "lcpolscope_20x"


def test_profiles_without_a_detector_suit_every_camera():
    profiles = {
        "bf_any": {"modality": "brightfield", "objective": OBJ20},
        "bf_null": {"modality": "brightfield", "objective": OBJ10, "detector": None},
        "bf_ham": {"modality": "brightfield", "objective": OBJ10, "detector": HAM},
    }
    assert resolve_profile_key(profiles, "brightfield", OBJ20, ORYX) == "bf_any"
    assert resolve_profile_key(profiles, "brightfield", OBJ10, ORYX) == "bf_null"
    # An empty string is a camera name, one that matches no detector.
    profiles["bf_null"]["detector"] = ""
    assert resolve_profile_key(profiles, "brightfield", OBJ10, ORYX) is None


def test_name_rules_run_over_the_candidates_only():
    # Keyed by name only, no modality/objective fields (the older PPM / OWS3 shape).
    profiles = {
        "Brightfield_10x": {"detector": HAM},
        "Brightfield_10x_oryx": {"detector": ORYX},
        "Brightfield": {"detector": HAM},
    }
    # Exact modality key, but it belongs to the other camera: skipped.
    assert resolve_profile_key(profiles, "Brightfield", "0.5NA_AIR_10x", ORYX) is None
    assert resolve_profile_key(profiles, "Brightfield", "0.5NA_AIR_10x", HAM) == "Brightfield"


def test_name_rules_longest_suffix_and_first_on_tie():
    profiles = {"bf_x": {}, "bf_10x": {}, "bf_0x": {}, "bf_ab": {}, "bf_aa": {}}
    assert resolve_profile_key(profiles, "bf", "OLY_10X_001") == "bf_10x"
    # Two suffixes of the same length both match: the first in file order wins.
    assert resolve_profile_key(profiles, "bf", "obj_ab_aa") == "bf_ab"


def test_name_rules_case_insensitive_prefix_fallback():
    profiles = {"brightfield_10x": {}}
    assert resolve_profile_key(profiles, "Brightfield", "0.5NA_AIR_10x") == "brightfield_10x"


def test_nothing_to_match():
    assert resolve_profile_key({}, "bf", "x") is None
    assert resolve_profile_key(None, "bf", "x") is None
    assert resolve_profile_key(LC_PROFILES, "", OBJ20) is None
    assert resolve_profile_key(LC_PROFILES, "shg", OBJ20, HAM) is None


def test_sibling_for_detector():
    assert sibling_for_detector(LC_PROFILES, "lcpolscope_20x", ORYX) == "lcpolscope_20x_oryx"
    assert sibling_for_detector(LC_PROFILES, "lcpolscope_20x_oryx", HAM) == "lcpolscope_20x"
    assert sibling_for_detector(LC_PROFILES, "lcpolscope_20x", HAM) == "lcpolscope_20x"
    assert sibling_for_detector(LC_PROFILES, "lcpolscope_20x", None) == "lcpolscope_20x"
    assert sibling_for_detector(LC_PROFILES, "lcpolscope_20x", "OTHER") == "lcpolscope_20x"
    # Different modalities on different cameras (OWS3) are not siblings.
    ows3 = {
        "BF_10x": {"modality": "brightfield", "objective": "o10", "detector": "CAM"},
        "LSM_10x": {"modality": "shg", "objective": "o10", "detector": "LSM"},
    }
    assert sibling_for_detector(ows3, "BF_10x", "LSM") == "BF_10x"


# ---------------------------------------------------------------- apply_mode_setup


class _Core:
    def __init__(self, loaded):
        self.loaded = set(loaded)
        self.camera = None

    def set_property(self, device, prop, value):
        if device == "Core" and prop == "Camera" and value not in self.loaded:
            raise RuntimeError(f'No device with label "{value}"')
        self.camera = value


class _Cam:
    flip_x = flip_y = False

    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


def _hardware(profiles, active, loaded):
    hw = PycromanagerHardware.__new__(PycromanagerHardware)
    hw.core = _Core(loaded)
    hw.settings = {"acquisition_profiles": profiles}
    hw._camera_registry = {ORYX: _Cam("Oryx"), HAM: _Cam("Flash4")}
    hw._active_detector_id = active
    hw._camera_name = {ORYX: "Oryx", HAM: "Flash4"}[active]
    hw._illumination = None
    hw._stage = None
    hw._active_profile = None
    hw.config_manager = None
    hw._safe_disable_pmt_and_shutters = lambda: None
    hw._disable_all_modality_illuminations = lambda: None
    hw.get_illumination_for_modality = lambda name: None
    return hw


def test_mode_setup_uses_the_active_cameras_profile(caplog):
    hw = _hardware(LC_PROFILES, ORYX, loaded={"Oryx"})
    with caplog.at_level(logging.INFO):
        hw.apply_mode_setup("lcpolscope_20x_3")
    assert hw._active_profile == "lcpolscope_20x_oryx"
    assert hw._active_detector_id == ORYX
    assert hw.core.camera is None  # no camera switch was attempted


def test_mode_setup_continues_when_the_camera_is_not_loaded(caplog):
    # A config with a single profile for the other camera, not loaded in Micro-Manager.
    profiles = {"lcpolscope_20x": LC_PROFILES["lcpolscope_20x"]}
    hw = _hardware(profiles, ORYX, loaded={"Oryx"})
    with caplog.at_level(logging.WARNING):
        hw.apply_mode_setup("lcpolscope_20x")
    assert hw._active_profile == "lcpolscope_20x"
    assert hw._active_detector_id == ORYX
    assert any("Continuing with the active camera" in r.getMessage() for r in caplog.records)


def test_set_active_camera_tells_the_config_manager():
    hw = _hardware(LC_PROFILES, ORYX, loaded={"Oryx", "Flash4"})
    hw.config_manager = SimpleNamespace(active_detector=ORYX)
    hw.set_active_camera(HAM)
    assert hw._active_detector_id == HAM
    assert hw.config_manager.active_detector == HAM


def test_refresh_active_detector_reads_the_core_camera():
    hw = _hardware(LC_PROFILES, ORYX, loaded={"Oryx", "Flash4"})
    hw.settings["id_detector"] = {ORYX: {"device": "Oryx"}, HAM: {"device": "Flash4"}}
    hw.core.get_camera_device = lambda: "Flash4"
    assert hw.refresh_active_detector() == HAM
    assert hw._camera_name == "Flash4"

    def broken():
        raise RuntimeError("bridge down")

    hw.core.get_camera_device = broken
    assert hw.refresh_active_detector() == HAM
