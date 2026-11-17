"""The per_detector overlay: one config, each camera's values merged in by detector ID.

QPSC (MicroscopeConfigManager) and microscope_configurations/checks/config_checks.py
apply the same rules; a case that changes here must change there too.
"""

from copy import deepcopy

import yaml

from microscope_control.config import ConfigManager, apply_detector_overlay

ORYX = "LCPS_DETECTOR_FLIR_ORYX_001"
HAM = "LCPS_DETECTOR_HAMAMATSU_FLASH_001"


def test_scalar_is_replaced():
    cfg = {"reconstruction": {"black_level": 51, "per_detector": {ORYX: {"black_level": 99}}}}
    out = apply_detector_overlay(cfg, ORYX)
    assert out["reconstruction"]["black_level"] == 99
    # The per_detector block stays where it was.
    assert out["reconstruction"]["per_detector"] == {ORYX: {"black_level": 99}}


def test_nested_mapping_is_merged_not_replaced():
    cfg = {
        "objective": {
            "mm_preset": {"group": "Objective", "preset": "20x", "motorized": False},
            "per_detector": {ORYX: {"mm_preset": {"group": "Objective Lens"}}},
        }
    }
    out = apply_detector_overlay(cfg, ORYX)
    assert out["objective"]["mm_preset"] == {
        "group": "Objective Lens",
        "preset": "20x",
        "motorized": False,
    }


def test_device_properties_replace_by_device_and_property_then_append():
    cfg = {
        "device_properties": [
            {"device": "LC", "property": "Wavelength", "value": 549.0},
            {"device": "LC", "property": "A", "value": 0.25},
            {"device": "LC", "property": "B", "value": 0.50},
        ],
        "per_detector": {
            ORYX: {
                "device_properties": [
                    {"device": "LC", "property": "B", "value": 0.43},
                    {"device": "Shutter", "property": "State", "value": 1},
                    {"device": "LC", "property": "A", "value": 0.22},
                ]
            }
        },
    }
    out = apply_detector_overlay(cfg, ORYX)
    assert out["device_properties"] == [
        {"device": "LC", "property": "Wavelength", "value": 549.0},
        {"device": "LC", "property": "A", "value": 0.22},
        {"device": "LC", "property": "B", "value": 0.43},
        {"device": "Shutter", "property": "State", "value": 1},
    ]


def test_each_channel_in_a_list_uses_its_own_per_detector():
    cfg = {
        "channels": [
            {"id": "State0", "exposure_ms": 141.5, "per_detector": {ORYX: {"exposure_ms": 30}}},
            {"id": "State1", "exposure_ms": 141.5, "per_detector": {ORYX: {"exposure_ms": 31}}},
            {"id": "State2", "exposure_ms": 141.5},
        ]
    }
    out = apply_detector_overlay(cfg, ORYX)
    assert [c["exposure_ms"] for c in out["channels"]] == [30, 31, 141.5]


def test_merge_continues_into_the_merged_values():
    # A per_detector entry inside a mapping that an outer per_detector just merged into.
    cfg = {
        "a": {
            "per_detector": {ORYX: {"b": {"x": 1}}},
            "b": {"x": 0, "c": {"y": 0, "per_detector": {ORYX: {"y": 2}}}},
        }
    }
    out = apply_detector_overlay(cfg, ORYX)
    assert out["a"]["b"]["x"] == 1
    assert out["a"]["b"]["c"]["y"] == 2


def test_absent_detector_leaves_the_config_unchanged():
    cfg = {"reconstruction": {"black_level": 51, "per_detector": {ORYX: {"black_level": 99}}}}
    assert apply_detector_overlay(cfg, HAM) == cfg
    assert apply_detector_overlay(cfg, None) == cfg


def test_non_mapping_entries_are_ignored():
    cfg = {"x": 1, "per_detector": {ORYX: None, HAM: [1, 2]}}
    assert apply_detector_overlay(cfg, ORYX) == cfg
    assert apply_detector_overlay(cfg, HAM) == cfg
    cfg = {"x": 1, "per_detector": [ORYX]}
    assert apply_detector_overlay(cfg, ORYX) == cfg


def test_input_is_not_modified():
    cfg = {
        "channels": [
            {
                "exposure_ms": 141.5,
                "device_properties": [{"device": "LC", "property": "A", "value": 0.25}],
                "per_detector": {
                    ORYX: {
                        "exposure_ms": 30,
                        "device_properties": [{"device": "LC", "property": "A", "value": 0.2}],
                    }
                },
            }
        ]
    }
    before = deepcopy(cfg)
    out = apply_detector_overlay(cfg, ORYX)
    assert cfg == before
    # Nor does the result share mutable state with the overlay source.
    out["channels"][0]["device_properties"][0]["value"] = 9
    assert cfg["channels"][0]["per_detector"][ORYX]["device_properties"][0]["value"] == 0.2


def test_config_manager_applies_the_active_detector(tmp_path):
    cfg = {
        "microscope": {"name": "X"},
        "modalities": {
            "lcpolscope": {
                "reconstruction": {
                    "black_level": 51,
                    "per_detector": {ORYX: {"black_level": 99}, HAM: {"black_level": 51}},
                }
            }
        },
    }
    path = tmp_path / "config_X.yml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    manager = ConfigManager(str(tmp_path / "empty"))

    def black_level(data):
        return data["modalities"]["lcpolscope"]["reconstruction"]["black_level"]

    assert black_level(manager.load_config_file(str(path))) == 51
    manager.active_detector = ORYX
    assert black_level(manager.load_config_file(str(path))) == 99
    assert black_level(manager.load_config_file(str(path), apply_detector_values=False)) == 51
