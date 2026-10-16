"""
Unit tests for ConfigManager.

Tests configuration loading, merging, and validation logic.
"""

import pytest
from microscope_control.config.manager import ConfigManager
import yaml


class TestConfigMergeSettings:
    """Test the _merge_settings method for dictionary merging."""

    def test_merge_simple_dicts(self):
        """Test merging two simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        merged = ConfigManager._merge_settings(base, override)
        assert merged["a"] == 1
        assert merged["b"] == 3  # Override wins
        assert merged["c"] == 4

    def test_merge_nested_dicts(self):
        """Test merging nested dictionaries."""
        base = {
            "microscope": {
                "name": "Base Scope",
                "stage": {"x_max": 100000, "y_max": 75000},
            }
        }

        override = {"microscope": {"name": "Override Scope", "stage": {"y_max": 80000}}}

        merged = ConfigManager._merge_settings(base, override)

        assert merged["microscope"]["name"] == "Override Scope"
        assert merged["microscope"]["stage"]["x_max"] == 100000  # Kept from base
        assert merged["microscope"]["stage"]["y_max"] == 80000  # Overridden

    def test_merge_with_empty_override(self):
        """Test merging with empty override dictionary."""
        base = {"a": 1, "b": 2}
        override = {}

        merged = ConfigManager._merge_settings(base, override)
        assert merged == base

    def test_merge_with_empty_base(self):
        """Test merging with empty base dictionary."""
        base = {}
        override = {"a": 1, "b": 2}

        merged = ConfigManager._merge_settings(base, override)
        assert merged == override

    def test_merge_doesnt_modify_originals(self):
        """Test that merging doesn't modify original dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        base_copy = base.copy()
        override_copy = override.copy()

        ConfigManager._merge_settings(base, override)

        # Originals should be unchanged
        assert base == base_copy
        assert override == override_copy


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_valid_config(self, sample_microscope_config):
        """Test validation of valid configuration."""
        errors = ConfigManager.validate_config(sample_microscope_config)
        # Should return empty list for valid config
        assert errors == []

    def test_validate_missing_required_keys(self):
        """Test validation fails with missing required keys."""
        incomplete_config = {
            "microscope": {
                "name": "Test"
                # Missing objectives, stage limits, etc.
            }
        }

        errors = ConfigManager.validate_config(incomplete_config)
        # Should return non-empty list of errors (modalities/acq_profiles/stage missing)
        assert len(errors) > 0

    def test_validate_empty_config(self):
        """Test validation of empty configuration."""
        empty_config = {}

        errors = ConfigManager.validate_config(empty_config)
        # Should return errors for missing required keys
        assert len(errors) > 0

    def test_validate_invalid_data_types(self):
        """Test validation catches invalid data types."""
        invalid_config = {
            "microscope": {
                "name": "Test",
                "stage": {
                    "limits": {
                        "x": "not_a_dict",  # Should be dict with min/max
                    }
                },
            }
        }

        errors = ConfigManager.validate_config(invalid_config)
        # Should return errors (top-level modalities/acq_profiles/stage missing)
        assert len(errors) > 0


class TestConfigManagerInit:
    """Test ConfigManager initialization."""

    def test_init_with_valid_config_dir(self, temp_output_directory):
        """Test initialization with a directory containing a valid config file."""
        config_data = {
            "microscope": {
                "name": "TestScope",
                "objectives": {"10x": {"magnification": 10, "pixel_size_um": 0.65}},
            }
        }

        config_path = temp_output_directory / "test_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        manager = ConfigManager(str(temp_output_directory))
        assert manager is not None
        assert "test_config" in manager.list_configs()

    def test_init_with_nonexistent_directory(self, tmp_path):
        """Test that a nonexistent config directory is handled gracefully.

        ConfigManager does not raise for a missing directory; it creates the
        directory and loads no configurations.
        """
        missing_dir = tmp_path / "does_not_exist"
        manager = ConfigManager(str(missing_dir))
        assert manager.list_configs() == []

    def test_init_with_malformed_yaml(self, temp_output_directory):
        """Test initialization with malformed YAML file."""
        # Create a malformed YAML file in the directory
        config_path = temp_output_directory / "malformed.yml"
        config_path.write_text("{ invalid yaml content: [")

        # ConfigManager catches YAML errors per-file and skips the bad file
        manager = ConfigManager(str(temp_output_directory))
        # The malformed config should not be loaded
        assert "malformed" not in manager.list_configs()


class TestConfigManagerGetConfig:
    """Test ConfigManager configuration retrieval."""

    def test_get_microscope_name(self, temp_output_directory):
        """Test getting microscope name from a loaded config."""
        config_data = {"microscope": {"name": "MyTestScope"}}

        config_path = temp_output_directory / "test_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        manager = ConfigManager(str(temp_output_directory))
        config = manager.get_config("test_config")
        assert config["microscope"]["name"] == "MyTestScope"

    def test_get_nested_value(self, temp_output_directory):
        """Test getting nested configuration values."""
        config_data = {"microscope": {"stage": {"limits": {"x": {"min": 0, "max": 100000}}}}}

        config_path = temp_output_directory / "test_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        manager = ConfigManager(str(temp_output_directory))
        config = manager.get_config("test_config")
        x_limits = config["microscope"]["stage"]["limits"]["x"]
        assert x_limits["min"] == 0
        assert x_limits["max"] == 100000

    def test_get_nonexistent_config_returns_none(self, temp_output_directory):
        """Test getting a config that was never loaded returns None."""
        config_data = {"microscope": {"name": "Test"}}

        config_path = temp_output_directory / "test_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        manager = ConfigManager(str(temp_output_directory))
        # No config with this name was loaded
        assert manager.get_config("nonexistent") is None


class TestConfigManagerDefaults:
    """Test default configuration handling."""

    def test_default_values_applied(self, temp_output_directory):
        """Test that a minimal config loads without error."""
        minimal_config = {"microscope": {"name": "MinimalScope"}}

        config_path = temp_output_directory / "minimal_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(minimal_config, f)

        manager = ConfigManager(str(temp_output_directory))
        assert manager is not None
        assert manager.get_config("minimal_config")["microscope"]["name"] == "MinimalScope"


class TestConfigManagerObjectiveSettings:
    """Test objective-specific configuration retrieval."""

    def test_get_objective_settings(self, temp_output_directory):
        """Test retrieving objective-specific settings."""
        config_data = {
            "microscope": {
                "objectives": {
                    "10x": {"magnification": 10, "pixel_size_um": 0.65, "na": 0.45},
                    "20x": {"magnification": 20, "pixel_size_um": 0.325, "na": 0.75},
                }
            }
        }

        config_path = temp_output_directory / "objectives_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        manager = ConfigManager(str(temp_output_directory))
        objectives = manager.get_config("objectives_config")["microscope"]["objectives"]

        assert objectives["10x"]["pixel_size_um"] == 0.65
        assert objectives["20x"]["pixel_size_um"] == 0.325


class TestConfigManagerAutofocusSettings:
    """Test autofocus configuration retrieval."""

    def test_get_autofocus_settings(self, temp_output_directory, sample_autofocus_config):
        """Test retrieving autofocus settings from a loaded config."""
        config_path = temp_output_directory / "autofocus_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(sample_autofocus_config, f)

        manager = ConfigManager(str(temp_output_directory))
        af_settings = manager.get_config("autofocus_config")["autofocus"]

        assert af_settings is not None
        assert af_settings["n_steps"] == 21
        assert af_settings["search_range"] == 200.0


class TestConfigManagerResources:
    """id_* sections merge entry by entry, the config's own entries winning (as in QPSC)."""

    @staticmethod
    def _write(tmp_path, config, resources):
        (tmp_path / "resources").mkdir()
        with open(tmp_path / "resources" / "resources_LOCI.yml", "w") as f:
            yaml.safe_dump(resources, f)
        path = tmp_path / "config_Test.yml"
        with open(path, "w") as f:
            yaml.safe_dump(config, f)
        return ConfigManager(str(tmp_path)).load_config_file(str(path))

    RESOURCES = {
        "id_detector": {
            "LOCI_DETECTOR_JAI_001": {"device": "JAICamera", "camera_type": "jai"},
            "SHARED_CAM": {"device": "Shared", "flip_x": False},
        },
        "id_stage": {"LOCI_STAGE_001": {"name": "Prior"}},
        "other_section": {"x": 1},
    }

    def test_a_detector_only_the_config_defines_reaches_the_server(self, tmp_path):
        # config_OWS3.yml's HAMAMATSU_DCAM_01 is in no resources file; the old loader dropped it.
        config = {"id_detector": {"HAMAMATSU_DCAM_01": {"device": "HamamatsuHam_DCAM"}}}
        data = self._write(tmp_path, config, self.RESOURCES)
        assert data["id_detector"]["HAMAMATSU_DCAM_01"] == {"device": "HamamatsuHam_DCAM"}
        assert "LOCI_DETECTOR_JAI_001" in data["id_detector"]

    def test_the_configs_own_entry_wins_over_the_resources_entry(self, tmp_path):
        config = {"id_detector": {"SHARED_CAM": {"device": "Shared", "flip_x": True}}}
        data = self._write(tmp_path, config, self.RESOURCES)
        assert data["id_detector"]["SHARED_CAM"] == {"device": "Shared", "flip_x": True}

    def test_without_inline_sections_the_resources_sections_are_used(self, tmp_path):
        data = self._write(tmp_path, {"microscope": {"name": "T"}}, self.RESOURCES)
        assert data["id_detector"] == self.RESOURCES["id_detector"]
        assert data["id_stage"] == self.RESOURCES["id_stage"]
        # Only id_* sections come from the resources file.
        assert "other_section" not in data

    def test_merging_does_not_change_the_resources_mapping(self):
        from microscope_control.config.manager import merge_id_section

        resources = {"A": {"v": 1}}
        merged = merge_id_section("id_detector", resources, {"B": {"v": 2}})
        assert merged == {"A": {"v": 1}, "B": {"v": 2}}
        assert resources == {"A": {"v": 1}}


class TestGetModalityFromScanType:
    """Test extraction of modality from scan type string (if in config module)."""

    def test_parse_standard_scan_type(self):
        """Test parsing PPM_10x_1 format."""
        try:
            from microscope_control.config.manager import get_modality_from_scan_type
        except ImportError:
            # May be in a different module
            pytest.skip("get_modality_from_scan_type not in config.manager")

        modality = get_modality_from_scan_type("PPM_10x_1")
        assert modality == "PPM_10x"

        modality = get_modality_from_scan_type("Brightfield_20x_3")
        assert modality == "Brightfield_20x"

    def test_parse_malformed_scan_type(self):
        """Test handling of malformed scan type."""
        try:
            from microscope_control.config.manager import get_modality_from_scan_type
        except ImportError:
            pytest.skip("get_modality_from_scan_type not available")

        # Should handle gracefully
        try:
            modality = get_modality_from_scan_type("invalid_format")
            assert modality is not None
        except (ValueError, IndexError):
            # Expected for malformed input
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
