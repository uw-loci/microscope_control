"""
Configuration Manager for QuPath Scope Control
Handles loading, saving, and accessing microscope configurations
without dataclass dependencies.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages microscope configurations and acquisition profiles.
    Works directly with YAML configuration files without dataclass conversions.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize ConfigManager with configuration directory.

        Args:
            config_dir: Path to configuration directory. If None, uses 'configurations'
                       subdirectory relative to this file.
        """
        if config_dir is None:
            # configurations folder is at smart_wsi_scanner/configurations/, not config/configurations/
            package_dir = Path(__file__).parent.parent  # smart_wsi_scanner/
            self.config_dir = package_dir / "configurations"
        else:
            self.config_dir = Path(config_dir)

        self._configs: Dict[str, Dict[str, Any]] = {}
        self._current_config_name: Optional[str] = None
        self._load_configs()
        logger.info(f"ConfigManager initialized with directory: {self.config_dir}")

    def _load_configs(self) -> None:
        """Load all configuration files from config directory."""
        if not self.config_dir.exists():
            logger.warning(f"Configuration directory not found: {self.config_dir}")
            self.config_dir.mkdir(parents=True, exist_ok=True)
            return

        for file in self.config_dir.glob("*.yml"):
            try:
                config_name = file.stem
                self._configs[config_name] = self.load_config_file(str(file))
                logger.info(f"Loaded configuration: {config_name}")
            except Exception as e:
                logger.error(f"Failed to load config {file}: {e}")

    def load_config_file(self, config_path: str) -> Dict[str, Any]:
        """
        Load a single configuration file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Dictionary containing configuration data
        """
        with open(config_path, "r") as file:
            data = yaml.safe_load(file)
        return data

    def get_config(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get configuration by name or return current config.

        Args:
            name: Configuration name. If None, returns current config.

        Returns:
            Configuration dictionary or None if not found
        """
        if name is None:
            name = self._current_config_name
        if name is None:
            return None
        return deepcopy(self._configs.get(name))

    def set_current_config(self, name: str) -> bool:
        """
        Set the current active configuration.

        Args:
            name: Name of configuration to set as current

        Returns:
            True if successful, False if config not found
        """
        if name in self._configs:
            self._current_config_name = name
            logger.info(f"Current config set to: {name}")
            return True
        logger.error(f"Configuration not found: {name}")
        return False

    def save_config(self, name: str, config: Dict[str, Any]) -> None:
        """
        Save configuration to file.

        Args:
            name: Name for the configuration
            config: Configuration dictionary to save
        """
        config_path = self.config_dir / f"{name}.yml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        self._configs[name] = deepcopy(config)
        logger.info(f"Saved configuration: {name}")

    def list_configs(self) -> List[str]:
        """List all available configurations."""
        return list(self._configs.keys())

    # Convenience methods for accessing common configuration elements

    def get_microscope_info(self, config_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get microscope information from config."""
        config = self.get_config(config_name)
        return config.get("microscope") if config else None

    def get_modalities(self, config_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get available imaging modalities."""
        config = self.get_config(config_name)
        return config.get("modalities") if config else None

    def get_acquisition_profile(
        self, modality: str, objective: str, detector: str, config_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get acquisition profile for specific combination.

        Args:
            modality: Imaging modality (e.g., 'ppm', 'brightfield')
            objective: Objective lens identifier
            detector: Detector identifier
            config_name: Configuration name (uses current if None)

        Returns:
            Acquisition profile with merged defaults and specific settings
        """
        config = self.get_config(config_name)
        if not config:
            return None

        acq_profiles = config.get("acq_profiles", {})
        defaults = acq_profiles.get("defaults", [])
        profiles = acq_profiles.get("profiles", [])

        # Find default settings for objective
        default_settings = {}
        for default in defaults:
            if default.get("objective") == objective:
                default_settings = deepcopy(default.get("settings", {}))
                break

        # Find specific profile
        for profile in profiles:
            if (
                profile.get("modality") == modality
                and profile.get("objective") == objective
                and profile.get("detector") == detector
            ):
                # NOTE: Settings (exposures, gains, white_balance) have been moved to imageprocessing config
                # Profiles now only contain modality/objective/detector identifiers
                # Return the profile identifiers without settings
                return {
                    "modality": modality,
                    "objective": objective,
                    "detector": detector,
                }

        return None

    def _merge_settings(self, defaults: Dict[str, Any], specific: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge default settings with specific settings.
        Specific settings override defaults.
        """
        result = deepcopy(defaults)
        for key, value in specific.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        return result

    def get_stage_limits(self, config_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get stage movement limits."""
        config = self.get_config(config_name)
        if not config:
            return None

        stage = config.get("stage", {})
        return stage.get("limits")

    def get_pixel_size(
        self, objective: str, detector: str, config_name: Optional[str] = None
    ) -> Optional[float]:
        """
        Get pixel size for objective/detector combination.

        Args:
            objective: Objective lens identifier
            detector: Detector identifier
            config_name: Configuration name (uses current if None)

        Returns:
            Pixel size in micrometers or None if not found
        """
        config = self.get_config(config_name)
        if not config:
            return None

        acq_profiles = config.get("acq_profiles", {})
        defaults = acq_profiles.get("defaults", [])

        for default in defaults:
            if default.get("objective") == objective:
                pixel_sizes = default.get("settings", {}).get("pixel_size_xy_um", {})
                return pixel_sizes.get(detector)

        return None

    def get_background_correction(
        self, modality: str, config_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get background correction settings for a modality.

        Background correction settings are now stored in imageprocessing_PPM.yml.

        Args:
            modality: Imaging modality
            config_name: Configuration name (uses current if None)

        Returns:
            Background correction settings or None
        """
        if config_name:
            # Derive imageprocessing config name from main config name
            # e.g., "config_PPM" -> "imageprocessing_PPM"
            imageprocessing_name = config_name.replace("config_", "imageprocessing_")
            imageprocessing_config = self.get_config(imageprocessing_name)

            if imageprocessing_config:
                bg_correction = imageprocessing_config.get("background_correction", {})
                return bg_correction.get(modality)

        return None

    def get_imaging_profile(
        self, modality: str, objective: str, detector: str, config_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get imaging profile settings (white_balance, exposures_ms, gains) from imageprocessing config.

        These settings were moved from acq_profiles to imageprocessing_PPM.yml.

        Args:
            modality: Imaging modality (e.g., 'ppm', 'brightfield')
            objective: Objective lens identifier
            detector: Detector identifier
            config_name: Configuration name (uses current if None)

        Returns:
            Dictionary with imaging settings or None if not found
        """
        if config_name:
            # Derive imageprocessing config name from main config name
            # e.g., "config_PPM" -> "imageprocessing_PPM"
            imageprocessing_name = config_name.replace("config_", "imageprocessing_")
            imageprocessing_config = self.get_config(imageprocessing_name)

            if imageprocessing_config:
                imaging_profiles = imageprocessing_config.get("imaging_profiles", {})
                modality_profiles = imaging_profiles.get(modality, {})
                objective_profiles = modality_profiles.get(objective, {})
                return objective_profiles.get(detector)

        return None

    def get_rotation_angles(
        self, config_name: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Get rotation angles for PPM modality."""
        config = self.get_config(config_name)
        if not config:
            return None

        ppm_config = config.get("modalities", {}).get("ppm", {})
        return ppm_config.get("rotation_angles")

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration structure and return list of errors.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check required top-level keys
        required_keys = ["microscope", "modalities", "acq_profiles", "stage"]
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")

        # Validate microscope section
        if "microscope" in config:
            microscope = config["microscope"]
            if not isinstance(microscope, dict):
                errors.append("'microscope' must be a dictionary")
            elif "name" not in microscope:
                errors.append("'microscope' must have a 'name' field")

        # Validate modalities
        if "modalities" in config:
            modalities = config["modalities"]
            if not isinstance(modalities, dict):
                errors.append("'modalities' must be a dictionary")

        # Validate acquisition profiles
        if "acq_profiles" in config:
            acq_profiles = config["acq_profiles"]
            if not isinstance(acq_profiles, dict):
                errors.append("'acq_profiles' must be a dictionary")
            else:
                if "defaults" in acq_profiles and not isinstance(acq_profiles["defaults"], list):
                    errors.append("'defaults' must be a list")
                if "profiles" in acq_profiles and not isinstance(acq_profiles["profiles"], list):
                    errors.append("'profiles' must be a list")

        return errors

    def create_empty_config(self, microscope_name: str, microscope_type: str) -> Dict[str, Any]:
        """
        Create an empty configuration template.

        Args:
            microscope_name: Name of the microscope
            microscope_type: Type of microscope

        Returns:
            Empty configuration dictionary
        """
        return {
            "microscope": {
                "name": microscope_name,
                "type": microscope_type,
                "detector_in_use": None,
                "objective_in_use": None,
                "modality": None,
            },
            "modalities": {},
            "acq_profiles": {"defaults": [], "profiles": []},
            "stage": {
                "stage_id": "",
                "limits": {
                    "x_um": {"low": 0, "high": 0},
                    "y_um": {"low": 0, "high": 0},
                    "z_um": {"low": 0, "high": 0},
                },
            },
            "slide_size_um": {"x": 0, "y": 0},
        }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Create config manager
    cm = ConfigManager()

    # List available configs
    print("Available configurations:", cm.list_configs())

    # Load and display a config
    if "config_PPM" in cm.list_configs():
        cm.set_current_config("config_PPM")

        # Get microscope info
        microscope = cm.get_microscope_info()
        print(f"\nMicroscope: {microscope}")

        # Get acquisition profile
        profile = cm.get_acquisition_profile(
            modality="ppm",
            objective="LOCI_OBJECTIVE_OLYMPUS_10X_001",
            detector="LOCI_DETECTOR_TELEDYNE_001",
        )
        print(f"\nAcquisition profile: {profile}")

        # Get pixel size
        pixel_size = cm.get_pixel_size(
            objective="LOCI_OBJECTIVE_OLYMPUS_10X_001", detector="LOCI_DETECTOR_TELEDYNE_001"
        )
        print(f"\nPixel size: {pixel_size} µm")

        # Get stage limits
        limits = cm.get_stage_limits()
        print(f"\nStage limits: {limits}")
