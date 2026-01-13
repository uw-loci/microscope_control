"""
JAI Camera Property Management for AP-3200T-USB.

This module provides type-safe access to JAI camera properties exposed through
Micro-Manager, specifically for the 3-CCD prism detector with individual
RGB channel control.

Properties Supported (from Micro-Manager PR #781):
- Per-channel exposure control (ExposureIsIndividual mode)
- Per-channel analog and digital gain
- Per-channel black level adjustment

Usage:
    from microscope_control.jai import JAICameraProperties

    props = JAICameraProperties(core)
    props.enable_individual_exposure()
    props.set_channel_exposures(red=100.0, green=120.0, blue=140.0)

Note:
    This module requires the JAI camera device adapter from Micro-Manager
    with support for individual channel control (PR #781 or later).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PropertyLimits:
    """Limits for a numeric property."""
    min_value: float
    max_value: float
    current_value: float


class JAICameraProperties:
    """
    Manages JAI AP-3200T-USB camera properties via Micro-Manager.

    This class provides type-safe access to the per-channel exposure, gain,
    and black level properties added in Micro-Manager PR #781 for JAI prism cameras.
    """

    # Device name for the JAI camera
    DEVICE_NAME = "JAICamera"

    # Property name constants - Exposure
    EXPOSURE_INDIVIDUAL = "ExposureIsIndividual"
    EXPOSURE_RED = "Exposure_Red"
    EXPOSURE_GREEN = "Exposure_Green"
    EXPOSURE_BLUE = "Exposure_Blue"

    # Property name constants - Gain
    GAIN_INDIVIDUAL = "GainIsIndividual"
    GAIN_ANALOG_RED = "Gain_AnalogRed"
    GAIN_ANALOG_GREEN = "Gain_AnalogGreen"
    GAIN_ANALOG_BLUE = "Gain_AnalogBlue"
    GAIN_DIGITAL_RED = "Gain_DigitalRed"
    GAIN_DIGITAL_BLUE = "Gain_DigitalBlue"
    # Note: No Gain_DigitalGreen per PR #781

    # Property name constants - Black Level
    BLACK_LEVEL_ALL = "BlackLevel_DigitalAll"
    BLACK_LEVEL_RED = "BlackLevel_DigitalRed"
    BLACK_LEVEL_BLUE = "BlackLevel_DigitalBlue"
    # Note: No BlackLevel_DigitalGreen per PR #781

    # Frame rate property (for exposure time adjustment)
    FRAME_RATE = "FrameRateHz"

    # Mode values
    MODE_ON = "On"
    MODE_OFF = "Off"

    # Hardware limits (discovered 2026-01-13 from JAI AP-3200T-USB)
    # Per-channel exposure limit depends on frame rate:
    #   At 38 Hz -> max ~25.85ms
    #   At 5 Hz -> max ~200ms
    #   At 0.125 Hz -> max ~7900ms
    # The _adjust_frame_rate_for_exposure() method handles this automatically.
    EXPOSURE_MIN_MS = 0.001
    FRAME_RATE_MIN = 0.125
    FRAME_RATE_MAX = 39.21

    # Analog gain ranges per channel (vary by channel!)
    GAIN_ANALOG_RED_RANGE = (0.47, 4.0)
    GAIN_ANALOG_GREEN_RANGE = (1.0, 64.0)  # Green has wider range
    GAIN_ANALOG_BLUE_RANGE = (0.47, 4.0)

    # Digital gain range (narrow)
    GAIN_DIGITAL_RANGE = (0.9, 1.1)

    # Black level ranges
    BLACK_LEVEL_ALL_RANGE = (-133, 255)
    BLACK_LEVEL_CHANNEL_RANGE = (-64, 64)

    # Built-in white balance property and options
    WHITE_BALANCE = "WhiteBalance"
    WHITE_BALANCE_OPTIONS = [
        "Off", "Continuous", "Once",
        "Preset3200K", "Preset5000K", "Preset6500K", "Preset7500K"
    ]

    def __init__(self, core: Any, device_name: Optional[str] = None):
        """
        Initialize JAI camera property manager.

        Args:
            core: Pycromanager Core instance
            device_name: Override default device name (for testing)
        """
        self.core = core
        self.device_name = device_name or self.DEVICE_NAME
        self._property_cache: Dict[str, PropertyLimits] = {}

    def validate_camera(self) -> bool:
        """
        Verify that JAI camera is available and active.

        Returns:
            True if JAI camera is available, False otherwise
        """
        try:
            active_camera = self.core.get_property("Core", "Camera")
            if active_camera != self.device_name:
                logger.warning(
                    f"JAI camera not active. Current camera: {active_camera}, "
                    f"expected: {self.device_name}"
                )
                return False
            return True
        except Exception as e:
            logger.error(f"Failed to validate JAI camera: {e}")
            return False

    def _property_exists(self, property_name: str) -> bool:
        """Check if a property exists on the device."""
        try:
            self.core.get_property(self.device_name, property_name)
            return True
        except Exception:
            return False

    def _get_property(self, property_name: str) -> str:
        """
        Get a property value from the camera.

        Args:
            property_name: Name of the property

        Returns:
            Property value as string

        Raises:
            RuntimeError: If property cannot be read
        """
        try:
            return self.core.get_property(self.device_name, property_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to get property '{property_name}' from {self.device_name}: {e}"
            )

    def _set_property(self, property_name: str, value: Any) -> None:
        """
        Set a property value on the camera.

        Args:
            property_name: Name of the property
            value: Value to set

        Raises:
            RuntimeError: If property cannot be set
        """
        try:
            self.core.set_property(self.device_name, property_name, str(value))
            self.core.wait_for_device(self.device_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to set property '{property_name}' to '{value}' on "
                f"{self.device_name}: {e}"
            )

    def get_property_limits(self, property_name: str) -> Optional[PropertyLimits]:
        """
        Get the limits for a numeric property.

        Args:
            property_name: Name of the property

        Returns:
            PropertyLimits with min, max, and current values, or None if not available
        """
        if property_name in self._property_cache:
            return self._property_cache[property_name]

        try:
            has_limits = self.core.has_property_limits(self.device_name, property_name)
            if not has_limits:
                return None

            limits = PropertyLimits(
                min_value=self.core.get_property_lower_limit(
                    self.device_name, property_name
                ),
                max_value=self.core.get_property_upper_limit(
                    self.device_name, property_name
                ),
                current_value=float(self._get_property(property_name)),
            )
            self._property_cache[property_name] = limits
            return limits
        except Exception as e:
            logger.warning(f"Failed to get limits for {property_name}: {e}")
            return None

    # ========== Exposure Mode Control ==========

    def is_individual_exposure_enabled(self) -> bool:
        """Check if individual exposure mode is enabled."""
        try:
            value = self._get_property(self.EXPOSURE_INDIVIDUAL)
            return value == self.MODE_ON
        except Exception:
            return False

    def enable_individual_exposure(self) -> None:
        """Enable per-channel exposure control mode."""
        if not self._property_exists(self.EXPOSURE_INDIVIDUAL):
            raise RuntimeError(
                f"Property '{self.EXPOSURE_INDIVIDUAL}' not available. "
                "Ensure JAI camera driver supports individual exposure mode (PR #781)."
            )
        self._set_property(self.EXPOSURE_INDIVIDUAL, self.MODE_ON)
        logger.info("Enabled individual exposure mode")

    def disable_individual_exposure(self) -> None:
        """Return to unified exposure control mode."""
        if self._property_exists(self.EXPOSURE_INDIVIDUAL):
            self._set_property(self.EXPOSURE_INDIVIDUAL, self.MODE_OFF)
            logger.info("Disabled individual exposure mode")

    def set_channel_exposures(
        self,
        red: float,
        green: float,
        blue: float,
        auto_enable: bool = True,
    ) -> None:
        """
        Set per-channel exposure times in milliseconds.

        Args:
            red: Red channel exposure in ms
            green: Green channel exposure in ms
            blue: Blue channel exposure in ms
            auto_enable: Automatically enable individual exposure mode if not already

        Raises:
            RuntimeError: If individual exposure mode not available or setting fails
        """
        if auto_enable and not self.is_individual_exposure_enabled():
            self.enable_individual_exposure()

        # Clamp minimum (max is handled by frame rate adjustment)
        red = max(self.EXPOSURE_MIN_MS, red)
        green = max(self.EXPOSURE_MIN_MS, green)
        blue = max(self.EXPOSURE_MIN_MS, blue)

        # Adjust frame rate for longest exposure
        max_exposure = max(red, green, blue)
        self._adjust_frame_rate_for_exposure(max_exposure)

        # Set per-channel exposures
        self._set_property(self.EXPOSURE_RED, red)
        self._set_property(self.EXPOSURE_GREEN, green)
        self._set_property(self.EXPOSURE_BLUE, blue)

        logger.info(f"Set channel exposures: R={red:.2f}ms, G={green:.2f}ms, B={blue:.2f}ms")

    def get_channel_exposures(self) -> Dict[str, float]:
        """
        Get current per-channel exposure times.

        Returns:
            Dictionary with 'red', 'green', 'blue' keys and exposure values in ms
        """
        return {
            "red": float(self._get_property(self.EXPOSURE_RED)),
            "green": float(self._get_property(self.EXPOSURE_GREEN)),
            "blue": float(self._get_property(self.EXPOSURE_BLUE)),
        }

    def _adjust_frame_rate_for_exposure(self, exposure_ms: float) -> None:
        """
        Adjust frame rate to accommodate exposure time.

        JAI camera requires frame rate adjustment when exposure changes.

        Args:
            exposure_ms: Exposure time in milliseconds
        """
        frame_rate_min = 0.125
        frame_rate_max = 38.0
        margin = 1.01

        exposure_s = exposure_ms / 1000.0
        required_frame_rate = round(1.0 / (exposure_s * margin), 3)
        frame_rate = min(max(required_frame_rate, frame_rate_min), frame_rate_max)

        self._set_property(self.FRAME_RATE, frame_rate)
        logger.debug(f"Adjusted frame rate to {frame_rate} Hz for {exposure_ms}ms exposure")

    # ========== Gain Mode Control ==========

    def is_individual_gain_enabled(self) -> bool:
        """Check if individual gain mode is enabled."""
        try:
            value = self._get_property(self.GAIN_INDIVIDUAL)
            return value == self.MODE_ON
        except Exception:
            return False

    def enable_individual_gain(self) -> None:
        """Enable per-channel gain control mode."""
        if not self._property_exists(self.GAIN_INDIVIDUAL):
            raise RuntimeError(
                f"Property '{self.GAIN_INDIVIDUAL}' not available. "
                "Ensure JAI camera driver supports individual gain mode (PR #781)."
            )
        self._set_property(self.GAIN_INDIVIDUAL, self.MODE_ON)
        logger.info("Enabled individual gain mode")

    def disable_individual_gain(self) -> None:
        """Return to unified gain control mode."""
        if self._property_exists(self.GAIN_INDIVIDUAL):
            self._set_property(self.GAIN_INDIVIDUAL, self.MODE_OFF)
            logger.info("Disabled individual gain mode")

    def set_analog_gains(
        self,
        red: float,
        green: float,
        blue: float,
        auto_enable: bool = True,
    ) -> None:
        """
        Set per-channel analog gain values.

        Args:
            red: Red channel analog gain
            green: Green channel analog gain
            blue: Blue channel analog gain
            auto_enable: Automatically enable individual gain mode if not already
        """
        if auto_enable and not self.is_individual_gain_enabled():
            self.enable_individual_gain()

        self._set_property(self.GAIN_ANALOG_RED, red)
        self._set_property(self.GAIN_ANALOG_GREEN, green)
        self._set_property(self.GAIN_ANALOG_BLUE, blue)

        logger.info(f"Set analog gains: R={red:.2f}, G={green:.2f}, B={blue:.2f}")

    def get_analog_gains(self) -> Dict[str, float]:
        """
        Get current per-channel analog gain values.

        Returns:
            Dictionary with 'red', 'green', 'blue' keys and gain values
        """
        return {
            "red": float(self._get_property(self.GAIN_ANALOG_RED)),
            "green": float(self._get_property(self.GAIN_ANALOG_GREEN)),
            "blue": float(self._get_property(self.GAIN_ANALOG_BLUE)),
        }

    def set_digital_gains(self, red: float, blue: float) -> None:
        """
        Set per-channel digital gain values.

        Note: Green digital gain is not available per PR #781.

        Args:
            red: Red channel digital gain
            blue: Blue channel digital gain
        """
        self._set_property(self.GAIN_DIGITAL_RED, red)
        self._set_property(self.GAIN_DIGITAL_BLUE, blue)

        logger.info(f"Set digital gains: R={red:.2f}, B={blue:.2f}")

    def get_digital_gains(self) -> Dict[str, float]:
        """
        Get current per-channel digital gain values.

        Returns:
            Dictionary with 'red', 'blue' keys (no green digital gain available)
        """
        return {
            "red": float(self._get_property(self.GAIN_DIGITAL_RED)),
            "blue": float(self._get_property(self.GAIN_DIGITAL_BLUE)),
        }

    # ========== Black Level Control ==========

    def set_black_level_all(self, value: float) -> None:
        """
        Set global digital black level for all channels.

        Args:
            value: Black level value
        """
        self._set_property(self.BLACK_LEVEL_ALL, value)
        logger.info(f"Set global black level to {value}")

    def set_black_levels(self, red: float, blue: float, all_channels: Optional[float] = None) -> None:
        """
        Set per-channel digital black levels.

        Note: Green black level is not separately configurable per PR #781.

        Args:
            red: Red channel black level
            blue: Blue channel black level
            all_channels: Optional global black level (affects all channels)
        """
        if all_channels is not None:
            self._set_property(self.BLACK_LEVEL_ALL, all_channels)

        self._set_property(self.BLACK_LEVEL_RED, red)
        self._set_property(self.BLACK_LEVEL_BLUE, blue)

        logger.info(f"Set black levels: R={red}, B={blue}, All={all_channels}")

    def get_black_levels(self) -> Dict[str, float]:
        """
        Get current black level values.

        Returns:
            Dictionary with 'all', 'red', 'blue' keys
        """
        return {
            "all": float(self._get_property(self.BLACK_LEVEL_ALL)),
            "red": float(self._get_property(self.BLACK_LEVEL_RED)),
            "blue": float(self._get_property(self.BLACK_LEVEL_BLUE)),
        }

    # ========== White Balance Settings Persistence ==========

    def apply_white_balance_settings(self, settings_path: str) -> bool:
        """
        Load and apply white balance settings from YAML file.

        Args:
            settings_path: Path to white_balance_settings.yml

        Returns:
            True if settings were applied successfully, False otherwise
        """
        settings_path = Path(settings_path)
        if not settings_path.exists():
            logger.warning(f"White balance settings file not found: {settings_path}")
            return False

        try:
            with open(settings_path, "r") as f:
                settings = yaml.safe_load(f)

            combined = settings.get("combined_settings", {})
            if not combined:
                logger.warning("No combined_settings found in white balance file")
                return False

            # Apply each property
            for prop_name, value in combined.items():
                try:
                    self._set_property(prop_name, value)
                except Exception as e:
                    logger.warning(f"Failed to set {prop_name}={value}: {e}")

            logger.info(f"Applied white balance settings from {settings_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load white balance settings: {e}")
            return False

    def save_current_settings(self, output_path: str) -> None:
        """
        Save current camera settings to YAML file.

        Args:
            output_path: Path to save the settings file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        settings = {
            "metadata": {
                "camera": self.device_name,
                "description": "JAI Camera White Balance Settings",
            },
            "exposure_mode": {
                "individual": self.is_individual_exposure_enabled(),
            },
            "gain_mode": {
                "individual": self.is_individual_gain_enabled(),
            },
        }

        # Only include per-channel settings if individual mode is enabled
        if self.is_individual_exposure_enabled():
            settings["per_channel_exposures_ms"] = self.get_channel_exposures()

        if self.is_individual_gain_enabled():
            settings["per_channel_analog_gains"] = self.get_analog_gains()
            try:
                settings["per_channel_digital_gains"] = self.get_digital_gains()
            except Exception:
                pass  # Digital gains may not be available

        try:
            settings["black_levels"] = self.get_black_levels()
        except Exception:
            pass  # Black levels may not be available

        # Build combined settings for easy application
        combined = {}
        if self.is_individual_exposure_enabled():
            combined[self.EXPOSURE_INDIVIDUAL] = self.MODE_ON
            exposures = self.get_channel_exposures()
            combined[self.EXPOSURE_RED] = exposures["red"]
            combined[self.EXPOSURE_GREEN] = exposures["green"]
            combined[self.EXPOSURE_BLUE] = exposures["blue"]

        if self.is_individual_gain_enabled():
            combined[self.GAIN_INDIVIDUAL] = self.MODE_ON
            gains = self.get_analog_gains()
            combined[self.GAIN_ANALOG_RED] = gains["red"]
            combined[self.GAIN_ANALOG_GREEN] = gains["green"]
            combined[self.GAIN_ANALOG_BLUE] = gains["blue"]

        settings["combined_settings"] = combined

        with open(output_path, "w") as f:
            yaml.dump(settings, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved current camera settings to {output_path}")

    # ========== Built-in White Balance Control ==========

    def get_white_balance_mode(self) -> str:
        """
        Get current white balance mode.

        Returns:
            Current mode: 'Off', 'Continuous', 'Once', or a preset name
        """
        try:
            return self._get_property(self.WHITE_BALANCE)
        except Exception:
            return "Off"

    def set_white_balance_mode(self, mode: str) -> None:
        """
        Set camera's built-in white balance mode.

        The camera has hardware white balance with several options:
        - 'Off': Manual control (use per-channel exposure/gain)
        - 'Continuous': Auto-adjusts continuously during capture
        - 'Once': One-shot auto white balance (calibrates then stops)
        - 'Preset3200K', 'Preset5000K', 'Preset6500K', 'Preset7500K': Fixed presets

        Args:
            mode: One of WHITE_BALANCE_OPTIONS

        Raises:
            ValueError: If mode is not valid
            RuntimeError: If property cannot be set
        """
        if mode not in self.WHITE_BALANCE_OPTIONS:
            raise ValueError(
                f"Invalid white balance mode '{mode}'. "
                f"Must be one of: {self.WHITE_BALANCE_OPTIONS}"
            )
        self._set_property(self.WHITE_BALANCE, mode)
        logger.info(f"Set white balance mode to: {mode}")

    def run_auto_white_balance(self, wait_time: float = 0.5) -> None:
        """
        Run one-shot auto white balance using camera's built-in algorithm.

        This sets WhiteBalance to 'Once', which triggers a single auto-calibration.
        The camera analyzes the current scene and adjusts internal parameters.

        Args:
            wait_time: Time to wait for calibration to complete (seconds)

        Note:
            After running, the mode returns to 'Off' automatically.
            Results are applied internally by the camera - no settings file generated.
            For reproducible calibration with saved settings, use JAIWhiteBalanceCalibrator.
        """
        import time

        self._set_property(self.WHITE_BALANCE, "Once")
        logger.info("Running one-shot auto white balance...")
        time.sleep(wait_time)
        logger.info("Auto white balance complete")

    def set_white_balance_preset(self, color_temp_k: int) -> None:
        """
        Set white balance to a color temperature preset.

        Args:
            color_temp_k: Color temperature in Kelvin (3200, 5000, 6500, or 7500)

        Raises:
            ValueError: If color temperature is not a valid preset
        """
        preset_map = {
            3200: "Preset3200K",
            5000: "Preset5000K",
            6500: "Preset6500K",
            7500: "Preset7500K",
        }
        if color_temp_k not in preset_map:
            raise ValueError(
                f"Invalid preset temperature {color_temp_k}K. "
                f"Must be one of: {list(preset_map.keys())}"
            )
        self.set_white_balance_mode(preset_map[color_temp_k])

    # ========== Utility Methods ==========

    def get_all_jai_properties(self) -> Dict[str, Any]:
        """
        Get all JAI-specific properties and their current values.

        Useful for diagnostics and debugging.

        Returns:
            Dictionary of property names and values
        """
        properties = {}
        prop_names = [
            self.EXPOSURE_INDIVIDUAL,
            self.EXPOSURE_RED,
            self.EXPOSURE_GREEN,
            self.EXPOSURE_BLUE,
            self.GAIN_INDIVIDUAL,
            self.GAIN_ANALOG_RED,
            self.GAIN_ANALOG_GREEN,
            self.GAIN_ANALOG_BLUE,
            self.GAIN_DIGITAL_RED,
            self.GAIN_DIGITAL_BLUE,
            self.BLACK_LEVEL_ALL,
            self.BLACK_LEVEL_RED,
            self.BLACK_LEVEL_BLUE,
            self.FRAME_RATE,
        ]

        for prop_name in prop_names:
            try:
                properties[prop_name] = self._get_property(prop_name)
            except Exception:
                properties[prop_name] = None

        return properties

    def reset_to_defaults(self) -> None:
        """
        Reset to unified exposure/gain mode (disable individual channel control).
        """
        try:
            self.disable_individual_exposure()
        except Exception as e:
            logger.warning(f"Failed to disable individual exposure: {e}")

        try:
            self.disable_individual_gain()
        except Exception as e:
            logger.warning(f"Failed to disable individual gain: {e}")

        logger.info("Reset JAI camera to default (unified) mode")
