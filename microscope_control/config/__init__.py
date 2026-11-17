"""
Config package - Configuration management.

This package contains the configuration management system
for loading and managing microscope settings from YAML files.

Modules:
    manager: ConfigManager for YAML configuration handling
    per_detector: merges the active camera's per_detector values into a config
    profiles: picks the acquisition profile for a modality, objective and camera
"""

from microscope_control.config.manager import ConfigManager
from microscope_control.config.per_detector import apply_detector_overlay
from microscope_control.config.profiles import resolve_profile_key, sibling_for_detector

__all__ = [
    "ConfigManager",
    "apply_detector_overlay",
    "resolve_profile_key",
    "sibling_for_detector",
]
