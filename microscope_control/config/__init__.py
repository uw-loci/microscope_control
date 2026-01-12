"""
Config package - Configuration management.

This package contains the configuration management system
for loading and managing microscope settings from YAML files.

Modules:
    manager: ConfigManager for YAML configuration handling
"""

from microscope_control.config.manager import ConfigManager

__all__ = ["ConfigManager"]
