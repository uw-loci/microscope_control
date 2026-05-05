"""Pycromanager hardware implementation for microscope control.

Stage movement and autofocus logic remain here. Camera operations
(snap, exposure, debayer, white balance, live mode) are delegated to
Camera subclasses in hardware/camera/. Rotation stage operations are
delegated to RotationStage subclasses in hardware/rotation.py.
"""

import warnings
from typing import Dict, Any, Optional, Tuple
import logging
import time
from pycromanager import Core, Studio
from microscope_control.hardware.base import MicroscopeHardware, is_mm_running, is_coordinate_in_range, Position
from microscope_control.autofocus.core import AutofocusUtils
from microscope_imageprocessing.focus import (
    UnknownMetricError,
    resolve_metric,
)

import numpy as np
import skimage.filters
import scipy.interpolate
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def obj_2_list(name):
    """Convert Java object to Python list."""
    return [name.get(i) for i in range(name.size())]


class MicroManagerConnectionError(Exception):
    """Raised when connection to Micro-Manager fails."""
    pass


def init_pycromanager(timeout_seconds: float = 30.0):
    """
    Initialize Pycromanager connection to Micro-Manager.

    Args:
        timeout_seconds: Maximum time to wait for connection (default 30s)

    Returns:
        Tuple of (core, studio) if successful, (None, None) if failed

    Raises:
        MicroManagerConnectionError: If connection fails with details about the failure
    """
    # Check if MM process is running
    if not is_mm_running():
        error_msg = (
            "Micro-Manager is not running.\n"
            "Please start Micro-Manager before starting the server."
        )
        logger.error(error_msg)
        raise MicroManagerConnectionError(error_msg)

    # Try to connect to Micro-Manager
    logger.info("Connecting to Micro-Manager...")
    try:
        # Core() can hang if MM is locked - pycromanager doesn't have a timeout option
        # so we just try and catch any exceptions
        core = Core()
        studio = Studio()
        core.set_timeout_ms(20000)

        # Verify connection is working by getting a simple property
        try:
            _ = core.get_version_info()
            logger.info("Successfully connected to Micro-Manager")
        except Exception as e:
            logger.warning(f"Connected but version check failed: {e}")

        return core, studio

    except Exception as e:
        error_type = type(e).__name__
        error_msg = (
            f"Failed to connect to Micro-Manager.\n"
            f"Error type: {error_type}\n"
            f"Error: {e}\n"
            f"\n"
            f"Possible causes:\n"
            f"  1. Micro-Manager is frozen/locked - restart Micro-Manager\n"
            f"  2. Another application is using Micro-Manager\n"
            f"  3. Micro-Manager crashed - check for error dialogs\n"
            f"  4. ZMQ port conflict - restart Micro-Manager"
        )
        logger.error(error_msg)
        raise MicroManagerConnectionError(error_msg) from e


class PycromanagerHardware(MicroscopeHardware):
    """Implementation for Pycromanager-based microscopes.

    Composes a Camera (auto-detected from MM) and optional RotationStage
    (auto-detected from YAML config). Stage movement and autofocus logic
    remain on this class; camera operations are delegated to self._camera.
    """

    def __init__(self, core: Core, studio: Studio, settings: Dict[str, Any]):
        """
        Initialize PycromanagerHardware with dictionary-based settings.

        Args:
            core: Pycromanager Core object
            studio: Pycromanager Studio object
            settings: Dictionary containing microscope configuration
        """
        self.core = core
        self.studio = studio
        self.settings = settings

        # Stage inversion correction flag. Set by apply_mode_setup() when
        # a profile requires software X-axis inversion because the merged
        # MM config uses a different Invert-X setting than the modality
        # originally expected. Acquisition/coordinate code checks this.
        self.stage_invert_x_correction = False

        # Log microscope info
        microscope_info = settings.get("microscope", {})
        logger.info(
            "Initializing hardware for microscope: %s",
            microscope_info.get("name", "Unknown"),
        )

        # Build camera registry: detector_id -> Camera instance
        # Multi-camera systems (e.g. brightfield + laser scanning) have
        # multiple cameras available. The active camera is whichever MM
        # currently has selected. The registry lets callers switch.
        self._camera_registry = self._build_camera_registry()
        self._camera_name = self._detect_camera_name()
        self._active_detector_id = self._find_detector_id(self._camera_name)

        # Create other components
        self._stage = self._create_stage()
        self._rotation_stage = self._create_rotation_stage()

        # Create optional illumination and detector components
        self._illumination = self._create_illumination()
        self._detector = self._create_detector()

        # Last successfully applied acquisition profile name (set by
        # apply_mode_setup). Used by GETCAP to answer "what's the current
        # state?" without the caller having to remember the last profile.
        # Strip any trailing "_<counter>" suffix to match the canonical
        # name in acquisition_profiles.
        self._active_profile: "str | None" = None

        # Objective swap is available when config defines sequences
        if self.settings.get("objective_swap_sequences"):
            logger.info("Objective swap sequences found in config")

    # --- Camera registry (multi-camera support) ---

    def _detect_camera_name(self) -> str:
        """Detect the active camera from MM Core."""
        props = self.get_device_properties()
        return props["Core"]["Camera"]

    def _build_camera_registry(self) -> Dict[str, Any]:
        """Build Camera instances for all detectors defined in config.

        Each detector in the id_detector config section gets a Camera
        instance. The appropriate subclass is chosen based on the
        device name (JAICamera, LaserScanningCamera, or generic).

        Returns:
            Dict of detector_id -> Camera instance
        """
        from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera
        from microscope_control.hardware.camera.jai_camera import JAICamera
        from microscope_control.hardware.camera.laser_scanning_camera import LaserScanningCamera

        registry = {}
        id_detectors = self.settings.get("id_detector", {})
        for det_id, det_config in id_detectors.items():
            if not isinstance(det_config, dict):
                continue
            device = det_config.get("device")
            if not device:
                continue
            try:
                cam = self._create_camera_for_device(
                    device, det_config, det_id,
                )
                registry[det_id] = cam
                logger.info(
                    "Registered camera: %s -> %s (flip_x=%s, flip_y=%s)",
                    det_id, type(cam).__name__,
                    cam.flip_x, cam.flip_y,
                )
            except Exception as e:
                logger.warning("Could not create camera for %s: %s", det_id, e)
        return registry

    def _create_camera_for_device(self, device_name: str,
                                   detector_config: Dict[str, Any],
                                   detector_id: str):
        """Create the appropriate Camera subclass for a device.

        Uses the ``camera_type`` field from detector config to select the
        Camera subclass via a registry dict.  New camera types are added
        by importing the class and adding one entry to CAMERA_TYPES.

        Args:
            device_name: MM device name (e.g. 'JAICamera', 'OSc-LSM', 'QCamera')
            detector_config: Config dict from id_detector section
            detector_id: The detector's LOCI ID (e.g. 'LOCI_DETECTOR_JAI_001')
        """
        from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera
        from microscope_control.hardware.camera.jai_camera import JAICamera
        from microscope_control.hardware.camera.laser_scanning_camera import LaserScanningCamera

        CAMERA_TYPES = {
            "jai": JAICamera,
            "laser_scanning": LaserScanningCamera,
            "generic": PycromanagerCamera,
        }

        camera_type = detector_config.get("camera_type", "generic")
        cls = CAMERA_TYPES.get(camera_type, PycromanagerCamera)
        if cls is PycromanagerCamera and camera_type != "generic":
            logger.warning(
                "Unknown camera_type '%s' for detector %s, "
                "falling back to generic PycromanagerCamera",
                camera_type, detector_id,
            )
        return cls(self.core, self.studio, detector_config)

    def _find_detector_id(self, camera_name: str) -> Optional[str]:
        """Find the detector ID for a given MM camera device name."""
        id_detectors = self.settings.get("id_detector", {})
        for det_id, det_config in id_detectors.items():
            if isinstance(det_config, dict) and det_config.get("device") == camera_name:
                return det_id
        return None

    def _find_detector_config(self, camera_name: Optional[str] = None) -> Dict[str, Any]:
        """Find detector configuration from YAML settings.

        Args:
            camera_name: MM device name to look up. Defaults to active camera.

        Returns:
            Config dict from id_detector section, or empty dict.
        """
        if camera_name is None:
            camera_name = self._camera_name
        id_detectors = self.settings.get("id_detector", {})
        for det_id, det_config in id_detectors.items():
            if isinstance(det_config, dict):
                if det_config.get("device") == camera_name:
                    return det_config
        return {}

    def get_camera_for_detector(self, detector_id: str):
        """Get the Camera instance for a specific detector.

        Args:
            detector_id: LOCI detector ID (e.g. 'LOCI_DETECTOR_JAI_001')

        Returns:
            Camera instance for that detector

        Raises:
            KeyError: If detector_id is not in the registry
        """
        if detector_id not in self._camera_registry:
            raise KeyError(
                f"Detector '{detector_id}' not in registry. "
                f"Available: {list(self._camera_registry.keys())}"
            )
        return self._camera_registry[detector_id]

    def set_active_camera(self, detector_id: str) -> None:
        """Switch the active camera to a different detector.

        Updates the MM Core camera device and sets the new Camera
        as the active one. Callers should also update illumination,
        shutter, and detector settings as needed for the new modality.

        Args:
            detector_id: LOCI detector ID to activate
        """
        cam = self.get_camera_for_detector(detector_id)
        device_name = cam.get_name()
        self.core.set_property("Core", "Camera", device_name)
        self._camera_name = device_name
        self._active_detector_id = detector_id
        logger.info(
            "Active camera switched to %s (%s, flip_x=%s, flip_y=%s)",
            detector_id, device_name, cam.flip_x, cam.flip_y,
        )

    @property
    def camera_registry(self) -> Dict[str, Any]:
        """All registered cameras, keyed by detector ID."""
        return self._camera_registry

    def _create_illumination(self):
        """Create illumination source from modality config, or None.

        Searches modalities for an illumination or pockels_cell section
        and creates the appropriate subclass. The first modality with
        config is used as the default; mode-switching code can replace
        it later via the illumination setter.
        """
        modalities = self.settings.get("modalities", {})
        for mod_name, mod_config in modalities.items():
            if not isinstance(mod_config, dict):
                continue
            source = self._build_illumination_from_config(mod_config)
            if source is not None:
                logger.info("Created default illumination: %s from modality '%s'",
                            type(source).__name__, mod_name)
                return source
        return None

    def _build_illumination_from_config(self, mod_config):
        """Build an Illumination instance from a modality config dict.

        Checks for 'illumination' (LED/DiaLamp) and 'pockels_cell'
        (laser power) sections. Returns the first one found, or None.
        """
        from microscope_control.hardware.illumination import (
            LEDIllumination, DevicePropertyIllumination, PockelsCell,
        )
        # Check for transmitted/epi illumination (LED, DiaLamp, etc.)
        illum = mod_config.get("illumination")
        if isinstance(illum, dict) and illum.get("device"):
            device = illum["device"]
            illum_type = illum.get("type", "device_property")
            # Validate required config BEFORE construction (not inside
            # try/except, so missing config is not silently swallowed)
            if illum_type == "analog_voltage":
                if "max_voltage" not in illum:
                    raise ValueError(
                        f"Illumination 'max_voltage' not specified in "
                        f"config for device '{device}'. Add max_voltage "
                        f"to the illumination section in your microscope "
                        f"config YAML."
                    )
            else:
                if "max_intensity" not in illum:
                    raise ValueError(
                        f"Illumination 'max_intensity' not specified in "
                        f"config for device '{device}'. Add max_intensity "
                        f"to the illumination section in your microscope "
                        f"config YAML."
                    )
            try:
                if illum_type == "analog_voltage":
                    return LEDIllumination(
                        self.core,
                        device_name=device,
                        max_voltage=illum["max_voltage"],
                        label=illum.get("label", device),
                    )
                else:
                    return DevicePropertyIllumination(
                        self.core,
                        device_name=device,
                        state_property=illum.get("state_property", "State"),
                        intensity_property=illum.get("intensity_property", "Intensity"),
                        max_intensity=illum["max_intensity"],
                        label=illum.get("label", device),
                    )
            except Exception as e:
                logger.warning("Could not create illumination for %s: %s", device, e)

        # Check for Pockels cell (laser power modulation)
        pockels = mod_config.get("pockels_cell")
        if isinstance(pockels, dict) and pockels.get("device"):
            device = pockels["device"]
            if "max_voltage" not in pockels:
                raise ValueError(
                    f"Pockels cell 'max_voltage' not specified in config "
                    f"for device '{device}'. Add max_voltage to the "
                    f"pockels_cell section in your microscope config YAML."
                )
            try:
                return PockelsCell(
                    self.core,
                    device_name=device,
                    max_voltage=pockels["max_voltage"],
                    label=pockels.get("label", "Pockels Cell"),
                )
            except Exception as e:
                logger.warning("Could not create Pockels cell for %s: %s", device, e)

        return None

    def get_illumination_for_modality(self, modality_name: str):
        """Create and return the illumination source for a specific modality.

        Looks up the modality in settings and builds the appropriate
        Illumination instance. Does NOT change self._illumination --
        callers should assign it if switching modes.

        Args:
            modality_name: Key in the modalities config (e.g. 'brightfield', '2p')

        Returns:
            Illumination instance, or None if not configured.
        """
        modalities = self.settings.get("modalities", {})
        mod_config = modalities.get(modality_name)
        if not isinstance(mod_config, dict):
            return None
        return self._build_illumination_from_config(mod_config)

    def _create_detector(self):
        """Create external detector (PMT) from modality config, or None.

        Searches modalities for a pmt section and creates the appropriate
        Detector subclass based on the device type.
        """
        from microscope_control.hardware.detector import PMTDetector, DCUDetector

        modalities = self.settings.get("modalities", {})
        for mod_name, mod_config in modalities.items():
            if not isinstance(mod_config, dict):
                continue
            pmt = mod_config.get("pmt")
            if not isinstance(pmt, dict):
                continue
            device = pmt.get("device")
            if not device:
                continue
            # Validate required config BEFORE construction (not inside
            # try/except, so missing config is not silently swallowed)
            if "connector" not in pmt:
                raise ValueError(
                    f"PMT 'connector' not specified in config for "
                    f"device '{device}'. Add connector to the pmt "
                    f"section in your microscope config YAML."
                )
            if "max_gain_percent" not in pmt:
                raise ValueError(
                    f"PMT 'max_gain_percent' not specified in config "
                    f"for device '{device}'. Add max_gain_percent to "
                    f"the pmt section in your microscope config YAML."
                )
            try:
                pmt_type = pmt.get("type", "dcc")
                connector = pmt["connector"]
                max_gain = pmt["max_gain_percent"]
                if pmt_type == "dcu":
                    det = DCUDetector(
                        self.core,
                        device_name=device,
                        channel=connector,
                        max_gain_percent=max_gain,
                    )
                else:
                    det = PMTDetector(
                        self.core,
                        device_name=device,
                        connector=connector,
                        max_gain_percent=max_gain,
                    )
                logger.info("Created detector: %s (%s) from modality '%s'",
                            type(det).__name__, device, mod_name)
                return det
            except Exception as e:
                logger.warning("Could not create detector for %s: %s",
                               device, e)
        return None

    # --- MM ConfigGroup preset application ---

    def apply_config_preset(self, group: str, preset: str) -> None:
        """Apply a Micro-Manager ConfigGroup preset.

        This is the primary mechanism for switching light paths, filter
        wheels, shutters, and other state devices. MM presets bundle
        multiple device property changes into a single atomic operation.

        Args:
            group: ConfigGroup name (e.g. 'Light Path', 'Lens Turret')
            preset: Preset name within the group (e.g. '2-R100 (BF Camera)')
        """
        logger.info("Applying MM config preset: [%s] -> '%s'", group, preset)
        self.core.set_config(group, preset)
        self.core.wait_for_config(group, preset)
        logger.debug("Config preset applied successfully")

    def apply_profile_illumination(self, profile_name: str) -> Optional[float]:
        """Apply ONLY the illumination intensity from an acquisition profile.

        This is a surgical subset of :meth:`apply_mode_setup` intended for
        background collection and other workflows that need the profile's
        lamp level without disturbing the rest of the hardware state.
        Specifically, this method DOES NOT:

        * Move any stages (Z/F)
        * Switch cameras or detectors
        * Apply MM ConfigGroup presets (light path, filter wheels, ...)
        * Disable PMTs or shutters

        It only reads ``illumination_intensity`` from the resolved profile
        and calls ``self._illumination.set_power(intensity)`` if both the
        profile value and the active illumination device are available.

        Background collection uses this to guarantee the sensor sees the
        same lamp level at collection time that the subsequent tiled
        acquisition will use, without disrupting the user's manually
        positioned blank area or the current detector/optics state.

        Args:
            profile_name: Key in ``acquisition_profiles``. May include a
                trailing ``_<counter>`` suffix (stripped via
                :meth:`_resolve_profile_key`).

        Returns:
            The illumination intensity that was applied, or ``None`` if
            the profile could not be resolved, the profile has no
            ``illumination_intensity`` entry, or no illumination device
            is active.
        """
        profiles = self.settings.get("acquisition_profiles", {})
        resolved_key = self._resolve_profile_key(profile_name, profiles)
        if resolved_key is None:
            logger.warning(
                "apply_profile_illumination: no acquisition profile found for '%s'. "
                "Available profiles: %s",
                profile_name, sorted(profiles.keys()),
            )
            return None

        profile = profiles[resolved_key]
        illum_intensity = profile.get("illumination_intensity")
        if illum_intensity is None:
            logger.info(
                "apply_profile_illumination: profile '%s' has no illumination_intensity; skipping",
                resolved_key,
            )
            return None

        if self._illumination is None:
            logger.info(
                "apply_profile_illumination: no active illumination device; cannot set intensity %s for profile '%s'",
                illum_intensity, resolved_key,
            )
            return None

        try:
            self._illumination.set_power(illum_intensity)
            logger.info(
                "apply_profile_illumination: set illumination to %s from profile '%s'",
                illum_intensity, resolved_key,
            )
            return float(illum_intensity)
        except Exception as e:
            logger.warning(
                "apply_profile_illumination: failed to set illumination %s for profile '%s': %s",
                illum_intensity, resolved_key, e,
            )
            return None

    def apply_mode_setup(self, profile_name: str) -> None:
        """Apply all setup steps for an acquisition profile.

        SAFETY: This method enforces a strict sequence to protect PMTs
        from bright light damage. Before any illumination or light path
        changes, all PMT outputs are disabled and detector shutters are
        closed. The sequence is:

        1. SAFETY: Disable PMT outputs + close detector shutters
        2. Turn off current illumination
        3. Apply MM ConfigGroup presets (light path, filter, etc.)
        4. Switch camera/detector
        5. Switch illumination source and set intensity
        6. Apply mode positions (Z, F stages)

        Args:
            profile_name: Key in acquisition_profiles. May include a
                trailing "_<counter>" suffix from the Java extension
                (e.g. ``Brightfield_10x_8``), which is stripped before
                lookup. Accepts both the canonical form
                (``Brightfield_10x``) and legacy short forms
                (``bf_10x``) via fallback matching.
        """
        profiles = self.settings.get("acquisition_profiles", {})

        # Microscopes without profile-based mode setup (e.g. PPM) have no
        # acquisition_profiles section at all. That is a valid configuration;
        # workflow.py calls apply_mode_setup unconditionally, so skip quietly
        # instead of logging a misleading "no profile found" warning.
        if not profiles:
            logger.debug(
                "apply_mode_setup('%s'): no acquisition_profiles configured; skipping",
                profile_name,
            )
            return

        # Resolve profile key, stripping any trailing "_<counter>" suffix
        # that the Java extension appends (e.g. Brightfield_10x_8 -> _8).
        resolved_key = self._resolve_profile_key(profile_name, profiles)
        if resolved_key is None:
            logger.warning(
                "No acquisition profile found for '%s' (tried exact match, "
                "stripped counter suffix, and short-form fallback). "
                "Available profiles: %s",
                profile_name, sorted(profiles.keys()),
            )
            return

        profile = profiles[resolved_key]
        if resolved_key != profile_name:
            logger.info(
                "Applying mode setup for profile: %s (resolved from '%s')",
                resolved_key, profile_name,
            )
        else:
            logger.info("Applying mode setup for profile: %s", resolved_key)
        profile_name = resolved_key  # Use the resolved key for the rest of the method

        # === STEP 0: SAFETY -- protect PMTs before any light changes ===
        self._safe_disable_pmt_and_shutters()

        # === STEP 1: Turn off ALL configured illumination sources ===
        # Not just self._illumination -- profile A may have enabled a light
        # via an mm_setup_presets ConfigGroup write rather than through the
        # tracked _illumination object, in which case .off() on the current
        # one is a no-op for the light that's actually emitting. Symptom
        # observed 2026-04-26 on OWS3: switching FL -> BF preset left the
        # fluorescence LED on, saturating the brightfield image.
        # Iterate every modality's illumination config and call .off() so
        # the new profile starts from a known-dark state.
        self._disable_all_modality_illuminations()

        # === STEP 2: Apply MM ConfigGroup presets ===
        for preset_spec in profile.get("mm_setup_presets", []):
            group = preset_spec.get("group")
            preset = preset_spec.get("preset")
            if group and preset:
                self.apply_config_preset(group, preset)

        # === STEP 3: Switch detector/camera if specified ===
        detector_id = profile.get("detector")
        if detector_id and detector_id != self._active_detector_id:
            if detector_id in self._camera_registry:
                self.set_active_camera(detector_id)
            else:
                logger.warning("Detector '%s' not in camera registry", detector_id)

        # === STEP 4: Switch illumination source ===
        modality_name = profile.get("modality")
        if modality_name:
            new_illum = self.get_illumination_for_modality(modality_name)
            if new_illum is not None:
                self._illumination = new_illum
                logger.info("Switched illumination to %s for modality '%s'",
                            type(new_illum).__name__, modality_name)

        # === STEP 5: Set illumination intensity ===
        # Channel-based profiles (widefield IF, BF+IF) drive illumination
        # per-channel through the channel library's device_properties, so the
        # profile-level illumination_intensity is meaningless -- and on some
        # hardware (OWS3 LappMainBranch1 where the legacy intensity_property
        # is the discrete "State" property) calling set_power() with a float
        # like 1.0 throws "Invalid property value" and crashes the whole
        # apply_mode_setup path before the channel loop ever runs. Skip the
        # legacy set_power entirely when channels are declared.
        profile_has_channels = bool(profile.get("channels"))
        illum_intensity = profile.get("illumination_intensity")
        if illum_intensity is not None and self._illumination is not None:
            if profile_has_channels:
                logger.info(
                    "Skipping profile-level illumination_intensity=%s for '%s': "
                    "profile is channel-based and the per-channel device_properties "
                    "drive illumination.",
                    illum_intensity,
                    profile_name,
                )
            else:
                self._illumination.set_power(illum_intensity)

        # === STEP 6: Apply mode positions (Z, F stages) ===
        mode_positions = self.settings.get("mode_positions", {})
        mode_pos = mode_positions.get(profile_name)
        if mode_pos:
            z_pos = mode_pos.get("z")
            if z_pos is not None:
                logger.info("Setting mode Z position: %.1f", z_pos)
                self._stage.move_z_no_wait(z_pos)
            f_pos = mode_pos.get("f")
            if f_pos is not None and hasattr(self._stage, 'move_f'):
                logger.info("Setting mode F position: %.1f", f_pos)
                self._stage.move_f(f_pos)

        # === STEP 7: Stage inversion correction ===
        # When BF and 2P MM configs are merged, one modality may need
        # software X-axis inversion because the merged config uses a
        # single Invert-X pre-init setting. Profiles that originally
        # used a different inversion set stage_invert_x_correction: true.
        self.stage_invert_x_correction = bool(
            profile.get("stage_invert_x_correction", False)
        )
        if self.stage_invert_x_correction:
            logger.info("Stage X-axis inversion correction ACTIVE for this profile")

        logger.info("Mode setup complete for profile: %s", profile_name)
        self._active_profile = profile_name

    def _disable_all_modality_illuminations(self) -> None:
        """Turn off every illumination source declared in the modalities config.

        Symmetric-teardown helper for profile switches: iterate every
        modality, build its illumination object the same way
        get_illumination_for_modality does, and call .off() on each. This
        guarantees that whatever was emitting before the switch -- whether
        it was the tracked self._illumination or a light enabled via a
        prior profile's mm_setup_presets ConfigGroup write -- is dark
        before the new profile applies its own setup.

        Failures per-source are logged but do not abort the sweep: a
        misconfigured optional source must not block teardown of the
        source that's actually saturating the next acquisition.
        """
        modalities = self.settings.get("modalities", {})
        if not isinstance(modalities, dict) or not modalities:
            return

        # Always include self._illumination explicitly even if it doesn't
        # round-trip through a modality (older configs / programmatic
        # overrides). De-dup by device name to avoid double-toggles.
        seen_devices: set = set()

        def _try_off(source, label: str) -> None:
            if source is None:
                return
            try:
                source.off()
                logger.info("Pre-switch teardown: turned off %s", label)
            except Exception as e:
                logger.warning(
                    "Pre-switch teardown: could not turn off %s: %s", label, e
                )

        if self._illumination is not None:
            dev = getattr(self._illumination, "_device", None) \
                or getattr(self._illumination, "_label", "current")
            seen_devices.add(str(dev))
            _try_off(self._illumination, f"current illumination ({dev})")

        for mod_name, mod_config in modalities.items():
            if not isinstance(mod_config, dict):
                continue
            illum_cfg = mod_config.get("illumination") \
                or mod_config.get("pockels_cell")
            device_name = (
                illum_cfg.get("device") if isinstance(illum_cfg, dict) else None
            )
            if device_name and device_name in seen_devices:
                continue
            try:
                source = self._build_illumination_from_config(mod_config)
            except Exception as e:
                logger.debug(
                    "Pre-switch teardown: could not build illumination for "
                    "modality '%s': %s", mod_name, e,
                )
                continue
            if source is None:
                continue
            if device_name:
                seen_devices.add(device_name)
            _try_off(source, f"modality '{mod_name}' illumination ({device_name})")

    def _safe_disable_pmt_and_shutters(self) -> None:
        """SAFETY: Disable all PMT outputs and close detector shutters.

        Called before ANY light path or illumination change to prevent
        PMT damage from bright light. This method is intentionally
        conservative -- it will attempt to disable everything even if
        some steps fail, and waits for the system after each step.

        Reads the ``pmt_safety`` section from config for device-specific
        shutter control. If no pmt_safety config exists, still attempts
        to disable the detector via the Detector abstraction.

        Config example::

            pmt_safety:
              detector_shutter:
                device: 'Arduino-Switch'
                closed_state: '0'          # String state, never boolean
                closed_label: 'Closed and Off'
              laser_shutter:
                device: 'LaserShutter'
                closed_state: '0'
        """
        logger.info("SAFETY: Disabling PMTs and closing shutters")

        # 1. Disable PMT via Detector abstraction (if configured)
        if self._detector is not None:
            try:
                # Use disable_all_channels if available (multi-channel PMTs)
                if hasattr(self._detector, 'disable_all_channels'):
                    self._detector.disable_all_channels()
                else:
                    self._detector.disable()
                logger.info("SAFETY: PMT disabled")
            except Exception as e:
                logger.error("SAFETY WARNING: Could not disable PMT: %s", e)

        # 2. Close detector shutters via config-driven device properties
        pmt_safety = self.settings.get("pmt_safety", {})

        for shutter_key in ("detector_shutter", "laser_shutter"):
            shutter_config = pmt_safety.get(shutter_key)
            if not isinstance(shutter_config, dict):
                continue
            device = shutter_config.get("device")
            if not device:
                continue
            try:
                # Prefer ConfigGroup preset if specified
                preset_group = shutter_config.get("preset_group")
                preset_name = shutter_config.get("closed_preset")
                if preset_group and preset_name:
                    self.apply_config_preset(preset_group, preset_name)
                else:
                    # Direct property set with explicit string state
                    closed_state = str(shutter_config.get("closed_state", "0"))
                    self.core.set_property(device, "State", closed_state)
                self.core.wait_for_system()
                logger.info("SAFETY: %s closed (%s)", shutter_key, device)
            except Exception as e:
                logger.error(
                    "SAFETY WARNING: Could not close %s (%s): %s",
                    shutter_key, device, e,
                )

        # 3. Final system wait to ensure all safety states are applied
        try:
            self.core.wait_for_system()
        except Exception:
            pass
        logger.info("SAFETY: PMT protection sequence complete")

    @staticmethod
    def _resolve_profile_key(profile_name: str, profiles: dict) -> "str | None":
        """Match a scan_type / profile_name against the acquisition_profiles dict.

        The Java extension sends scan_type as ``<Modality>_<Objective>_<counter>``
        (e.g. ``Brightfield_10x_8``). The counter varies across acquisitions
        and cannot be a stable profile key, so this helper tries several
        lookup strategies in order:

        1. Exact match (legacy behaviour)
        2. Strip a trailing ``_<digits>`` counter and retry
        3. Translate ``Modality_Objective`` to a short form by taking the
           first two letters of the modality (``bf_10x``, ``fl_10x``, etc.)
        4. Case-insensitive matching against all keys

        Returns the matched key (as stored in the profiles dict), or
        ``None`` if no match can be found.
        """
        if not profile_name or not isinstance(profiles, dict) or not profiles:
            return None

        # Strategy 1: exact match
        if profile_name in profiles:
            return profile_name

        # Strategy 2: strip trailing "_<digits>" counter
        import re
        stripped = re.sub(r"_\d+$", "", profile_name)
        if stripped != profile_name and stripped in profiles:
            return stripped

        # Strategy 3: short-form translation for common modality names.
        # "Brightfield_10x" -> "bf_10x", "Fluorescence_20x" -> "fl_20x"
        short_form = None
        parts = stripped.split("_", 1)
        if len(parts) == 2:
            modality_short_forms = {
                "brightfield": "bf",
                "fluorescence": "fl",
                "widefield": "wf",
                "polarized": "ppm",
                "ppm": "ppm",
                "laserscanning": "ls",
                "multiphoton": "2p",
                "shg": "shg",
            }
            short = modality_short_forms.get(parts[0].lower())
            if short:
                short_form = short + "_" + parts[1]
                if short_form in profiles:
                    return short_form

        # Strategy 4: case-insensitive match against all keys
        stripped_lower = stripped.lower()
        for key in profiles:
            if key.lower() == stripped_lower:
                return key
        if short_form:
            short_lower = short_form.lower()
            for key in profiles:
                if key.lower() == short_lower:
                    return key

        return None

    def _create_stage(self):
        """Create the Stage instance and (re)build the position cache.

        The stage and its background-polling cache have the same
        lifetime: any time we rebuild the stage (initial construction
        or config reload), we tear down the old cache thread and start
        a fresh one bound to the new stage. This keeps cache and
        hardware in lockstep without requiring callers (e.g. the
        CONFIG handler) to remember to update both.
        """
        from microscope_control.hardware.stage import PycromanagerStage
        from microscope_control.hardware.stage_cache import StagePositionCache

        new_stage = PycromanagerStage(self.core, self.settings)

        # Replace any existing cache. On first construction
        # _stage_cache won't exist yet; on config reload it does
        # and its polling thread must be stopped before we drop
        # the reference.
        old_cache = getattr(self, "_stage_cache", None)
        if old_cache is not None:
            try:
                old_cache.stop()
            except Exception as e:
                logger.warning(
                    "Could not cleanly stop previous StagePositionCache: %s", e,
                )
        self._stage_cache = StagePositionCache(new_stage)
        self._stage_cache.start()

        return new_stage

    def _create_rotation_stage(self):
        """Create the appropriate RotationStage subclass, or None."""
        from microscope_control.hardware.rotation import (
            PIZRotationStage, ThorRotationStage, DummyRotationStage,
        )

        rotation_config = self._find_rotation_stage_config()
        if rotation_config is None:
            return None

        r_device_name, optics_disabled = rotation_config

        if optics_disabled:
            logger.info("Rotation optics disabled (ppm_optics=NA), using DummyRotationStage")
            return DummyRotationStage()

        # Look up the actual MM device name from id_stage config
        mm_device = (
            self.settings.get("id_stage", {}).get(r_device_name, {}).get("device")
        )
        if not mm_device:
            raise ValueError(
                f"No rotation stage device found in configuration. "
                f"Expected device '{r_device_name}' in id_stage section."
            )

        # Use rotation_type from config to select subclass via registry.
        # Falls back to matching device name for backward compatibility.
        stage_config = self.settings.get("id_stage", {}).get(r_device_name, {})
        rotation_type = stage_config.get("rotation_type", "").lower()

        ROTATION_TYPES = {
            "piz": PIZRotationStage,
            "thor": ThorRotationStage,
            "dummy": DummyRotationStage,
        }

        try:
            if rotation_type in ROTATION_TYPES:
                cls = ROTATION_TYPES[rotation_type]
            else:
                logger.debug(
                    "No rotation_type in config for '%s', using device name '%s'",
                    r_device_name, mm_device,
                )
                # Legacy fallback: match on device name
                cls = None

            if cls == PIZRotationStage or (cls is None and "PIZ" in mm_device.upper()):
                # PIZ offset lookup:
                #   1. id_stage.<device>.piz_offset  (per-stage in resources)
                #   2. modalities.<mod>.pizstage_offset  (per-modality in config)
                offset = stage_config.get("piz_offset")
                if offset is None:
                    for _mn, _mc in self.settings.get("modalities", {}).items():
                        if isinstance(_mc, dict) and "pizstage_offset" in _mc:
                            offset = _mc["pizstage_offset"]
                            break
                if offset is None:
                    raise ValueError(
                        "PIZ rotation stage requires pizstage_offset in the "
                        "modality config (or piz_offset in stage config). "
                        "Run Polarizer Calibration to determine this value."
                    )
                offset = float(offset)
                units_per_deg = stage_config.get("units_per_deg")
                if units_per_deg is None:
                    raise ValueError(
                        f"PIZ rotation stage 'units_per_deg' not specified "
                        f"in config for device '{r_device_name}'. "
                        f"Add units_per_deg to the id_stage section in your "
                        f"microscope config YAML (e.g. units_per_deg: 1000.0)."
                    )
                stage = PIZRotationStage(self.core, mm_device, offset,
                                         float(units_per_deg))
            elif cls == ThorRotationStage or (cls is None and "Thor" in mm_device):
                units_per_deg = stage_config.get("units_per_deg")
                if units_per_deg is None:
                    raise ValueError(
                        f"Thor rotation stage 'units_per_deg' not specified "
                        f"in config for device '{r_device_name}'. "
                        f"Add units_per_deg to the id_stage section in your "
                        f"microscope config YAML (e.g. units_per_deg: 2.0)."
                    )
                thor_offset = stage_config.get("thor_offset")
                if thor_offset is None:
                    raise ValueError(
                        f"Thor rotation stage 'thor_offset' not specified "
                        f"in config for device '{r_device_name}'. "
                        f"Add thor_offset to the id_stage section in your "
                        f"microscope config YAML (e.g. thor_offset: 276)."
                    )
                stage = ThorRotationStage(self.core, mm_device,
                                          float(units_per_deg),
                                          float(thor_offset))
            elif cls == DummyRotationStage:
                stage = DummyRotationStage()
            else:
                raise ValueError(
                    f"Unknown rotation device: {mm_device}. "
                    f"Set 'rotation_type' in id_stage config to one of: "
                    f"{list(ROTATION_TYPES.keys())}"
                )
            # Verify we can read the current angle
            _ = stage.get_angle()
            logger.info("Rotation stage initialized (device: %s)", r_device_name)
            return stage
        except Exception as e:
            logger.warning("Rotation stage init failed: %s. Continuing without.", e)
            return None

    # --- Component properties ---

    @property
    def camera(self):
        """The currently active camera.

        On multi-camera systems, this returns the Camera for whichever
        detector is currently active. Use set_active_camera() to switch,
        or get_camera_for_detector() for a specific detector.
        """
        if self._active_detector_id and self._active_detector_id in self._camera_registry:
            return self._camera_registry[self._active_detector_id]
        # Fallback: if no detector ID matched, return first registered camera
        if self._camera_registry:
            return next(iter(self._camera_registry.values()))
        # Last resort: create and cache a generic camera for the active MM camera
        if not hasattr(self, '_fallback_camera'):
            from microscope_control.hardware.camera.pycromanager_camera import PycromanagerCamera
            self._fallback_camera = PycromanagerCamera(self.core, self.studio, {})
        return self._fallback_camera

    @property
    def stage(self):
        """The XYZ stage attached to this microscope."""
        return self._stage

    @property
    def stage_cache(self):
        """Background-polled cache of the latest XYZ stage position.

        Use ``hardware.stage_cache.get_cached_position()`` for non-
        critical reads (live position display, frame overlays,
        progress logging) -- they hit an in-memory snapshot instead
        of the serial bus. Critical reads (focus-scan endpoints,
        per-tile metadata) should call ``force_refresh()`` or use
        ``get_current_position()`` which always live-queries.
        """
        return self._stage_cache

    @property
    def rotation_stage(self):
        """The rotation stage, or None if not configured."""
        return self._rotation_stage

    @property
    def illumination(self):
        """The primary illumination source, or None."""
        return self._illumination

    @illumination.setter
    def illumination(self, value):
        """Allow setting illumination (e.g. during modality switch)."""
        self._illumination = value

    @property
    def detector(self):
        """External detector (e.g. PMT), or None."""
        return self._detector

    @detector.setter
    def detector(self, value):
        """Allow setting detector (e.g. during modality switch)."""
        self._detector = value

    def _find_rotation_stage_config(self):
        """Check modalities config for a rotation stage.

        Returns:
            Tuple of (device_name, optics_disabled) if a rotation stage is
            configured, or None if no modality has a rotation stage.
            optics_disabled is True when ppm_optics == "NA" (hardware present
            but disabled).
        """
        modalities = self.settings.get("modalities", {})
        for mod_name, mod_config in modalities.items():
            if not isinstance(mod_config, dict):
                continue
            rot_stage = mod_config.get("rotation_stage", {})
            if isinstance(rot_stage, dict) and rot_stage.get("device"):
                # Check if optics are disabled (modalities.<mod>.optics == "NA")
                optics_value = mod_config.get("optics", 1)
                optics_disabled = (str(optics_value) == "NA")
                return (rot_stage["device"], optics_disabled)
        return None

    def move_to_position(self, position: Position) -> None:
        """Move stage to specified position with coordinate validation.

        Overrides the base class delegation to add:
        - Coordinate range validation against configured stage limits
        - Per-axis timing diagnostics

        Only moves axes that are explicitly specified (not None).
        For example, Position(x, y) moves XY only without touching Z.

        The entire non-blocking-set + wait sequence runs under the stage's
        reentrant lock so that concurrent callers from multiple client
        threads (rapid Z scroll, position pollers, acquisition, Smooth
        Focus) are serialized. Without this lock, overlapping move_z
        calls caused the 2026-04-15 Z scroll incident where wait_z
        busy-poll hung for the full 10 s timeout because the stage kept
        getting retargeted mid-wait.
        """
        t_total = time.perf_counter()

        # Validate only the originally-specified axes
        specified_axes = position.get_specified_axes()
        has_xy = "x" in specified_axes and "y" in specified_axes
        has_z = "z" in specified_axes

        if not is_coordinate_in_range(self.settings, position, axes=specified_axes):
            logger.info(f"Current stage limits: {self.settings.get('stage', {})}")
            logger.info(f"Requested position: {position}")
            raise ValueError(f"Position out of range: {position}")

        with self._stage.lock:
            # Issue moves (non-blocking)
            t0 = time.perf_counter()
            if has_z:
                self._stage.move_z_no_wait(position.z)
            if has_xy:
                self._stage.move_xy_no_wait(position.x, position.y)
            t_set = (time.perf_counter() - t0) * 1000

            # Wait for completion
            t_wait_xy = 0
            t_wait_z = 0
            if has_xy:
                t0 = time.perf_counter()
                self._stage.wait_xy()
                t_wait_xy = (time.perf_counter() - t0) * 1000
            if has_z:
                t0 = time.perf_counter()
                self._stage.wait_z()
                t_wait_z = (time.perf_counter() - t0) * 1000

        t_total_ms = (time.perf_counter() - t_total) * 1000
        logger.debug(
            f"move_to_position timing: total={t_total_ms:.0f}ms "
            f"[set={t_set:.0f}ms, "
            f"wait_xy={t_wait_xy:.0f}ms, wait_z={t_wait_z:.0f}ms] "
            f"-> {position}"
        )

    def get_current_position(self, max_retries: int = 3) -> Position:
        """Get current stage position with retry on transient serial errors.

        Overrides the base class delegation to add retry logic for
        transient serial communication failures.
        """
        for attempt in range(1, max_retries + 1):
            try:
                x, y, z = self._stage.get_xyz()
                return Position(x, y, z)
            except Exception as e:
                if attempt < max_retries and "Serial command failed" in str(e):
                    logger.warning(
                        f"get_current_position failed (attempt {attempt}/{max_retries}), "
                        f"retrying in 200ms: {e}"
                    )
                    time.sleep(0.2)
                else:
                    raise

    # snap_image, set_exposure, get_exposure, get_fov, get_pixel_size_um,
    # start/stop_continuous_acquisition, get_live_frame, and white_balance
    # are now delegated through MicroscopeHardware -> self._camera.
    # See camera/base.py, camera/pycromanager_camera.py, camera/jai_camera.py

    def autofocus(
        self,
        n_steps=4000000,
        search_range=20,
        interp_strength=100,
        interp_kind="quadratic",
        score_metric=skimage.filters.sobel,
        pop_a_plot=False,
        move_stage_to_estimate=True,
        raise_on_invalid_peak=True,
        diagnostic_output_path=None,
        position_index=None,
        edge_retries=0,
        edge_widen_factor=2.0,
    ):
        """
        Perform autofocus using specified score metric.

        Args:
            n_steps: Number of Z positions to sample
            search_range: Total Z range to search in micrometers
            interp_strength: Interpolation density factor
            interp_kind: Type of interpolation ('linear', 'quadratic', 'cubic')
            score_metric: Function to score image focus
            pop_a_plot: Whether to show a focus score plot
            move_stage_to_estimate: Whether to move to best focus position
            raise_on_invalid_peak: If True (default), raises RuntimeError when peak validation fails,
                                  stopping acquisitions. If False, logs warning and continues
                                  (use for diagnostic tests where plots must always be generated).
            diagnostic_output_path: Optional path to save autofocus diagnostic CSV (for standard autofocus during acquisition)
            position_index: Optional position index to include in CSV filename
            edge_retries: When > 0 and the validated peak lands at an edge of
                          the swept range with a directional trend
                          (asc XOR desc), shift the sweep center toward the
                          inferred peak direction and re-sweep with
                          ``search_range *= edge_widen_factor``. Same algorithm
                          as ``autofocus_sweep_drift_check``'s edge-retry,
                          applied to the dense full-AF path. Sweeps are clamped
                          to the configured stage Z limits; if clamping leaves
                          no new ground to search, the retry is skipped.
            edge_widen_factor: Multiplier applied to search_range on each edge retry.

        Returns:
            float: Best focus Z position (in micrometers) on success
            dict: Failure information dict with keys:
                - 'success': False
                - 'message': Error message
                - 'quality_score': Focus quality score
                - 'peak_prominence': Peak prominence value
                - 'attempted_z': Z position that was attempted
                - 'original_z': Original Z position before autofocus
                - 'validation': Full validation dict
        """
        # Stop streaming -- snap_image() cannot run during sequence acquisition
        self.camera.stop_if_streaming()

        from microscope_control.autofocus.core import AutofocusUtils

        # Read stage Z limits once for edge-retry clamping. Same accessor as
        # sweep_focus / autofocus_sweep_drift_check.
        stage_limits = self.settings.get("stage", {}).get("limits", {}).get("z_um", {})
        z_min_cfg = stage_limits.get("low")
        z_max_cfg = stage_limits.get("high")

        # Capture the original starting position once. Used for the failure dict
        # 'original_z' field and as the unwind target if we throw.
        starting_pos = self.get_current_position()
        edge_retries_used = 0
        # Mutable per-attempt state -- start at the user-requested center/range.
        attempt_center = starting_pos.z
        attempt_range = float(search_range)

        try:
            while True:
                steps = np.linspace(0, attempt_range, n_steps) - (attempt_range / 2)
                current_pos = Position(starting_pos.x, starting_pos.y, attempt_center)
                z_steps = attempt_center + steps

                scores = []
                fallback_scores = []  # p98_p2 computed alongside primary metric
                for step_number in range(n_steps):
                    new_pos = Position(current_pos.x, current_pos.y, attempt_center + steps[step_number])
                    self.move_to_position(new_pos)

                    img, tags = self.snap_image()

                    # Extract green/grayscale channel for focus calculation
                    # (camera-specific: JAI uses mean across RGB, Bayer extracts green pixels)
                    img_gray = self.camera.extract_green_channel(img)

                    score = score_metric(img_gray)
                    if hasattr(score, "ndim") and score.ndim == 2:
                        score = np.mean(score)
                    scores.append(score)

                    # Always compute p98_p2 as fallback (negligible cost, no extra acquisition)
                    p98_p2 = float(np.percentile(img_gray, 98) - np.percentile(img_gray, 2))
                    fallback_scores.append(p98_p2)

                # VALIDATE FOCUS PEAK QUALITY
                scores_array = np.array(scores)
                validation = AutofocusUtils.validate_focus_peak(z_steps, scores_array)

                # Interpolate to find best focus (do this before validation check).
                # np.clip prevents the interpolated argmax from landing
                # outside the actually-sampled Z range -- without it, a
                # cubic/quadratic curve fit can extrapolate the maximum
                # beyond [z_steps[0], z_steps[-1]] (e.g. a Z=1944.41
                # result on a sweep that only sampled 1943..2017), and
                # the stage then moves to a Z position whose focus
                # quality was never actually measured.
                interp_x = np.linspace(z_steps[0], z_steps[-1], n_steps * interp_strength)
                interp_y = scipy.interpolate.interp1d(z_steps, scores, kind=interp_kind)(interp_x)
                new_z = float(np.clip(interp_x[np.argmax(interp_y)], z_steps[0], z_steps[-1]))

                # Save diagnostic CSV BEFORE validation check (so it saves even on failure)
                if diagnostic_output_path is not None:
                    self._save_autofocus_diagnostic_csv(
                        z_steps, scores_array, validation, new_z,
                        diagnostic_output_path, position_index, current_pos
                    )

                # Edge-retry has priority over p98_p2 fallback. When the
                # primary metric shows a clear peak with rising trend that
                # is cut off at the search-window edge, the right answer is
                # to extend the search in that direction rather than swap
                # in a different metric (which previously produced
                # silently-different focus answers; see commit history).
                # Trigger an edge-retry whenever:
                #   - we have retries left, AND
                #   - the validator set should_extend_direction (one-sided
                #     confirmed trend; the other side is unconfirmed
                #     because too few samples or peak landed at the edge),
                #     AND
                #   - the stage Z limits leave room to extend.
                if (edge_retries_used < edge_retries
                        and validation.get('should_extend_direction') is not None):
                    retry_center, retry_range = self._compute_edge_retry_window(
                        validation=validation,
                        cur_center=attempt_center,
                        cur_range=attempt_range,
                        widen_factor=float(edge_widen_factor),
                        z_min=z_min_cfg,
                        z_max=z_max_cfg,
                    )
                    if retry_center is not None:
                        edge_retries_used += 1
                        logger.info(
                            "  Edge retry %d/%d: one-sided trend (extend %s); "
                            "center %.2f -> %.2f um, range %.1f -> %.1f um "
                            "(window [%.2f, %.2f])",
                            edge_retries_used, edge_retries,
                            validation['should_extend_direction'],
                            attempt_center, retry_center,
                            attempt_range, retry_range,
                            retry_center - retry_range / 2.0,
                            retry_center + retry_range / 2.0,
                        )
                        attempt_center = retry_center
                        attempt_range = retry_range
                        continue

                if not validation['is_valid']:
                    logger.warning("*** AUTOFOCUS PEAK QUALITY WARNING ***")
                    logger.warning(f"  {validation['message']}")
                    for warning in validation['warnings']:
                        logger.warning(f"    - {warning}")
                    logger.warning(f"  Quality metrics: prominence={validation['peak_prominence']:.2f}, "
                                 f"quality={validation['quality_score']:.2f}")

                    # Try p98_p2 fallback (already computed, no re-acquisition).
                    # Only reached when the primary metric has no usable peak
                    # at all -- not when the peak is good but at the edge,
                    # which is handled above via edge-retry on the primary.
                    if fallback_scores:
                        fallback_array = np.array(fallback_scores)
                        fallback_validation = AutofocusUtils.validate_focus_peak(z_steps, fallback_array)
                        if fallback_validation['is_valid']:
                            fallback_interp_y = scipy.interpolate.interp1d(
                                z_steps, fallback_scores, kind=interp_kind)(interp_x)
                            fallback_z = float(np.clip(
                                interp_x[np.argmax(fallback_interp_y)],
                                z_steps[0], z_steps[-1]))
                            logger.info(f"  p98_p2 fallback found valid peak at Z={fallback_z:.2f} um "
                                        f"(quality={fallback_validation['quality_score']:.2f})")
                            new_z = fallback_z
                            validation = fallback_validation
                        else:
                            logger.warning(f"  p98_p2 fallback also invalid: {fallback_validation['message']}")

                    if not validation['is_valid'] and raise_on_invalid_peak:
                        # Return failure dict for manual focus fallback loop
                        logger.warning("  Autofocus failed - moving to computed best Z and returning failure dict")
                        best_pos = Position(current_pos.x, current_pos.y, new_z)
                        self.move_to_position(best_pos)
                        return {
                            'success': False,
                            'message': validation['message'],
                            'quality_score': validation['quality_score'],
                            'peak_prominence': validation['peak_prominence'],
                            'attempted_z': new_z,
                            'original_z': starting_pos.z,
                            'validation': validation
                        }
                    elif not validation['is_valid']:
                        # Test mode: log and proceed for diagnostic purposes
                        logger.warning("  Proceeding with autofocus result for diagnostic analysis")
                else:
                    logger.info(f"Autofocus peak validation: {validation['message']}")
                    logger.debug(f"  Quality score: {validation['quality_score']:.2f}, "
                               f"prominence: {validation['peak_prominence']:.2f}")

                if pop_a_plot:
                    plt.figure()
                    plt.bar(z_steps, scores)
                    plt.plot(interp_x, interp_y, "k")
                    plt.plot(interp_x[np.argmax(interp_y)], interp_y.max(), "or")
                    plt.xlabel("Z-axis (um)")
                    plt.title(f"Autofocus at X={current_pos.x:.1f}, Y={current_pos.y:.1f}")
                    plt.show()

                if move_stage_to_estimate:
                    new_pos = Position(current_pos.x, current_pos.y, new_z)
                    self.move_to_position(new_pos)

                return new_z

        except Exception as e:
            logger.error(f"Autofocus failed: {e}")
            self.move_to_position(starting_pos)
            raise e

    def _compute_edge_retry_window(
        self,
        validation,
        cur_center,
        cur_range,
        widen_factor,
        z_min,
        z_max,
        margin=5.0,
    ):
        """Compute (new_center, new_range) for a peak-at-edge AF retry.

        Returns ``(None, None)`` when no retry should be attempted: either the
        failure isn't a directional edge, stage Z limits aren't configured, or
        the clamped new sweep wouldn't cover any ground outside the band the
        previous sweep already covered.
        """
        has_asc = bool(validation.get('has_ascending', False))
        has_desc = bool(validation.get('has_descending', False))
        # Directional edge: asc XOR desc. asc-only -> peak at end (search higher);
        # desc-only -> peak at start (search lower). Both-or-neither isn't a
        # directional edge -- there's no informative direction to shift.
        if has_asc == has_desc:
            return None, None
        if z_min is None or z_max is None:
            logger.warning(
                "  Edge retry: stage Z limits not configured; skipping"
            )
            return None, None

        new_range = float(cur_range) * float(widen_factor)
        shift_sign = 1 if (has_asc and not has_desc) else -1
        target_center = float(cur_center) + shift_sign * (float(cur_range) / 2.0)

        available = (z_max - margin) - (z_min + margin)
        if available <= 0:
            logger.warning(
                "  Edge retry: stage Z limits leave no usable sweep band "
                "(z_min=%.1f, z_max=%.1f, margin=%.1f); skipping",
                z_min, z_max, margin,
            )
            return None, None
        if available < new_range:
            new_range = available
        half = new_range / 2.0
        clamped_center = max(
            z_min + margin + half,
            min(z_max - margin - half, target_center),
        )

        # Sanity: did clamping leave us with a window that actually extends
        # in the requested direction? If stage Z limits prevent shifting the
        # center toward the trend, the clamp can pull the center the
        # OPPOSITE way (e.g. "extend low" requested but z_min forces center
        # UP), producing a wider window that scans new ground in the WRONG
        # direction. That's worse than refusing -- on 2026-05-05 a clamp
        # inversion walked the stage from -68 to -15 because the widened
        # window hit a junk peak at the high edge.
        new_low = clamped_center - half
        new_high = clamped_center + half
        old_low = float(cur_center) - float(cur_range) / 2.0
        old_high = float(cur_center) + float(cur_range) / 2.0
        if shift_sign < 0:
            extended_in_direction = new_low < old_low - 1e-3
            direction_label = "low"
        else:
            extended_in_direction = new_high > old_high + 1e-3
            direction_label = "high"
        if not extended_in_direction:
            logger.warning(
                "  Edge retry: stage Z limits prevent extending toward %s "
                "(would clamp window to [%.2f, %.2f] um vs prior [%.2f, %.2f]); "
                "skipping",
                direction_label, new_low, new_high, old_low, old_high,
            )
            return None, None

        return clamped_center, new_range

    def _save_autofocus_diagnostic_csv(self, z_positions, scores, validation, result_z,
                                       output_path, position_index, current_pos):
        """
        Save autofocus diagnostic data to CSV file in the acquisition folder.

        Args:
            z_positions: Array of Z positions sampled
            scores: Array of focus scores
            validation: Validation results dict
            result_z: Resulting Z position from autofocus
            output_path: Path to save the CSV file
            position_index: Position index for filename
            current_pos: Starting position for autofocus
        """
        try:
            import csv
            from datetime import datetime
            from pathlib import Path

            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pos_str = f"_pos{position_index}" if position_index is not None else ""
            csv_filename = f"autofocus_diagnostic{pos_str}_{timestamp}.csv"
            csv_path = output_path / csv_filename

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header with metadata
                writer.writerow(['# Autofocus Diagnostic Data (Standard Autofocus)'])
                writer.writerow(['# Timestamp', timestamp])
                if position_index is not None:
                    writer.writerow(['# Position Index', position_index])
                writer.writerow(['# Starting Position', f'X={current_pos.x:.2f}, Y={current_pos.y:.2f}, Z={current_pos.z:.2f}'])
                writer.writerow(['# Autofocus Result Z', f'{result_z:.2f}'])
                writer.writerow(['# Z Shift', f'{result_z - current_pos.z:.2f}'])
                writer.writerow(['#'])
                writer.writerow(['# VALIDATION RESULTS'])
                writer.writerow(['# Status', 'VALID' if validation['is_valid'] else 'INVALID'])
                writer.writerow(['# Quality Score', f"{validation['quality_score']:.3f}"])
                writer.writerow(['# Peak Prominence', f"{validation['peak_prominence']:.3f}"])
                writer.writerow(['# Has Ascending', validation['has_ascending']])
                writer.writerow(['# Has Descending', validation['has_descending']])
                writer.writerow(['# Symmetry Score', f"{validation['symmetry_score']:.3f}"])
                writer.writerow(['# Message', validation['message']])
                if validation['warnings']:
                    for warning in validation['warnings']:
                        writer.writerow(['# Warning', warning])
                writer.writerow(['#'])

                # Write data header
                writer.writerow(['Z_Position_um', 'Focus_Score'])

                # Write data
                for z, score in zip(z_positions, scores):
                    writer.writerow([f'{z:.2f}', f'{score:.4f}'])

            logger.info(f"Autofocus diagnostic CSV saved: {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save autofocus diagnostic CSV: {e}")

    def _score_single_metric(self, pixels, img_w, img_h, nch, cy, cx,
                             metric_name="normalized_variance"):
        """Score a frame with a single focus metric on center crop.

        Crop size: min(512, image_dim) to handle cameras smaller than 512px.
        Selects the best non-saturated channel for multi-channel images.
        Saturation threshold is derived from image dtype (8-bit: 245, 16-bit: ~64000).

        Args:
            pixels: Raw pixel buffer from core.get_tagged_image().pix
            img_w: Image width in pixels
            img_h: Image height in pixels
            nch: Number of color channels
            cy: Center Y coordinate for crop
            cx: Center X coordinate for crop
            metric_name: Canonical metric name from
                focus_metrics_manifest.yml (e.g. "tenengrad",
                "laplacian_variance", "normalized_variance",
                "brenner_gradient", "sobel", "p98_p2"). Looked up via
                ``microscope_imageprocessing.focus.resolve_metric``.

        Returns:
            (score, ch_mean) tuple -- the focus score and channel mean intensity.
        """
        # Derive saturation threshold from image dtype
        if pixels.dtype == np.uint16:
            sat_threshold = 64000.0
        elif pixels.dtype == np.uint8:
            sat_threshold = 245.0
        else:
            sat_threshold = 245.0  # fallback for other dtypes

        # Crop size: use 512x512 or smaller if image is smaller
        crop_half = min(256, cy, cx, img_h - cy, img_w - cx)

        if nch > 1:
            img = pixels.reshape(img_h, img_w, nch)
            # Find best channel (non-saturated, prefer B then G then R)
            best_ch = None
            for ch in [0, 1, 2]:  # B, G, R in BGRA
                roi_test = img[cy - crop_half:cy + crop_half, cx - crop_half:cx + crop_half, ch]
                if float(np.mean(roi_test)) < sat_threshold:
                    best_ch = ch
                    break
            if best_ch is None:
                best_ch = 1  # fallback to green
            roi = img[cy - crop_half:cy + crop_half, cx - crop_half:cx + crop_half, best_ch].astype(np.float32)
        else:
            roi = pixels.reshape(img_h, img_w)[cy - crop_half:cy + crop_half, cx - crop_half:cx + crop_half].astype(np.float32)

        ch_mean = float(np.mean(roi))

        # Dispatch via the consolidated registry. The dispatcher raises
        # UnknownMetricError on misses; let it propagate so a typo in
        # caller config surfaces immediately rather than silently
        # substituting a different metric.
        fn = resolve_metric(metric_name)
        score = float(fn(roi))

        return score, ch_mean

    def autofocus_sweep_drift_check(
        self,
        range_um=10.0,
        n_steps=5,
        score_metric="normalized_variance",
        max_retries=2,
    ) -> float:
        """Drift check using stepped Z sweep with blocking moves and snaps.

        Uses direct core.snap_image() at each Z position (no continuous
        acquisition -- buffer frame freshness was unreliable). Scores
        with the specified metric on center-crop of the best non-saturated
        channel.

        When the peak is at a boundary (monotonic profile), retries up to
        max_retries additional times shifting the window in the peak
        direction. Total coverage is up to (max_retries+1) * range_um,
        clamped to stage Z limits.

        If sweep fails for any reason, returns starting Z unchanged.

        Args:
            range_um: Total sweep range in micrometers (default 10 = +/-5um)
            n_steps: Number of Z positions to sample (default 5 -> 6 pts)
            score_metric: Focus metric name passed to _score_single_metric().
                One of "normalized_variance" (default), "laplacian_variance",
                "sobel", "brenner_gradient", "p98_p2".
            max_retries: Additional sweep attempts on boundary peaks (default 2).
                0 disables retries. Each retry extends range by one window.

        Returns:
            The best-focus Z position (or current Z if sweep failed).
        """

        self.camera.stop_if_streaming()

        initial_z = self.get_z_position()

        stage_limits = self.settings.get("stage", {}).get("limits", {})
        z_limits = stage_limits.get("z_um", {})
        if "low" not in z_limits or "high" not in z_limits:
            raise ValueError(
                "Z stage limits (stage.limits.z_um.low and "
                "stage.limits.z_um.high) not specified in microscope "
                "config YAML. These are required to prevent stage "
                "collisions during autofocus sweeps."
            )
        z_min = z_limits["low"]
        z_max = z_limits["high"]

        half = range_um / 2.0

        stage_config = self.settings.get("stage", {})
        z_stage_device = stage_config.get("z_stage", None)
        if z_stage_device and self.core.get_focus_device() != z_stage_device:
            self.core.set_focus_device(z_stage_device)
        z_dev = z_stage_device or self.core.get_focus_device()

        img_w = self.core.get_image_width()
        img_h = self.core.get_image_height()
        nch = self.core.get_number_of_components()
        cy, cx = img_h // 2, img_w // 2

        # Tolerance for "did the stage actually arrive at the commanded
        # Z?" diagnostic. Tight enough to catch real misses, loose
        # enough to absorb encoder noise.
        arrival_tol_um = 0.5

        def _sweep_one_window(z_positions_list):
            """Run pipelined snap+move sweep, return [(actual_z, score)].

            Holds the stage lock for the whole sweep. Without this, the
            StagePositionCache poll thread (500 ms get_xyz polls) competes
            with set_position on the same serial bus and can stall moves --
            the 2026-05-05 user report ("sweep and streaming used to
            work, now broken; both at the same time") fits this exact
            failure mode since both bypass the lock added in commit
            e7f1467 (Apr 15) at the same time the cache thread was added
            in fe6f5ee.
            """
            result = []
            stage_lock = getattr(self._stage, "lock", None)
            if stage_lock is not None:
                stage_lock.acquire()
            try:
                try:
                    self.core.set_position(z_positions_list[0])
                    self.core.wait_for_device(z_dev)
                    arrival_misses = 0
                    for i in range(len(z_positions_list)):
                        target_now = float(z_positions_list[i])
                        actual_z = self.core.get_position()
                        # Arrival check on every step. Logged at WARNING
                        # only when off-target so a healthy sweep stays
                        # quiet.
                        err = abs(actual_z - target_now)
                        if err > arrival_tol_um:
                            arrival_misses += 1
                            logger.warning(
                                f"Sweep step {i}: stage arrival miss -- "
                                f"target={target_now:.3f}, "
                                f"actual={actual_z:.3f}, err={err:.3f} um "
                                f"(tol={arrival_tol_um:.2f})"
                            )
                        self.core.snap_image()
                        if i < len(z_positions_list) - 1:
                            self.core.set_position(z_positions_list[i + 1])
                        tagged = self.core.get_tagged_image()
                        pixels = tagged.pix
                        sc, _ = self._score_single_metric(
                            pixels, img_w, img_h, nch, cy, cx, score_metric)
                        if i < len(z_positions_list) - 1:
                            self.core.wait_for_device(z_dev)
                        result.append((actual_z, sc))
                    if arrival_misses > 0:
                        logger.warning(
                            f"Sweep drift: {arrival_misses}/"
                            f"{len(z_positions_list)} steps had stage "
                            f"arrival errors > {arrival_tol_um:.2f} um. "
                            f"Sweep results may be unreliable."
                        )
                except Exception as e:
                    logger.warning(f"Sweep window failed at step: {e}")
            finally:
                if stage_lock is not None:
                    stage_lock.release()
            return result

        def _make_z_positions(center):
            """Build clamped Z position list for a sweep centered at center."""
            s = max(center - half, z_min + 5)
            e = min(center + half, z_max - 5)
            rng = e - s
            if rng < 1.0:
                return []
            stp = rng / n_steps
            return [s + i * stp for i in range(n_steps + 1)]

        # --- First attempt ---
        current_center = initial_z
        z_positions = _make_z_positions(current_center)
        if not z_positions:
            logger.warning("Sweep drift check: range too small, keeping current Z")
            return initial_z

        try:
            exposure = self.core.get_exposure()
            logger.info(
                f"Sweep drift check: [{z_positions[0]:.1f} -> "
                f"{z_positions[-1]:.1f}] "
                f"step={z_positions[1]-z_positions[0]:.1f}um, "
                f"range={z_positions[-1]-z_positions[0]:.1f}um, "
                f"current={initial_z:.1f}, exposure={exposure:.2f}ms, "
                f"img={img_w}x{img_h}x{nch}")
        except Exception:
            logger.info(
                f"Sweep drift check: [{z_positions[0]:.1f} -> "
                f"{z_positions[-1]:.1f}], current={initial_z:.1f}")

        t0 = time.perf_counter()
        total_pts = 0

        measurements = _sweep_one_window(z_positions)
        total_pts += len(measurements)
        elapsed = (time.perf_counter() - t0) * 1000

        if len(measurements) < 3:
            logger.warning(
                f"Sweep drift check: only {len(measurements)} "
                f"measurements in {elapsed:.0f}ms, keeping current Z")
            self.core.set_position(initial_z)
            self.core.wait_for_device(z_dev)
            return initial_z

        z_arr = np.array([m[0] for m in measurements])
        scores = np.array([m[1] for m in measurements])
        best_idx = int(np.argmax(scores))

        # --- Edge-retry loop (up to max_retries additional attempts) ---
        for retry in range(max_retries):
            if best_idx != 0 and best_idx != len(z_arr) - 1:
                break

            sr = float(scores.max() - scores.min())
            sr_pct = sr / max(float(scores.mean()), 1.0) * 100
            if sr_pct < 2.0:
                break

            boundary_z = z_arr[best_idx]
            if best_idx == len(z_arr) - 1:
                new_center = boundary_z + half
            else:
                new_center = boundary_z - half

            ext_positions = _make_z_positions(new_center)
            if not ext_positions:
                logger.info(
                    f"Sweep retry {retry+1}: window would exceed Z limits, "
                    f"stopping")
                break

            logger.info(
                f"Sweep drift check: peak at boundary (idx={best_idx}), "
                f"score trend {sr_pct:.1f}% -- retry {retry+1} "
                f"[{ext_positions[0]:.1f} -> {ext_positions[-1]:.1f}]")

            ext_measurements = _sweep_one_window(ext_positions)
            total_pts += len(ext_measurements)

            if len(ext_measurements) < 3:
                break

            z_arr = np.array([m[0] for m in ext_measurements])
            scores = np.array([m[1] for m in ext_measurements])
            best_idx = int(np.argmax(scores))

        elapsed = (time.perf_counter() - t0) * 1000

        # --- Evaluate final result ---
        if best_idx == 0 or best_idx == len(z_arr) - 1:
            # Peak is at boundary even after retries -- the metric is not
            # bracketing focus inside any window we tried. Previously
            # the code moved to the boundary value on the assumption
            # "it's better than initial_z even if we didn't find a true
            # peak", which silently walked the stage off-focus when the
            # operator had pre-focused (e.g. OWS3 BF 10x ground-truth
            # Z=2003 -> sweep accepted Z=1944, 75 um below). Stay at
            # initial_z instead: an already-good manual focus must not
            # be overwritten by an edge-peak guess.
            boundary_z = float(z_arr[best_idx])
            logger.warning(
                f"Sweep drift check: peak at boundary after all attempts "
                f"(boundary Z={boundary_z:.2f} vs initial Z={initial_z:.2f}). "
                f"Metric did not bracket focus -- holding at initial Z "
                f"({elapsed:.0f}ms)")
            self.core.set_position(initial_z)
            self.core.wait_for_device(z_dev)
            return initial_z

        # U-shape rejection: when the score curve has a clear interior
        # minimum with edges elevated near the global max, the metric is
        # contrast-inverted at this XY (saturation, debris, polarizer
        # artifact). The "best" interior idx is then just a noise bump
        # on the side of the U; moving there walks the stage AWAY from
        # focus. Hold at initial_z instead. See validate_focus_peak() in
        # autofocus/core.py for the same logic on the standard AF path.
        if len(scores) >= 5:
            min_idx_local = int(np.argmin(scores))
            if 0 < min_idx_local < len(scores) - 1:
                edge_score = float(min(scores[0], scores[-1]))
                full_range = float(scores.max() - scores.min())
                if full_range > 0:
                    valley_depth = (edge_score - float(scores[min_idx_local])) / full_range
                    if valley_depth >= 0.5:
                        logger.warning(
                            f"Sweep drift check: U-shape detected (interior "
                            f"min at Z={float(z_arr[min_idx_local]):.2f}, edges "
                            f"{float(scores[0]):.2f}/{float(scores[-1]):.2f}, "
                            f"valley_depth={valley_depth:.2f}) -- metric is "
                            f"contrast-inverted, holding at initial Z "
                            f"({elapsed:.0f}ms)")
                        self.core.set_position(initial_z)
                        self.core.wait_for_device(z_dev)
                        return initial_z

        start_idx = int(np.argmin(np.abs(z_arr - initial_z)))
        start_score = scores[start_idx]
        peak_score = scores[best_idx]
        improvement = (peak_score - start_score) / max(start_score, 1.0)
        score_range = float(scores.max() - scores.min())
        score_range_pct = score_range / max(float(scores.mean()), 1.0) * 100

        if score_range_pct < 2.0:
            logger.info(
                f"Sweep drift check: score range {score_range_pct:.1f}% "
                f"< 2% -- metric not discriminating focus, "
                f"keeping current Z ({elapsed:.0f}ms)")
            self.core.set_position(initial_z)
            self.core.wait_for_device(z_dev)
            return initial_z

        best_z = z_arr[best_idx]

        z0, z1, z2 = z_arr[best_idx - 1], z_arr[best_idx], z_arr[best_idx + 1]
        s0, s1, s2 = scores[best_idx - 1], scores[best_idx], scores[best_idx + 1]
        denom = 2.0 * (s0 - 2.0 * s1 + s2)
        if abs(denom) > 1e-6:
            z_peak = z1 - (s2 - s0) * (z2 - z0) / (2.0 * denom)
            if min(z0, z2) <= z_peak <= max(z0, z2):
                best_z = z_peak

        drift = best_z - initial_z
        self.core.set_position(best_z)
        self.core.wait_for_device(z_dev)

        logger.info(
            f"Sweep drift check: {drift:+.2f}um, improvement "
            f"{improvement:.1%} ({total_pts} pts in {elapsed:.0f}ms)")
        return best_z

    def white_balance(self, img=None, background_image=None, gain=1.0,
                      white_balance_profile=None):
        """Apply software white balance (delegates to camera with settings)."""
        return self.camera.white_balance(
            img, background_image=background_image, gain=gain,
            white_balance_profile=white_balance_profile, settings=self.settings,
        )

    def get_device_properties(self, scope: str = "used") -> Dict[str, Dict[str, Any]]:
        """
        Get device properties from MM device manager.

        Args:
            scope: 'used' for current values, 'allowed' for possible values

        Returns:
            Dictionary of device properties
        """
        device_dict = {}
        for device_name in obj_2_list(self.core.get_loaded_devices()):
            device_property_names = self.core.get_device_property_names(device_name)
            property_names = obj_2_list(device_property_names)
            prop_dict = {}

            for prop in property_names:
                if scope == "allowed":
                    values = self.core.get_allowed_property_values(device_name, prop)
                    prop_dict[prop] = obj_2_list(values)
                elif scope == "used":
                    values = self.core.get_property(device_name, prop)
                    prop_dict[prop] = values
                else:
                    warnings.warn(f"Unknown metadata scope {scope}")

            device_dict[device_name] = prop_dict

        return device_dict

    # Rotation stage methods (set_psg_ticks, get_psg_ticks, etc.) are now
    # delegated through MicroscopeHardware -> self._rotation_stage.
    # See hardware/rotation.py for PIZRotationStage, ThorRotationStage, DummyRotationStage.

    def swap_objective(self, target_profile: str) -> None:
        """Swap objective lens using config-driven sequences.

        Reads ``objective_swap_sequences`` from settings to determine the
        safe order of operations for switching objectives. Different
        objectives may require different Z/turret/F staging sequences
        (e.g., low-mag objectives can move Z first, high-mag must move
        the turret first to avoid collisions).

        Config example::

            objective_swap_sequences:
              low_mag:
                objectives: ['OBJ_4X_001']
                sequence:
                  - {action: set_focus_device, device_key: z_stage}
                  - {action: move_position, device_key: z_stage, value_key: z}
                  - {action: set_turret}
                  - {action: set_focus_device, device_key: f_stage}
              high_mag:
                objectives: ['OBJ_20X_001']
                sequence:
                  - {action: set_turret}
                  - {action: set_focus_device, device_key: z_stage}
                  - {action: move_position, device_key: z_stage, value_key: z}
                  - {action: set_focus_device, device_key: f_stage}
                  - {action: move_position, device_key: f_stage, value_key: f}

        Args:
            target_profile: Acquisition profile name whose objective to switch to.
        """
        swap_sequences = self.settings.get("objective_swap_sequences")
        if not swap_sequences:
            logger.debug("No objective_swap_sequences in config, skipping")
            return

        profiles = self.settings.get("acquisition_profiles", {})
        profile = profiles.get(target_profile, {})
        target_objective = profile.get("objective")
        if not target_objective:
            logger.debug("No objective in profile '%s'", target_profile)
            return

        # Find which sequence group this objective belongs to
        sequence_steps = None
        for group_name, group_config in swap_sequences.items():
            if not isinstance(group_config, dict):
                continue
            obj_list = group_config.get("objectives", [])
            if target_objective in obj_list:
                sequence_steps = group_config.get("sequence", [])
                logger.info("Objective swap: using '%s' sequence for %s",
                            group_name, target_objective)
                break

        if sequence_steps is None:
            logger.debug("Objective '%s' not in any swap sequence, skipping",
                         target_objective)
            return

        # Get turret device and desired position from config
        obj_slider = self.settings.get("obj_slider")
        if not obj_slider:
            logger.warning("No obj_slider config, cannot swap objective")
            return

        # Check if turret is already at the right position
        mode_positions = self.settings.get("mode_positions", {})
        mode_pos = mode_positions.get(target_profile, {})
        desired_turret_label = mode_pos.get("turret_label")

        if desired_turret_label:
            current_label = self.core.get_property(*obj_slider)
            if current_label == desired_turret_label:
                logger.info("Objective already at %s, no swap needed",
                            desired_turret_label)
                return

        # Execute the sequence
        stage_config = self.settings.get("stage", {})
        for step in sequence_steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            device_key = step.get("device_key")

            if action == "set_focus_device" and device_key:
                device = stage_config.get(device_key)
                if device:
                    self.core.set_focus_device(device)
                    logger.debug("Focus device set to %s", device)

            elif action == "move_position" and device_key:
                device = stage_config.get(device_key)
                value_key = step.get("value_key")
                value = mode_pos.get(value_key, 0) if value_key else 0
                if device:
                    self.core.set_position(value)
                    self.core.wait_for_device(device)
                    logger.debug("Moved %s to %.1f", device, value)

            elif action == "set_turret":
                if desired_turret_label and obj_slider:
                    self.core.set_property(*obj_slider, desired_turret_label)
                    self.core.wait_for_device(obj_slider[0])
                    logger.debug("Turret set to %s", desired_turret_label)

            elif action == "wait":
                self.core.wait_for_system()

        # Always restore Z as the focus device
        z_stage = stage_config.get("z_stage")
        if z_stage:
            self.core.set_focus_device(z_stage)

        logger.info("Objective swap complete for profile: %s", target_profile)
