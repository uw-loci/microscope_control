"""Per-camera values in one microscope config: the ``per_detector`` overlay.

A microscope with two cameras keeps one config. Values that depend on the camera
(exposures, a black level, LC retardances, the Micro-Manager preset group of a camera
that runs its own Micro-Manager config) sit in a ``per_detector`` mapping beside the
shared values, keyed by detector ID::

    reconstruction:
      black_level: 51
      per_detector:
        LCPS_DETECTOR_FLIR_ORYX_001: {black_level: 99}
        LCPS_DETECTOR_HAMAMATSU_FLASH_001: {black_level: 51}

:func:`apply_detector_overlay` merges the active camera's entry into its parent
mapping everywhere in the tree. QPSC applies the same rules
(``MicroscopeConfigManager`` in qupath-extension-qpsc), and
``microscope_configurations/checks/per_detector.py`` carries a copy of this module's
function; the three must stay identical.

Rules, for every mapping that has a ``per_detector`` key whose value is a mapping, when
``detector_id`` is not None and ``per_detector`` has that exact key with a mapping value
``M``, for each key ``k`` of ``M``:

* ``k == "device_properties"`` and both sides are lists: merged by
  ``(device, property)``. An entry of ``M`` replaces the existing entry with the same
  pair in place; new pairs are appended in ``M``'s order.
* both values are mappings: merged recursively with these same rules.
* otherwise the value is replaced.

The ``per_detector`` key itself stays in place. After a mapping is merged, the walk
continues into its values (mappings, and lists of mappings). A detector of None, or one
a node's ``per_detector`` does not name, leaves that node unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

PER_DETECTOR_KEY = "per_detector"
DEVICE_PROPERTIES_KEY = "device_properties"


def apply_detector_overlay(config: Any, detector_id: Optional[str]) -> Any:
    """A deep copy of ``config`` with ``detector_id``'s ``per_detector`` entries merged in.

    The input is never modified. See the module docstring for the rules.
    """
    result = deepcopy(config)
    _walk(result, detector_id)
    return result


def _walk(node: Any, detector_id: Optional[str]) -> None:
    if isinstance(node, dict):
        if detector_id is not None:
            per_detector = node.get(PER_DETECTOR_KEY)
            if isinstance(per_detector, dict):
                entry = per_detector.get(detector_id)
                if isinstance(entry, dict):
                    _merge_into(node, deepcopy(entry))
        for value in node.values():
            _walk(value, detector_id)
    elif isinstance(node, list):
        for item in node:
            _walk(item, detector_id)


def _merge_into(target: dict, overlay: dict) -> None:
    for key, value in overlay.items():
        current = target.get(key)
        if key == DEVICE_PROPERTIES_KEY and isinstance(current, list) and isinstance(value, list):
            target[key] = _merge_device_properties(current, value)
        elif isinstance(current, dict) and isinstance(value, dict):
            _merge_into(current, value)
        else:
            target[key] = value


def _property_key(entry: Any):
    if isinstance(entry, dict):
        return (entry.get("device"), entry.get("property"))
    return None


def _merge_device_properties(base: list, overlay: list) -> list:
    merged = list(base)
    index = {}
    for i, entry in enumerate(merged):
        key = _property_key(entry)
        if key is not None and key not in index:
            index[key] = i
    for entry in overlay:
        key = _property_key(entry)
        if key is not None and key in index:
            merged[index[key]] = entry
        else:
            if key is not None:
                index[key] = len(merged)
            merged.append(entry)
    return merged
