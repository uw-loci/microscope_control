"""Choosing an ``acquisition_profiles`` entry for a modality, objective and camera.

A microscope with two cameras keeps one config, with one profile set per camera (each
profile's ``detector`` field names its camera). :func:`resolve_profile_key` is the one
rule for picking a profile; QPSC's ``MicroscopeConfigManager.resolveProfileKey``
(qupath-extension-qpsc) applies the same rule, and the two must stay identical so the
``--profile`` key QPSC sends and the one the server derives agree.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


def _candidates(profiles: Mapping[str, Any], detector: Optional[str]) -> list:
    """Profile keys usable with ``detector``, in file order.

    Every profile when ``detector`` is None; otherwise the profiles whose ``detector``
    equals it plus those with no detector (the key absent or null).
    """
    keys = []
    for key, profile in profiles.items():
        if detector is None:
            keys.append(key)
            continue
        profile_detector = profile.get("detector") if isinstance(profile, dict) else None
        if profile_detector is None or profile_detector == detector:
            keys.append(key)
    return keys


def resolve_profile_key(
    profiles: Optional[Mapping[str, Any]],
    modality: Optional[str],
    objective: Optional[str],
    detector: Optional[str] = None,
) -> Optional[str]:
    """The ``acquisition_profiles`` key for (modality, objective, detector), or None.

    1. Candidates: every profile when ``detector`` is None, otherwise the profiles whose
       ``detector`` field equals it plus the profiles with no ``detector`` field.
    2. Among the candidates, the first profile (file order) whose ``modality`` equals
       ``modality`` ignoring case and whose ``objective`` equals ``objective`` exactly.
    3. Otherwise the name rules, over the candidates only: a key equal to ``modality``;
       then the longest ``<modality>_<suffix>`` whose suffix is a case-insensitive
       substring of the objective ID (the first in file order on a tie); then the first
       key that starts with ``<modality>_`` ignoring case and whose suffix is such a
       substring.
    """
    if not modality or not profiles:
        return None
    keys = _candidates(profiles, detector)
    modality_lower = modality.lower()

    for key in keys:
        profile = profiles[key]
        if not isinstance(profile, dict):
            continue
        prof_modality = profile.get("modality")
        if (
            isinstance(prof_modality, str)
            and prof_modality.lower() == modality_lower
            and objective is not None
            and profile.get("objective") == objective
        ):
            return key

    if modality in keys:
        return modality

    prefix = modality + "_"
    objective_lower = (objective or "").lower()
    best_key = None
    best_len = -1
    for key in keys:
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix and suffix.lower() in objective_lower and len(suffix) > best_len:
            best_len = len(suffix)
            best_key = key
    if best_key is not None:
        return best_key

    prefix_lower = prefix.lower()
    for key in keys:
        if not key.lower().startswith(prefix_lower):
            continue
        suffix = key[len(prefix) :]
        if suffix and suffix.lower() in objective_lower:
            return key
    return None


def sibling_for_detector(
    profiles: Optional[Mapping[str, Any]], key: Optional[str], detector: Optional[str]
) -> Optional[str]:
    """The profile for the same modality and objective as ``key`` but ``detector``'s camera.

    A profile picked by name (``lcpolscope_20x``) belongs to one camera's set; when the
    active camera is the other one, its own profile for the same modality and objective
    (``lcpolscope_20x_oryx``) is the one to use. Returns ``key`` itself when it already
    names ``detector`` or names no camera, or when the config has no such sibling.
    """
    if not profiles or key not in profiles or detector is None:
        return key
    profile = profiles[key]
    if not isinstance(profile, dict):
        return key
    own = profile.get("detector")
    if own is None or own == detector:
        return key
    modality = profile.get("modality")
    objective = profile.get("objective")
    if not isinstance(modality, str) or objective is None:
        return key
    for other_key, other in profiles.items():
        if (
            isinstance(other, dict)
            and other.get("detector") == detector
            and isinstance(other.get("modality"), str)
            and other["modality"].lower() == modality.lower()
            and other.get("objective") == objective
        ):
            return other_key
    return key
