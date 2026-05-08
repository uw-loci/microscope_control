"""Compatibility shim: autofocus strategies moved to
microscope_imageprocessing.focus on 2026-05-01.

This module re-exports the consolidated implementations so existing
imports continue to work during the migration window. Update your
imports to:

    from microscope_imageprocessing.focus import (
        AutofocusStrategy,
        DarkFieldStrategy,
        DenseTextureStrategy,
        ManualOnlyStrategy,
        SparseSignalStrategy,
        StrategyFailureMode,
        build_strategy,
    )

The shim will be removed in v2.0. See
``microscope_imageprocessing/microscope_imageprocessing/focus/`` for
the canonical implementations and the manifest-driven dispatcher.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "microscope_control.autofocus.strategies moved to "
    "microscope_imageprocessing.focus. Update your imports; this shim "
    "will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the consolidated public API. New code should import from
# microscope_imageprocessing.focus directly.
from microscope_imageprocessing.focus import (  # noqa: F401, E402
    AutofocusStrategy,
    DarkFieldStrategy,
    DenseTextureStrategy,
    ManualOnlyStrategy,
    SparseSignalStrategy,
    StrategyFailureMode,
    build_strategy,
)

__all__ = [
    "AutofocusStrategy",
    "DarkFieldStrategy",
    "DenseTextureStrategy",
    "ManualOnlyStrategy",
    "SparseSignalStrategy",
    "StrategyFailureMode",
    "build_strategy",
]
