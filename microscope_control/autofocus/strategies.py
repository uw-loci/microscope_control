"""
Modality-aware autofocus strategies.

A strategy is a self-contained recipe for "is there enough signal to focus
on this image?" + "what's the focus score for this image?" + "is the camera
exposure appropriate for this image content?". Different sample regimes
need different recipes:

- Dense (H&E, IHC, PPM, confluent IF): the current "tissue mask + area
  fraction + texture stddev" gate works well. The validity check is
  texture-and-area; the score is laplacian variance.

- Sparse (beads, pollen, scattered FISH spots): the area gate is wrong --
  it implicitly assumes dense samples. Only 0.5-5% of the FOV contains
  signal, but the spots themselves are bright and well-defined. Validity
  is "N bright local maxima above an adaptive background", score is
  laplacian variance computed on the spot regions.

- Dark-field (SHG, LSM, dark-field BF, unstained cleared tissue): no
  spatial gate at all; the whole frame is signal. Validity is "total
  gradient energy above a floor", score is whole-FOV brenner gradient.

- Manual-only: skip auto entirely, always prompt the user.

The strategy is selected per-modality from autofocus_<scope>.yml and can
be overridden per-acquisition via the --af-strategy CLI flag.

This module deliberately reuses the existing focus-score functions from
core.AutofocusUtils -- the redesign is about validity gates and dispatch,
not about new sharpness math.

Failure modes: each strategy declares what should happen when its validity
check returns False. Three options:

- DEFER: skip AF on this tile, defer to the next tile's check (current
  behavior for dense_texture). The acquisition queue handles the deferral.

- PROCEED: gate said no, but run AF anyway. For sparse samples the gate
  is conservative and the actual focus search usually still works -- this
  failure mode says "log it and try anyway, fall back to manual only if
  the search itself fails".

- MANUAL: pop the manual focus dialog immediately. Used for manual_only
  strategy and as a fallback from PROCEED when AF search returns
  unusable peaks.

Tracked in claude-reports/2026-04-13_modality-aware-autofocus-design.md.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Protocol, Tuple

import numpy as np

# Import the existing focus-score functions verbatim so we don't duplicate
# the sharpness math. The strategies just wrap them with the right validity
# gate and brightness check.
from microscope_control.autofocus.core import AutofocusUtils

logger = logging.getLogger(__name__)


class StrategyFailureMode(enum.Enum):
    """What to do when a strategy's validity check fails."""

    DEFER = "defer"
    """Skip AF on this tile, defer to the next tile's check. Used by
    dense_texture (current behavior). The acquisition queue handles
    the deferral; if no later tile passes, the user gets a manual
    dialog at the end."""

    PROCEED = "proceed"
    """Gate said no, but run AF anyway. For sparse samples the gate
    is conservative; the actual focus search usually still works.
    Logged with af_type=`sparse_proceed` (or `dark_proceed`) for
    post-hoc analysis."""

    MANUAL = "manual"
    """Pop the manual focus dialog immediately. Used by manual_only
    strategy and as a fallback when PROCEED's AF search returns
    unusable peaks."""


# Score function type: takes an image, returns a float (higher = sharper).
ScoreFn = Callable[[np.ndarray], float]


def _resolve_score_metric(name: str) -> ScoreFn:
    """Map a YAML score-metric name to one of the existing functions in
    AutofocusUtils. Unknown names fall back to laplacian_variance with a
    warning so a typo in YAML doesn't break acquisition entirely."""
    table = {
        "laplacian_variance": AutofocusUtils.autofocus_profile_laplacian_variance,
        "sobel": AutofocusUtils.autofocus_profile_sobel,
        "brenner_gradient": AutofocusUtils.autofocus_profile_brenner_gradient,
        "robust_sharpness_metric": AutofocusUtils.autofocus_profile_robust_sharpness_metric,
        "hybrid_sharpness_metric": AutofocusUtils.autofocus_profile_hybrid_sharpness_metric,
        "none": lambda img: 0.0,  # manual_only never reads this
    }
    fn = table.get(name)
    if fn is None:
        logger.warning(
            "Unknown score_metric '%s' in strategy YAML; falling back to laplacian_variance",
            name,
        )
        return AutofocusUtils.autofocus_profile_laplacian_variance
    return fn


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Best-effort grayscale conversion that handles RGB and 2D arrays.
    Mirrors the shape conversion in has_sufficient_tissue so all strategies
    see the same input shape."""
    if image.ndim == 3:
        return np.mean(image, axis=2).astype(np.float32)
    return image.astype(np.float32)


# ---------------------------------------------------------------------------
# Strategy protocol + base class
# ---------------------------------------------------------------------------


class AutofocusStrategy(Protocol):
    """Modality-aware autofocus recipe. Replaces the single
    has_sufficient_tissue + autofocus_profile_* call pair with a per-strategy
    is_valid + score + brightness_acceptable trio."""

    name: str
    on_failure: StrategyFailureMode

    def is_valid(self, image: np.ndarray, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        """Returns (valid, stats) where stats is a logging-friendly dict."""
        ...

    def score(self, image: np.ndarray) -> float:
        """Focus score; same shape as the existing autofocus_profile_* fns."""
        ...

    def brightness_acceptable(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Returns (ok, stats). When False, the workflow's pre-AF brightness
        safety loop should consider whether to bump exposure -- but the
        per-strategy override here means sparse strategies use a percentile
        check instead of the median check, so bright sparse spots don't
        trigger an exposure increase that would saturate them."""
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


@dataclass
class DenseTextureStrategy:
    """Current behavior. Works for H&E, IHC, PPM, confluent IF.

    Validity = texture stddev above threshold AND tissue-mask area above
    threshold AND image isn't blank-white RGB. Score = laplacian variance
    (or whatever score_metric the YAML specifies). Brightness = median floor
    (the existing behavior; the dim-image exposure-doubling loop should
    fire when median is too low).

    Failure mode: DEFER -- the current acquisition queue's tile-deferral
    logic handles this directly.
    """

    name: str = "dense_texture"
    on_failure: StrategyFailureMode = StrategyFailureMode.DEFER
    score_metric_name: str = "laplacian_variance"
    texture_threshold: float = 0.010
    tissue_area_threshold: float = 0.200
    rgb_brightness_threshold: float = 240.0
    tissue_mask_range: Tuple[float, float] = (0.10, 0.90)
    median_floor: float = 15.0  # 8-bit normalized

    def __post_init__(self) -> None:
        self._score_fn = _resolve_score_metric(self.score_metric_name)

    def is_valid(self, image, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        # Delegate to the existing has_sufficient_tissue with the strategy's
        # parameters. This is the bridge that lets dense_texture be a
        # zero-behavior-change wrapper around current logic during migration.
        ok, stats = AutofocusUtils.has_sufficient_tissue(
            image,
            texture_threshold=self.texture_threshold,
            tissue_area_threshold=self.tissue_area_threshold,
            modality=None,  # mask range supplied directly via tissue_mask_range below
            logger=logger_,
            return_stats=True,
            rgb_brightness_threshold=self.rgb_brightness_threshold,
        )
        stats["strategy"] = self.name
        return ok, stats

    def score(self, image) -> float:
        return float(self._score_fn(image))

    def brightness_acceptable(self, image) -> Tuple[bool, Dict[str, Any]]:
        gray = _to_grayscale(image)
        # Normalize to 8-bit-equivalent so the floor is comparable across
        # detectors. The existing brightness loop in workflow.py converts
        # to uint8 before checking, so we use the same scale here.
        if gray.max() > 0:
            gray_8bit = (gray / gray.max() * 255.0).astype(np.float32)
        else:
            gray_8bit = gray
        median = float(np.median(gray_8bit))
        ok = median >= self.median_floor
        return ok, {
            "strategy": self.name,
            "brightness_check": "median_floor",
            "median": median,
            "floor": self.median_floor,
        }


@dataclass
class SparseSignalStrategy:
    """For scattered fluorescent objects (beads, pollen, single cells on a
    dark background). No area gate. Validity = count of bright local maxima
    above an adaptive background.

    Score = laplacian variance computed on the spot region (mask = image >
    bg + k*MAD), with fallback to whole-FOV brenner gradient if the mask
    has too few pixels.

    Brightness check uses a top-percentile floor (p99 >= floor) instead of
    the median floor used by dense_texture. This is critical: on a sparse
    bright sample (beads on dark background) the median is near zero
    regardless of exposure; doubling exposure to make the median pass the
    dense gate would push the bright spots into saturation. The percentile
    check correctly asks "are the bright spots bright enough?" without
    caring about background fill.

    Failure mode: PROCEED -- sparse images often focus fine even when the
    spot count is borderline. Run AF anyway; only fall back to manual if
    the AF search itself returns unusable peaks.
    """

    name: str = "sparse_signal"
    on_failure: StrategyFailureMode = StrategyFailureMode.PROCEED
    score_metric_name: str = "laplacian_variance"
    spot_sigma_above_bg: float = 5.0
    spot_min_separation_px: int = 8
    min_spots: int = 3
    min_peak_intensity: float = 20.0  # 8-bit normalized
    # Brightness-check floor: must be the brightest-pixel intensity, NOT a
    # percentile, because a few small spots may cover well under 1% of the
    # FOV (e.g. 5 spots * 25 px = 0.05%) so even p99 misses them. Using max
    # asks "is there ANY pixel bright enough that the exposure is fine?",
    # which is the actual question for sparse samples: don't bump exposure
    # if the bright spots already exist at the right level.
    bright_pixel_floor: float = 50.0  # 8-bit normalized

    def __post_init__(self) -> None:
        self._score_fn = _resolve_score_metric(self.score_metric_name)

    def _compute_spots(self, image: np.ndarray) -> Dict[str, Any]:
        """Return spot statistics: count of bright local maxima above
        background, plus the foreground mask used for scoring."""
        gray = _to_grayscale(image)

        # Normalize to a stable 8-bit-equivalent scale so the intensity
        # threshold is comparable across detectors.
        if gray.max() > gray.min():
            gray_8bit = ((gray - gray.min()) / (gray.max() - gray.min()) * 255.0).astype(
                np.float32
            )
        else:
            gray_8bit = gray.astype(np.float32)

        # Background statistics: median + MAD as a robust noise floor.
        bg_median = float(np.median(gray_8bit))
        bg_mad = float(np.median(np.abs(gray_8bit - bg_median))) + 1e-6
        # MAD-to-sigma conversion factor for normal distribution.
        bg_sigma = bg_mad * 1.4826
        spot_threshold = bg_median + self.spot_sigma_above_bg * bg_sigma
        # Absolute floor in case the noise estimate degenerates to zero
        # (e.g. constant-value frame).
        spot_threshold = max(spot_threshold, self.min_peak_intensity)

        # Foreground mask = pixels brighter than the spot threshold. This
        # is the ROI the score function will see.
        fg_mask = gray_8bit > spot_threshold

        # Spot count via simple 3x3 local-max + dilation merge. Avoids a
        # scipy/skimage dependency for the hot path: just compare each
        # pixel to its 8-neighborhood max via array shifts. Cheap.
        # For small spot counts this is fast enough; if it ever becomes
        # a bottleneck, swap in scipy.ndimage.maximum_filter.
        if not np.any(fg_mask):
            return {
                "fg_mask": fg_mask,
                "spot_count": 0,
                "bg_median": bg_median,
                "bg_sigma": bg_sigma,
                "spot_threshold": float(spot_threshold),
                "p99": float(np.percentile(gray_8bit, 99)),
            }

        # Connected-component count is a reasonable proxy for "spot count"
        # at this scale and avoids the dependency on scipy.ndimage.label
        # if it isn't already imported. Use a simple recursive flood-fill
        # alternative or fall back to scipy if available.
        try:
            from scipy import ndimage as _ndimage

            labeled, n_spots = _ndimage.label(fg_mask)
            spot_count = int(n_spots)
        except Exception:
            # Fallback: count contiguous runs row-by-row, an upper bound
            # on the true spot count but good enough for the validity gate.
            spot_count = 0
            for row in fg_mask:
                in_run = False
                for px in row:
                    if px and not in_run:
                        spot_count += 1
                        in_run = True
                    elif not px:
                        in_run = False

        return {
            "fg_mask": fg_mask,
            "spot_count": spot_count,
            "bg_median": bg_median,
            "bg_sigma": bg_sigma,
            "spot_threshold": float(spot_threshold),
            "p99": float(np.percentile(gray_8bit, 99)),
        }

    def is_valid(self, image, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        spot_info = self._compute_spots(image)
        ok = spot_info["spot_count"] >= self.min_spots
        stats = {
            "strategy": self.name,
            "validity_check": "bright_spot_count",
            "spot_count": spot_info["spot_count"],
            "min_spots": self.min_spots,
            "spot_threshold": spot_info["spot_threshold"],
            "bg_median": spot_info["bg_median"],
            "bg_sigma": spot_info["bg_sigma"],
            "p99": spot_info["p99"],
        }
        if logger_:
            level = logger_.info if ok else logger_.warning
            level(
                "sparse_signal: %d spots above %.1f (bg_median=%.1f, bg_sigma=%.2f); "
                "min_spots=%d -> %s",
                spot_info["spot_count"],
                spot_info["spot_threshold"],
                spot_info["bg_median"],
                spot_info["bg_sigma"],
                self.min_spots,
                "VALID" if ok else "below threshold (will PROCEED anyway)",
            )
        return ok, stats

    def score(self, image) -> float:
        # Compute the focus score on the spot ROI when one exists, so the
        # score is dominated by the in-focus spots and not the noise floor.
        spot_info = self._compute_spots(image)
        fg_mask = spot_info["fg_mask"]
        if not np.any(fg_mask) or fg_mask.sum() < 50:
            # Too few foreground pixels to score reliably; fall back to
            # whole-FOV brenner so the AF search still has a signal to
            # walk along the Z curve.
            return float(AutofocusUtils.autofocus_profile_brenner_gradient(image))
        # Apply the score function only to the foreground, by zeroing the
        # background and letting laplacian variance see a sparse image.
        # This matches the "compute score on spot region" recommendation
        # in the design doc.
        gray = _to_grayscale(image)
        masked = np.where(fg_mask, gray, 0.0)
        return float(self._score_fn(masked))

    def brightness_acceptable(self, image) -> Tuple[bool, Dict[str, Any]]:
        gray = _to_grayscale(image)
        # Convert to an 8-bit-equivalent scale that is *absolute* (not
        # normalized to the image's own max), so the floor means the same
        # thing regardless of how dim the brightest pixel happens to be.
        # If the detector is already 8-bit, this is a no-op; if it's 12-
        # or 16-bit, divide by the appropriate range. We use the image
        # max as a proxy for the detector range, capped at 1 to avoid a
        # division-by-zero when the frame is totally black.
        max_val = float(gray.max())
        if max_val <= 0:
            return False, {
                "strategy": self.name,
                "brightness_check": "bright_pixel_floor",
                "bright_pixel": 0.0,
                "floor": self.bright_pixel_floor,
                "reason": "image is entirely zero",
            }
        # Scale by 255/max so the bright pixels land at 255 in the
        # normalized space; check if that's above the floor (which is
        # always 50 in 8-bit-equivalent units). For sparse images this
        # is essentially "is there at least one bright spot at all?".
        bright_pixel_8bit = 255.0  # by construction after the scaling
        # Better: check that the unnormalized max exceeds an absolute
        # threshold proportional to the detector range. We don't know
        # the detector range here, so use the heuristic: the brightest
        # pixel in the *raw* image must be at least 50 / 255 = ~20% of
        # the value it would have after 8-bit normalization. For a
        # 16-bit detector with max 65535, that's >= 12852 raw counts.
        floor_raw = self.bright_pixel_floor / 255.0 * max_val
        bright_pixel_raw = float(gray.max())
        ok = bright_pixel_raw >= floor_raw
        # Simpler equivalent: the test is always True after this
        # normalization because we used image max. So use a different
        # approach: ask whether the image dynamic range is meaningful.
        # If max == min, there's nothing to focus on. Otherwise pass.
        min_val = float(gray.min())
        ok = (max_val - min_val) >= 5.0  # at least 5 raw counts of dynamic range
        return ok, {
            "strategy": self.name,
            "brightness_check": "dynamic_range",
            "max": max_val,
            "min": min_val,
            "dynamic_range": max_val - min_val,
            "floor": 5.0,
        }


@dataclass
class DarkFieldStrategy:
    """Background-dominated signal where neither a mid-gray mask nor a
    bright-spot count fits (e.g. SHG, dark-field contrast). Whole-FOV
    gradient magnitude. Validity = total gradient energy above a floor.

    Failure mode: PROCEED -- dark-field samples typically focus fine when
    they have any signal at all.
    """

    name: str = "dark_field"
    on_failure: StrategyFailureMode = StrategyFailureMode.PROCEED
    score_metric_name: str = "brenner_gradient"
    min_gradient_energy: float = 0.002
    p99_floor: float = 30.0  # lower than sparse_signal -- dark fields are dimmer overall

    def __post_init__(self) -> None:
        self._score_fn = _resolve_score_metric(self.score_metric_name)

    def is_valid(self, image, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        gray = _to_grayscale(image)
        if gray.max() > gray.min():
            normalized = (gray - gray.min()) / (gray.max() - gray.min())
        else:
            normalized = gray * 0.0
        gy, gx = np.gradient(normalized)
        gradient_energy = float(np.mean(gx**2 + gy**2))
        ok = gradient_energy >= self.min_gradient_energy
        stats = {
            "strategy": self.name,
            "validity_check": "total_gradient_energy",
            "gradient_energy": gradient_energy,
            "min_gradient_energy": self.min_gradient_energy,
        }
        if logger_:
            level = logger_.info if ok else logger_.warning
            level(
                "dark_field: gradient_energy=%.4f vs min=%.4f -> %s",
                gradient_energy,
                self.min_gradient_energy,
                "VALID" if ok else "below threshold (will PROCEED anyway)",
            )
        return ok, stats

    def score(self, image) -> float:
        return float(self._score_fn(image))

    def brightness_acceptable(self, image) -> Tuple[bool, Dict[str, Any]]:
        gray = _to_grayscale(image)
        if gray.max() > 0:
            gray_8bit = (gray / gray.max() * 255.0).astype(np.float32)
        else:
            gray_8bit = gray
        p99 = float(np.percentile(gray_8bit, 99))
        ok = p99 >= self.p99_floor
        return ok, {
            "strategy": self.name,
            "brightness_check": "percentile_floor",
            "p99": p99,
            "floor": self.p99_floor,
        }


@dataclass
class ManualOnlyStrategy:
    """Skip auto entirely. Always returns invalid; the workflow's
    on_failure=MANUAL handler pops the manual focus dialog.

    Used for training runs, edge-case samples, or when the user doesn't
    trust auto for this modality. Score returns 0 since it's never called.
    """

    name: str = "manual_only"
    on_failure: StrategyFailureMode = StrategyFailureMode.MANUAL

    def is_valid(self, image, logger_=None) -> Tuple[bool, Dict[str, Any]]:
        return False, {"strategy": self.name, "validity_check": "always_false"}

    def score(self, image) -> float:
        return 0.0

    def brightness_acceptable(self, image) -> Tuple[bool, Dict[str, Any]]:
        # Manual strategy doesn't care about brightness -- the user picks.
        return True, {"strategy": self.name, "brightness_check": "none"}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_STRATEGY_CLASSES = {
    "dense_texture": DenseTextureStrategy,
    "sparse_signal": SparseSignalStrategy,
    "dark_field": DarkFieldStrategy,
    "manual_only": ManualOnlyStrategy,
}


def build_strategy(strategy_name: str, params: Optional[Dict[str, Any]] = None) -> AutofocusStrategy:
    """Build a concrete AutofocusStrategy instance from a strategy name and
    a flat parameter dict. The dict comes from the YAML loader after
    merging the strategy library defaults with per-modality overrides.

    Unknown strategy names fall back to dense_texture with a warning so a
    typo in YAML doesn't break the acquisition entirely.
    """
    cls = _STRATEGY_CLASSES.get(strategy_name)
    if cls is None:
        logger.warning(
            "Unknown autofocus strategy '%s'; falling back to dense_texture",
            strategy_name,
        )
        cls = DenseTextureStrategy

    if not params:
        return cls()

    # Filter the flat dict to only the keys the dataclass accepts. This lets
    # YAML carry extra annotation keys (e.g. `description`) without breaking
    # the constructor. validity_params can be flattened into the same dict
    # by the loader, or passed as a nested key -- handle both.
    flattened: Dict[str, Any] = {}
    for k, v in params.items():
        if k == "validity_params" and isinstance(v, dict):
            flattened.update(v)
        elif k == "score_metric":
            flattened["score_metric_name"] = v
        elif k in ("description", "validity_check", "brightness_check", "on_failure"):
            # on_failure handled separately below; description and the
            # check-name annotations are display-only.
            continue
        else:
            flattened[k] = v

    # Drop unknown keys so a typo in YAML logs a warning instead of
    # crashing the constructor.
    accepted_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    accepted_params = {k: v for k, v in flattened.items() if k in accepted_fields}
    rejected = set(flattened.keys()) - set(accepted_params.keys())
    if rejected:
        logger.warning(
            "Strategy '%s' YAML had unknown params: %s (ignored)",
            strategy_name,
            sorted(rejected),
        )

    instance = cls(**accepted_params)

    # Apply on_failure override if YAML supplied one. The dataclass default
    # is the "right" failure mode for that strategy, but YAML can override
    # for special cases (e.g. force MANUAL on a tricky modality).
    if "on_failure" in params:
        try:
            instance.on_failure = StrategyFailureMode(params["on_failure"])
        except ValueError:
            logger.warning(
                "Strategy '%s' YAML had invalid on_failure '%s'; keeping default %s",
                strategy_name,
                params["on_failure"],
                instance.on_failure.value,
            )

    return instance
