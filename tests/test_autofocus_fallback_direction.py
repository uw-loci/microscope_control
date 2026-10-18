"""Which metric steers a widened autofocus search, and how a refused fallback is reported.

Slide 4 of 2026-10-04 (claude-reports/2026-10-06_slide4-autofocus-stale-seed-z.md): the
primary metric read flat 157 times, 60 um from focus, while p98_p2 from the same frames
pointed toward low Z every time. Nothing followed it, and the refusal was logged as
"p98_p2 fallback also invalid: Valid focus peak detected", which read as a bug.
"""

from microscope_control.hardware.pycromanager import (
    describe_fallback_refusal,
    edge_retry_guide,
)

FLAT = {"is_valid": False, "should_extend_direction": None, "message": "score range too small"}
LOW_TREND = {
    "is_valid": True,
    "should_extend_direction": "low",
    "message": "Valid focus peak detected",
}
BRACKETED = {
    "is_valid": True,
    "should_extend_direction": None,
    "message": "Valid focus peak detected",
}


def test_the_primary_metric_steers_when_it_has_a_direction():
    primary = {"is_valid": True, "should_extend_direction": "high"}
    assert edge_retry_guide(primary, LOW_TREND) == (primary, "primary")


def test_a_flat_primary_follows_the_fallback_trend():
    assert edge_retry_guide(FLAT, LOW_TREND) == (LOW_TREND, "p98_p2")


def test_no_direction_anywhere_means_no_retry():
    assert edge_retry_guide(FLAT, BRACKETED) == (None, None)
    assert edge_retry_guide(FLAT, None) == (None, None)


def test_a_valid_primary_without_direction_is_not_overridden():
    valid = {"is_valid": True, "should_extend_direction": None}
    assert edge_retry_guide(valid, LOW_TREND) == (None, None)


def test_a_refused_edge_peak_says_why():
    text = describe_fallback_refusal(LOW_TREND)
    assert "refused" in text and "low-Z edge" in text and "never bracketed" in text
    assert "Valid focus peak detected" not in text


def test_a_fallback_with_no_peak_is_still_called_invalid():
    assert describe_fallback_refusal(FLAT) == "p98_p2 fallback also invalid: score range too small"
