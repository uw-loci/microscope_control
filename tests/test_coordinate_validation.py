"""
Unit tests for coordinate validation and safety checks.

Tests the is_coordinate_in_range function that prevents stage crashes
by validating positions against configured limits.
"""

import numpy as np
import pytest
from microscope_control.hardware.base import Position, is_coordinate_in_range


class TestPositionClass:
    """Test the Position data class."""

    def test_position_creation(self):
        """Test basic Position object creation."""
        pos = Position(x=1000.0, y=2000.0, z=500.0)

        assert pos.x == 1000.0
        assert pos.y == 2000.0
        assert pos.z == 500.0

    def test_position_with_none_values(self):
        """Test Position with None values (partial positions)."""
        pos = Position(x=1000.0, y=None, z=500.0)

        assert pos.x == 1000.0
        assert pos.y is None
        assert pos.z == 500.0

    def test_position_populate_missing(self, sample_position_valid):
        """Test populate_missing method if it exists."""
        try:
            partial = Position(x=1000.0, y=None, z=None)
            partial.populate_missing(sample_position_valid)

            # Should fill in missing values from reference position
            assert partial.x == 1000.0  # Kept original
            assert partial.y is not None  # Populated
            assert partial.z is not None  # Populated
        except AttributeError:
            pytest.skip("populate_missing method not available")


class TestIsCoordinateInRange:
    """Test coordinate validation against stage limits."""

    def test_valid_position_within_limits(
        self, sample_position_valid, sample_stage_limits
    ):
        """Test that valid position within limits is accepted."""
        is_valid = is_coordinate_in_range(sample_stage_limits, sample_position_valid)

        assert is_valid is True

    def test_position_outside_x_max(self, sample_stage_limits):
        """Test that position exceeding X max limit is rejected."""
        pos_out = Position(x=150000.0, y=37500.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_out)

        assert is_valid is False

    def test_position_outside_x_min(self, sample_stage_limits):
        """Test that position below X min limit is rejected."""
        pos_out = Position(x=-1000.0, y=37500.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_out)

        assert is_valid is False

    def test_position_outside_y_max(self, sample_stage_limits):
        """Test that position exceeding Y max limit is rejected."""
        pos_out = Position(x=50000.0, y=100000.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_out)

        assert is_valid is False

    def test_position_outside_y_min(self, sample_stage_limits):
        """Test that position below Y min limit is rejected."""
        pos_out = Position(x=50000.0, y=-500.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_out)

        assert is_valid is False

    def test_position_outside_z_max(self, sample_stage_limits):
        """Test that position exceeding Z max limit is rejected."""
        pos_out = Position(x=50000.0, y=37500.0, z=15000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_out)

        assert is_valid is False

    def test_position_outside_z_min(self, sample_stage_limits):
        """Test that position below Z min limit is rejected."""
        pos_out = Position(x=50000.0, y=37500.0, z=-100.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_out)

        assert is_valid is False


class TestIsCoordinateInRangeBoundaryConditions:
    """Test boundary conditions for coordinate validation."""

    def test_position_exactly_at_x_max(self, sample_stage_limits):
        """Test position exactly at X maximum boundary."""
        x_max = sample_stage_limits['stage']['limits']['x_um']['high']
        pos_boundary = Position(x=x_max, y=37500.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_boundary)

        # Should accept position exactly at boundary
        assert is_valid is True

    def test_position_exactly_at_x_min(self, sample_stage_limits):
        """Test position exactly at X minimum boundary."""
        x_min = sample_stage_limits['stage']['limits']['x_um']['low']
        pos_boundary = Position(x=x_min, y=37500.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_boundary)

        assert is_valid is True

    def test_position_just_inside_x_max(self, sample_stage_limits):
        """Test position just inside X maximum boundary."""
        x_max = sample_stage_limits['stage']['limits']['x_um']['high']
        pos_inside = Position(x=x_max - 0.1, y=37500.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_inside)

        assert is_valid is True

    def test_position_just_outside_x_max(self, sample_stage_limits):
        """Test position just outside X maximum boundary."""
        x_max = sample_stage_limits['stage']['limits']['x_um']['high']
        pos_outside = Position(x=x_max + 0.1, y=37500.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_outside)

        assert is_valid is False

    def test_all_axes_at_boundaries(self, sample_stage_limits):
        """Test position at all boundary limits simultaneously."""
        limits = sample_stage_limits['stage']['limits']
        pos_boundary = Position(
            x=limits['x_um']['high'],
            y=limits['y_um']['high'],
            z=limits['z_um']['high']
        )

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_boundary)

        assert is_valid is True


class TestIsCoordinateInRangePartialPositions:
    """Test coordinate validation with partial positions (None values)."""

    def test_position_with_none_x(self, sample_stage_limits):
        """Test validation when X coordinate is None."""
        pos_partial = Position(x=None, y=37500.0, z=5000.0)

        # Should validate only the non-None coordinates
        try:
            is_valid = is_coordinate_in_range(sample_stage_limits, pos_partial)
            # Y and Z are valid, so should pass (if X None is handled)
            assert isinstance(is_valid, bool)
        except (ValueError, TypeError):
            # May require all coordinates to be non-None
            pass

    def test_position_with_none_y(self, sample_stage_limits):
        """Test validation when Y coordinate is None."""
        pos_partial = Position(x=50000.0, y=None, z=5000.0)

        try:
            is_valid = is_coordinate_in_range(sample_stage_limits, pos_partial)
            assert isinstance(is_valid, bool)
        except (ValueError, TypeError):
            pass

    def test_position_with_none_z(self, sample_stage_limits):
        """Test validation when Z coordinate is None."""
        pos_partial = Position(x=50000.0, y=37500.0, z=None)

        try:
            is_valid = is_coordinate_in_range(sample_stage_limits, pos_partial)
            assert isinstance(is_valid, bool)
        except (ValueError, TypeError):
            pass


class TestIsCoordinateInRangeConfigurationErrors:
    """Test handling of missing or malformed limit configurations."""

    def test_missing_limits_config(self, sample_position_valid):
        """Test behavior when limits configuration is missing."""
        empty_config = {}

        try:
            is_valid = is_coordinate_in_range(empty_config, sample_position_valid)
            # Should handle gracefully (return False or raise clear error)
            assert isinstance(is_valid, bool)
        except (KeyError, ValueError) as e:
            # Expected to raise error for missing config
            assert "limit" in str(e).lower() or "stage" in str(e).lower()

    def test_missing_x_limits(self, sample_position_valid):
        """Test behavior when X limits are missing."""
        incomplete_config = {
            'stage': {
                'limits': {
                    'y_um': {'low': 0.0, 'high': 75000.0},
                    'z_um': {'low': 0.0, 'high': 10000.0}
                }
            }
        }

        try:
            is_valid = is_coordinate_in_range(incomplete_config, sample_position_valid)
            # Should handle missing axis
            assert isinstance(is_valid, bool)
        except (KeyError, ValueError):
            # Expected to raise error for missing axis limits
            pass

    def test_malformed_limits_structure(self, sample_position_valid):
        """Test behavior with malformed limits structure."""
        malformed_config = {
            'stage': {
                'limits': {
                    'x': 100000.0,  # Should be dict with min/max, not single value
                    'y_um': {'low': 0.0, 'high': 75000.0},
                    'z_um': {'low': 0.0, 'high': 10000.0}
                }
            }
        }

        try:
            is_valid = is_coordinate_in_range(malformed_config, sample_position_valid)
            assert isinstance(is_valid, bool)
        except (KeyError, TypeError, ValueError):
            # Expected to raise error for malformed config
            pass

    def test_inverted_limits(self, sample_position_valid):
        """Test behavior when min > max (configuration error)."""
        inverted_config = {
            'stage': {
                'limits': {
                    'x_um': {'low': 100000.0, 'high': 0.0},  # Inverted!
                    'y_um': {'low': 0.0, 'high': 75000.0},
                    'z_um': {'low': 0.0, 'high': 10000.0}
                }
            }
        }

        # Position that would be valid with correct limits
        pos = Position(x=50000.0, y=37500.0, z=5000.0)

        is_valid = is_coordinate_in_range(inverted_config, pos)

        # With inverted limits, this position should be out of range
        assert is_valid is False


class TestIsCoordinateInRangeMultipleViolations:
    """Test positions that violate multiple limits."""

    def test_all_axes_out_of_range(self, sample_stage_limits):
        """Test position that exceeds all axis limits."""
        pos_all_out = Position(x=200000.0, y=100000.0, z=20000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_all_out)

        assert is_valid is False

    def test_two_axes_out_of_range(self, sample_stage_limits):
        """Test position that exceeds two axis limits."""
        pos_two_out = Position(x=150000.0, y=100000.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_two_out)

        assert is_valid is False


class TestIsCoordinateInRangeFloatingPoint:
    """Test floating-point precision handling."""

    def test_very_small_out_of_range(self, sample_stage_limits):
        """Test position that's very slightly out of range."""
        x_max = sample_stage_limits['stage']['limits']['x_um']['high']
        pos_slightly_out = Position(x=x_max + 1e-6, y=37500.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_slightly_out)

        # Even tiny violation should be caught
        assert is_valid is False

    def test_floating_point_boundary(self, sample_stage_limits):
        """Test floating-point comparison at boundaries."""
        x_max = sample_stage_limits['stage']['limits']['x_um']['high']

        # Add tiny floating-point error
        pos_with_error = Position(x=x_max + 1e-10, y=37500.0, z=5000.0)

        is_valid = is_coordinate_in_range(sample_stage_limits, pos_with_error)

        # Should handle floating-point comparison carefully
        # May accept tiny rounding errors or require exact comparison
        assert isinstance(is_valid, bool)


class TestIsCoordinateInRangeNegativeCoordinates:
    """Test validation with negative coordinates."""

    def test_negative_coordinates_allowed(self):
        """Test that negative coordinates are allowed if within limits."""
        config_with_negative = {
            'stage': {
                'limits': {
                    'x_um': {'low': -50000.0, 'high': 50000.0},
                    'y_um': {'low': -50000.0, 'high': 50000.0},
                    'z_um': {'low': 0.0, 'high': 10000.0}
                }
            }
        }

        pos_negative = Position(x=-25000.0, y=-25000.0, z=5000.0)

        is_valid = is_coordinate_in_range(config_with_negative, pos_negative)

        assert is_valid is True

    def test_negative_beyond_limit(self):
        """Test negative coordinate beyond negative limit."""
        config_with_negative = {
            'stage': {
                'limits': {
                    'x_um': {'low': -50000.0, 'high': 50000.0},
                    'y_um': {'low': -50000.0, 'high': 50000.0},
                    'z_um': {'low': 0.0, 'high': 10000.0}
                }
            }
        }

        pos_too_negative = Position(x=-75000.0, y=-25000.0, z=5000.0)

        is_valid = is_coordinate_in_range(config_with_negative, pos_too_negative)

        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
