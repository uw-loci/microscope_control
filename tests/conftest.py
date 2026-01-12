"""
Shared pytest fixtures for microscope_control tests.

Provides synthetic test data for image processing, autofocus, and coordinate testing.
"""

import numpy as np
import pytest
from microscope_control.hardware.base import Position


@pytest.fixture
def synthetic_focused_image():
    """
    Generate a synthetic focused image with high-frequency content.

    Returns:
        np.ndarray: 512x512 uint8 image with sharp edges and high contrast
    """
    size = 512
    img = np.zeros((size, size), dtype=np.uint8)

    # Add sharp edges (high frequency content indicates good focus)
    for i in range(0, size, 50):
        img[i:i+10, :] = 255
        img[:, i:i+10] = 255

    # Add some noise for texture
    noise = np.random.randint(0, 30, (size, size), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


@pytest.fixture
def synthetic_blurred_image():
    """
    Generate a synthetic blurred image with low-frequency content.

    Returns:
        np.ndarray: 512x512 uint8 image with blurred edges and low contrast
    """
    from scipy.ndimage import gaussian_filter

    size = 512
    img = np.zeros((size, size), dtype=np.uint8)

    # Add edges that will be blurred
    for i in range(0, size, 50):
        img[i:i+10, :] = 255
        img[:, i:i+10] = 255

    # Apply strong blur (simulates out-of-focus image)
    img = gaussian_filter(img.astype(float), sigma=15)
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img


@pytest.fixture
def synthetic_empty_image():
    """
    Generate a synthetic empty slide image (uniform white background).

    Returns:
        np.ndarray: 512x512x3 uint8 RGB image with near-white uniform color
    """
    size = 512
    # Slightly off-white to simulate real background (not pure 255)
    background_value = 245
    img = np.full((size, size, 3), background_value, dtype=np.uint8)

    # Add minimal noise to simulate camera noise
    noise = np.random.randint(-5, 5, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


@pytest.fixture
def synthetic_tissue_image():
    """
    Generate a synthetic tissue image with texture and variation.

    Returns:
        np.ndarray: 512x512x3 uint8 RGB image with tissue-like texture
    """
    size = 512
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Create colored regions simulating tissue
    img[:, :, 0] = 200 + np.random.randint(-50, 50, (size, size))  # Red channel
    img[:, :, 1] = 150 + np.random.randint(-50, 50, (size, size))  # Green channel
    img[:, :, 2] = 180 + np.random.randint(-50, 50, (size, size))  # Blue channel

    # Add some texture (spots, fibers)
    for _ in range(100):
        x, y = np.random.randint(0, size, 2)
        radius = np.random.randint(5, 20)
        y_grid, x_grid = np.ogrid[:size, :size]
        mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
        img[mask] = np.random.randint(50, 150, 3)

    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def sample_stage_limits():
    """
    Sample microscope stage limits configuration for coordinate validation testing.

    Returns:
        dict: Stage limit configuration dictionary
    """
    return {
        'stage': {
            'limits': {
                'x_um': {'low': 0.0, 'high': 100000.0},  # micrometers
                'y_um': {'low': 0.0, 'high': 75000.0},
                'z_um': {'low': 0.0, 'high': 10000.0}
            }
        }
    }


@pytest.fixture
def sample_position_valid():
    """
    Sample valid position within stage limits.

    Returns:
        Position: Valid position object
    """
    return Position(x=50000.0, y=37500.0, z=5000.0)


@pytest.fixture
def sample_position_out_of_range():
    """
    Sample position outside stage limits.

    Returns:
        Position: Out-of-range position object
    """
    return Position(x=150000.0, y=37500.0, z=5000.0)  # X exceeds max


@pytest.fixture
def sample_autofocus_config():
    """
    Sample autofocus configuration for testing.

    Returns:
        dict: Autofocus settings dictionary
    """
    return {
        'autofocus': {
            'n_steps': 21,
            'search_range': 200.0,  # micrometers
            'interp_strength': 10,
            'interp_kind': 'cubic',
            'score_metric': 'tenenbaum_gradient',
            'min_peak_prominence': 0.15,
            'max_peak_width': 0.7
        }
    }


@pytest.fixture
def temp_output_directory(tmp_path):
    """
    Create a temporary directory for test output files.

    Returns:
        Path: Path to temporary directory
    """
    return tmp_path


@pytest.fixture
def sample_microscope_config():
    """
    Sample microscope configuration for testing.

    Returns:
        dict: Complete microscope configuration dictionary
    """
    return {
        'microscope': {
            'name': 'Test Microscope',
            'type': 'PPM'
        },
        'modalities': {
            'ppm': {
                'rotation_angles': [0, 45, 90, 135]
            },
            'brightfield': {}
        },
        'acq_profiles': {
            'defaults': [
                {
                    'objective': '10x',
                    'settings': {
                        'pixel_size_xy_um': {'camera1': 0.645}
                    }
                }
            ],
            'profiles': [
                {
                    'modality': 'ppm',
                    'objective': '10x',
                    'detector': 'camera1'
                }
            ]
        },
        'stage': {
            'limits': {
                'x_um': {'low': 0.0, 'high': 100000.0},
                'y_um': {'low': 0.0, 'high': 75000.0},
                'z_um': {'low': 0.0, 'high': 10000.0}
            }
        },
        'objectives': {
            '10x': {
                'magnification': 10,
                'na': 0.3,
                'pixel_size_um': 0.645
            },
            '20x': {
                'magnification': 20,
                'na': 0.5,
                'pixel_size_um': 0.3225
            }
        },
        'autofocus': {
            'n_steps': 21,
            'search_range': 200.0,
            'interp_strength': 10,
            'interp_kind': 'cubic',
            'score_metric': 'tenenbaum_gradient',
            'min_peak_prominence': 0.15,
            'max_peak_width': 0.7
        }
    }
