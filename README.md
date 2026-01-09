# Microscope Control

Hardware control library for microscopes via Pycromanager (Micro-Manager).

> **Part of the QPSC (QuPath Scope Control) system**
> For complete installation instructions, see: https://github.com/uw-loci/QPSC

## Features

- **Hardware Abstraction**: Platform-independent microscope control interface
- **Pycromanager Integration**: Full Micro-Manager hardware support
- **Autofocus**: Multiple autofocus algorithms and metrics
- **Stage Control**: XYZ stage positioning and movement
- **Configuration Management**: YAML-based microscope configuration

## Installation

**Part of [QPSC (QuPath Scope Control)](https://github.com/uw-loci/QPSC)**

**Requirements:**
- Python 3.9 or later
- pip (Python package installer)
- Git (for `pip install git+https://...` commands)
- Micro-Manager 2.0+

⚠️ **Important**: This package depends on `ppm-library` and requires Micro-Manager 2.0+.
See the [QPSC Installation Guide](https://github.com/uw-loci/QPSC#automated-installation-windows) for complete setup instructions.

### Quick Install (from GitHub)

**Install dependencies first:**
```bash
# 1. Install ppm-library
pip install git+https://github.com/uw-loci/ppm_library.git

# 2. Then install microscope-control
pip install git+https://github.com/uw-loci/microscope_control.git
```

### Development Install (editable mode)

```bash
git clone https://github.com/uw-loci/microscope_control.git
cd microscope_control
pip install -e .
```

**For automated setup**, use the [QPSC setup script](https://github.com/uw-loci/QPSC/blob/main/PPM-QuPath.ps1).

### Troubleshooting Installation

#### Problem: `ModuleNotFoundError: No module named 'microscope_control'`

**Cause:** Editable install issue with package structure.

**Solution:**

The repository has been updated to fix this issue. If you encounter this error:

1. **Update to latest version:**
   ```bash
   cd microscope_control
   git pull
   ```

2. **Verify `pyproject.toml` has the correct configuration:**
   ```toml
   [tool.hatch.build.targets.wheel]
   packages = ["."]
   ```

3. **Reinstall in editable mode:**
   ```bash
   pip install -e . --force-reinstall --no-deps
   ```

4. **Verify installation:**
   ```bash
   pip show microscope-control
   ```

#### Problem: `ModuleNotFoundError: No module named 'cv2'`

**Cause:** Missing OpenCV dependency (required by autofocus metrics).

**Solution:**

Install OpenCV:
```bash
pip install opencv-python
```

This dependency will be added to `pyproject.toml` requirements in a future update.

For more troubleshooting, see the [QPSC Installation Guide](https://github.com/uw-loci/QPSC#troubleshooting-python-package-installation).

## Quick Start

```python
from microscope_control import init_pycromanager, PycromanagerHardware, ConfigManager

# Initialize hardware
core, studio = init_pycromanager()
config_mgr = ConfigManager()
settings = config_mgr.get_config('config_PPM')

hardware = PycromanagerHardware(core, studio, settings)

# Move stage
from microscope_control import Position
hardware.move_to_position(Position(x=1000, y=2000, z=-3000))

# Run autofocus
from microscope_control import AutofocusUtils
af = AutofocusUtils(hardware, settings)
best_z = af.run_autofocus(current_position)
```

## Testing

This package includes two types of testing tools:

### Hardware Diagnostic Tools

Hardware characterization and calibration tools are located in the source modules:
- **`microscope_control/autofocus/test.py`** - Autofocus diagnostic testing

These tools are called from the QuPath QPSC extension GUI during microscope setup and calibration. They:
- Require live hardware connections
- Generate diagnostic plots and CSV data
- Test autofocus algorithms at current microscope position
- Validate autofocus parameter tuning

**Not intended for automated CI/CD** - these are interactive diagnostic tools.

### Automated Unit Tests

Automated pytest-compatible unit tests are located in the `tests/` directory:
- **`tests/test_autofocus_metrics.py`** - Tests for 13 autofocus metric calculations
- **`tests/test_tissue_detection.py`** - Tests for empty region detection algorithms
- **`tests/test_coordinate_validation.py`** - Tests for stage safety checks
- **`tests/test_config_manager.py`** - Tests for configuration management

These tests:
- Run without hardware (use synthetic test data)
- Can be integrated into CI/CD pipelines
- Test pure-function components (math, validation, parsing)

**Running Unit Tests:**

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run specific test file
pytest tests/test_autofocus_metrics.py

# Run with coverage report
pytest --cov=microscope_control --cov-report=html

# View coverage report
open htmlcov/index.html  # or xdg-open on Linux
```

**Test Coverage:**

Current automated tests achieve ~75-85% coverage for testable components:
- ✅ Autofocus metrics (all 13 metrics)
- ✅ Empty region detection (5 detection methods)
- ✅ Coordinate validation (safety-critical)
- ✅ Configuration management
- ⏸️ Hardware-dependent code (requires diagnostic tools, not unit tests)

## License

MIT License
