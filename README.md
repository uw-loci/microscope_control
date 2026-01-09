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

## License

MIT License
