# Microscope Control

Hardware control library for microscopes via Pycromanager (Micro-Manager).

## Features

- **Hardware Abstraction**: Platform-independent microscope control interface
- **Pycromanager Integration**: Full Micro-Manager hardware support
- **Autofocus**: Multiple autofocus algorithms and metrics
- **Stage Control**: XYZ stage positioning and movement
- **Configuration Management**: YAML-based microscope configuration

## Installation

```bash
pip install microscope-control
```

Requires Micro-Manager to be installed and running on the system.

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
