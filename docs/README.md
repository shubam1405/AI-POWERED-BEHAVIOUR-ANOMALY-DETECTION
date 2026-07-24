# Virtual Enterprise Simulator Foundation

This directory contains the documentation for the Virtual Enterprise foundation setup for the anomaly detection simulator.

## Getting Started

### Prerequisites
Install the required dependencies (primarily `faker` for data generation):
```bash
pip install faker
```

### Running the Generator
To generate the virtual enterprise in-memory, validate it, and export the CSV outputs to `data/raw/`:
```bash
python main.py
```

### Running the Test Suite
To verify the system validation rules, orphan tracking, duplicate checks, and distributions:
```bash
python -m unittest src/tests/test_validation.py
```

## Directory Layout
* `src/config/`: Configuration parameters (`config.py`).
* `src/models/`: Type-safe Enum classes and core entity dataclasses.
* `src/generators/`: Modular data generators (Employees, Devices, Resources, Enterprise).
* `src/simulator/`: Company representation and clock controls.
* `src/exporters/`: CSV serializing exporter.
* `src/utils/`: Logger initialization and strict validator.
* `src/tests/`: Unit test suite.
* `data/raw/`: Generated CSV data files (`employees.csv`, `devices.csv`, `resources.csv`).
