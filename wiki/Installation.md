# Installation

`pymodal` requires **Python ≥ 3.9**. The library is pure-Python, but a few of its dependencies (`h5py`, `torch`, `audiomentations`) ship native wheels.

## Standard install

```bash
pip install pymodal
```

This pulls in the runtime dependencies declared in `setup.py`:

- `numpy>=1.18.1`
- `scipy>=1.4.1`
- `matplotlib>=3.1.3`
- `pandas>=1.0.1`
- `Pint>=0.20.1` — unit-aware computation
- `pyFRF>=0.40` — H1 / H2 / Hv / vector / ODS FRF estimators
- `h5py>=3.9.0` — on-disk persistence
- `audiomentations>=0.31.0` — Gaussian-noise augmentation
- `torch>=2.0.1` — `HDF5Dataset` and tensor I/O

## Development install

Clone the repository and install in editable mode with the `dev` extra:

```bash
git clone https://github.com/grcarmenaty/pymodal.git
cd pymodal
pip install -e .[dev]
```

The `dev` extra adds:

- `pytest`, `pytest-cov` — test suite
- `flake8`, `black` — linting / formatting
- `docutils`, `doc8` — RST validation

Run the test suite:

```bash
pytest tests/
```

The repository ships 84 tests across five files (`test_0_change_resolution.py`, `test_1_collection_parents.py`, `test_2_frf_timeseries.py`, `test_3_indicators.py`, `test_4_torch_dataset.py`). CI runs them on Python 3.9 and 3.10 across Ubuntu, macOS, and Windows.

## MCP extra (optional)

To expose pymodal as a Model Context Protocol server (so an LLM agent can drive collection construction, signal processing, indicator computation, and PyTorch handoff as tools), install the `mcp` extra:

```bash
pip install -e .[mcp]
```

This adds `mcp>=1.0`. Run the server with:

```bash
python -m pymodal.mcp
# or, after install:
pymodal-mcp
```

See [MCP Server](MCP-Server) for the full tool catalogue and a typical agent flow.

## Conda environment

The repository ships an `environment.yml` for users who prefer conda:

```bash
conda env create -f environment.yml
conda activate pymodal
```

## Verify the install

```python
import pymodal
import numpy as np

f = pymodal.frf(
    measurements=np.ones((100, 1, 1), dtype=complex),
    freq_resolution=1.0,
)
print(len(f), f.freq_array.shape, f.measurements_units)
# 1 (100,) millimeter / second ** 2 / newton
f.close()
```

If this prints without error, the install is functional.

## Notes

- `HDF5_USE_FILE_LOCKING=FALSE` is set at module import time in `collection_parent.py` to avoid HDF5 locking errors on networked or container filesystems. You can override it from the shell before importing pymodal if you need locking.
- `torch` is a hard runtime dependency because `_collection.torch_dataset()` returns a `torch.utils.data.Dataset`. If you only need offline pre-processing without PyTorch handoff, you can still import the library — `torch` is imported lazily inside `torch_dataset()` and `HDF5Dataset`.
- The legacy ANSYS module mentioned in the README is not part of the current codebase; ANSYS-flavoured FEA is handled externally via the `scenarios` builder (see [Quickstart](Quickstart) for the pattern).
