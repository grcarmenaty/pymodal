# pymodal — Claude Instructions

## Core Mission

**pymodal's primary purpose is to make it easy to build large, labelled collections of vibrational signals stored on disk, and to feed those collections directly into PyTorch ML training pipelines — without loading the dataset into RAM.**

The intended workflow is:

1. Acquire or simulate vibrational measurements (time-series or FRFs).
2. Assemble them into a `timeseries_collection` or `frf_collection`, which writes every measurement array to an HDF5 file on disk as it is added.
3. Pre-process the collection in-place (resample, change span, augment with noise, convert to FRF) — all operations stream through the HDF5 file without materialising the full dataset in memory.
4. Call `.torch_dataset()` to obtain a `HDF5Dataset` (`torch.utils.data.Dataset`) that lazy-loads individual samples on demand during training.

Signal processing, metrics, simulation, and plotting are secondary capabilities that support building and validating those collections. They are not the end goal.

### What this means in practice

- **Collections are the primary data structure**, not individual signals. Individual `frf` and `timeseries` objects exist to be assembled into collections.
- **HDF5 is the persistence layer.** Collections always have a backing `.h5` file. The file is created on construction and stays open. Never design around keeping large arrays in Python memory.
- **Labels are first-class.** Every signal in a collection can carry a numeric label (e.g. damage state index) that is stored alongside the data in the HDF5 file and surfaced as the `y` in PyTorch `(x, y)` batches.
- **Disk I/O throughput matters.** Batch operations on collections (`change_freq_span`, `AddGaussianNoise`, etc.) are designed to be parallelisable (multiprocessing infrastructure is scaffolded but currently sequential — do not remove the commented `Pool` blocks).

## Secondary Capabilities

- Structured signal containers (`frf`, `timeseries`) with full unit awareness via pint
- FRF estimation from time-domain data (pyFRF: H1, H2, Hv, vector, ODS estimators)
- Signal processing utilities: resolution change, domain span change, interpolation
- SHM damage-detection metrics: CFDAC, FDAC, RVAC, GAC, SCI, DRQ, AIGAC, FRFRMS, FRFSF, FRFSM, M2L
- Synthetic FRF and time-series generation via modal superposition
- Data augmentation: Gaussian noise injection (`AddGaussianNoise`) via audiomentations

## Repository Structure

```
pymodal/
├── pymodal/
│   ├── __init__.py                  # Public API exports
│   ├── utils.py                     # Signal processing, metrics, I/O, plotting
│   ├── signal_parent.py             # Base class _signal (shared by frf and timeseries)
│   ├── frf.py                       # frf class — frequency response function container
│   ├── timeseries.py                # timeseries class — time-domain signal container
│   ├── signal_collection_parent.py  # Base class _signal_collection (HDF5-backed)
│   ├── frf_collection.py            # frf_collection class
│   ├── timeseries_collection.py     # timeseries_collection class
│   └── hdf5_dataset.py              # HDF5Dataset — PyTorch Dataset wrapper
├── tests/
│   ├── test_0_change_resolution.py  # 48 parametric tests for change_domain_resolution
│   └── aux_test_utils.py            # Test helper for generating synthetic arrays
├── setup.py
├── requirements.txt
├── requirements-dev.txt
└── environment.yml
```

## Class Hierarchy

```
_signal  (signal_parent.py)
├── frf             (frf.py)
└── timeseries      (timeseries.py)

_signal_collection  (signal_collection_parent.py)
├── frf_collection          (frf_collection.py)
└── timeseries_collection   (timeseries_collection.py)

HDF5Dataset  (hdf5_dataset.py)   ← torch.utils.data.Dataset
```

## Core Design Decisions

### Measurement array shape
All `measurements` arrays are 3-D: `(domain_samples, outputs, inputs)`. Single-channel signals are stored as `(N, 1, 1)`. This is enforced in `_signal.__init__` and never changes across the class hierarchy.

### Method types
The `method` attribute controls how the second and third dimensions are interpreted:
- `"SIMO"` — single input, multiple outputs; indexing slices the output axis
- `"MISO"` — multiple inputs, single output; indexing slices the input axis
- `"MIMO"` — multiple inputs and outputs; indexing slices the output axis by default
- `"excitation"` — input-only signal; only valid for `timeseries`, not `frf`

### Units
All numeric quantities are pint `Quantity` objects. Default units:
- `frf.measurements_units` → `"millimeter / second ** 2 / newton"` (accelerance)
- `frf.freq_units` → `"hertz"`
- `timeseries.measurements_units` → `"meter / second ** 2"` (acceleration)
- `timeseries.time_units` → `"second"`

### HDF5 collections
`_signal_collection` stores each signal's measurement array in an HDF5 file under `measurements/{name}/data`. The HDF5 file stays open for the lifetime of the collection. Attributes shared by all signals are stored in `measurements.attrs`. Callers must call `.close()` when done to release the file handle.

### Deep copy pattern
Individual signal operations (`change_domain_resolution`, `change_domain_span`, `__getitem__`) return deep copies; they never mutate the original. Collection operations mutate the HDF5 file in-place and return `self` for chaining.

## Key Modules

### utils.py
Contains all standalone functions. Grouped by concern:
- **Domain manipulation**: `change_domain_resolution`, `change_domain_span`
- **Plotting**: `lineplot`, `plot_control_chart`
- **Array I/O**: `save_array`, `load_array` (supports `.npy`, `.npz`, `.mat`)
- **FRF comparison metrics**: `value_CFDAC`, `value_CFDAC_A`, `value_FDAC`, `value_RVAC`, `value_RVAC_2d`, `value_GAC`
- **SHM scalar metrics**: `FRFRMS`, `FRFSF`, `FRFSM`, `ODS_diff`, `r2_imag`, `SCI`, `unsigned_SCI`, `DRQ`, `AIGAC`, `M2L`, `M2L_func`
- **Simulation**: `synthetic_FRF`, `modal_superposition`, `damping_coefficient`

### signal_parent.py — `_signal`
Never instantiated directly. Shared logic for both `frf` and `timeseries`. Key methods:
- `__init__`: validates and normalises all inputs; generates default coordinates/orientations
- `__getitem__`: slices outputs or inputs depending on `method`, returns deep copy
- `change_domain_resolution` / `change_domain_span`: wrappers around utils functions
- `plot`: thin wrapper over `lineplot` with unit-aware labels

### frf.py — `frf`
Subclass of `_signal` for frequency-domain data. Adds frequency-specific terminology (`freq_array`, `freq_units`, `freq_start`, etc. as aliases), stricter method validation (no `"excitation"`), and a richer `plot` method supporting formats: `"mod"`, `"phase"`, `"mod-phase"`, `"real"`, `"imag"`, `"real-imag"`.

### timeseries.py — `timeseries`
Subclass of `_signal` for time-domain data. Adds time-specific aliases (`time_array`, `sampling_rate`, etc.) and `to_FRF(excitation, FRF_type, resp_delay)` which delegates to pyFRF to compute H1, H2, Hv, vector, or ODS FRF estimates.

### signal_collection_parent.py — `_signal_collection`
HDF5-backed container for lists of signals. All signals in a collection must share the same non-measurement attributes (coordinates, units, method, etc.). Key design notes:
- Constructor asserts attribute consistency across the input list
- `__getitem__` modifies the HDF5 file in-place when slicing spatially
- `append` validates attribute consistency before adding
- `torch_dataset()` closes the file and wraps it in `HDF5Dataset` for ML use

### frf_collection.py and timeseries_collection.py
Thin subclasses of `_signal_collection` that add domain-specific batch operations (`change_freq_span`, `change_freq_resolution` / `change_time_span`, `change_sampling_rate`), collection-level `plot`, and `timeseries_collection.to_FRF` for batch FRF estimation and `AddGaussianNoise` for data augmentation.

### hdf5_dataset.py — `HDF5Dataset`
`torch.utils.data.Dataset` wrapper over the HDF5 file written by `_signal_collection`. Supports lazy loading with an LRU file-level cache (`data_cache_size` files). Expected HDF5 structure: `/measurements/{name}/data` and optionally `/measurements/{name}/label`.

## Testing

The only test file is `tests/test_0_change_resolution.py`, which has 48 parametric tests covering `change_domain_resolution` across 1D–4D arrays, real and complex data, and multiple resolution scenarios. Run with:

```bash
pytest tests/
```

Tests pass on Python 3.9+. The CI workflow tests Python 3.9 and 3.10 on ubuntu, macos, and windows.

## Development Notes

- Python ≥ 3.9 required. Use `Optional[X]` from `typing` for optional type hints — **not** `X | None` (which requires Python 3.10+).
- The active development branch is `master` (formerly `signal`). The old `master` is preserved as `legacy`.
- Multiprocessing is scaffolded in collection classes (commented-out `Pool` calls) but currently runs sequentially. Do not remove the commented blocks — they document the intended parallelism.
- `HDF5_USE_FILE_LOCKING=FALSE` is set as an environment variable in `signal_collection_parent.py` to prevent HDF5 file locking issues on some filesystems.

## Git and Authorship

**NEVER appear as the commit author.** Every commit must be authored by the repository owner (Guillermo Reyes Carmenaty `<grcarmenaty@gmail.com>`). Before making any commit, verify that the git user identity is set to the owner's name and email — not to Claude, Anthropic, or any AI identity. If the local `user.name` / `user.email` config does not match the owner's identity, set it explicitly with:

```bash
git config user.name "Guillermo Reyes Carmenaty"
git config user.email "grcarmenaty@gmail.com"
```

This must be checked at the start of every session and before every commit.
