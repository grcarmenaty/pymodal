# pymodal — Claude Instructions

## Core Mission

**pymodal's primary purpose is to make it easy to build large, labelled collections of vibrational signals stored on disk, and to feed those collections directly into PyTorch ML training pipelines — without loading the dataset into RAM.**

The intended workflow is:

1. Acquire or simulate vibrational measurements (time-series or FRFs).
2. Assemble them into a `timeseries` or `frf` collection (a single ndarray builds a 1-item collection; a list of ndarrays builds a multi-item collection). Every measurement array is written to an HDF5 file on disk as it is added.
3. Pre-process the collection in-place (resample, change span, augment with noise, convert to FRF) — all operations stream through the HDF5 file without materialising the full dataset in memory.
4. Optionally compute SHM indicators (CFDAC, RVAC, SCI, …) — each returns a *typed* indicator collection (e.g. `cfdac_collection`, `sci_collection`) with the source `reference` and `damaged` collections embedded as references for full provenance.
5. Call `.torch_dataset()` to obtain a `HDF5Dataset` (`torch.utils.data.Dataset`) that lazy-loads individual samples on demand during training.

Signal processing, metrics, simulation, and plotting are secondary capabilities that support building and validating those collections. They are not the end goal.

### What this means in practice

- **Collections are organised by domain dimensionality, not by signal type.** Every collection is a subclass of one of the four dimensional parents — `_collection_0d`, `_collection_1d`, `_collection_2d`, `_collection_3d` — and items always have shape `(D1?, D2?, D3?, n_outputs, n_inputs)`, with channel/DOF axes preserved at every dimensionality.
- **`frf` and `timeseries` ARE collections.** They subclass `_collection_1d`. Single-item construction is just a 1-item collection. There is no separate `frf_collection` or `timeseries_collection`.
- **HDF5 is the persistence layer.** Collections always have a backing `.h5` file. The file is created on construction and stays open. Never design around keeping large arrays in Python memory.
- **Labels are first-class.** Every item in a collection can carry a numeric label (e.g. damage state index) that is stored alongside the data in the HDF5 file and surfaced as the `y` in PyTorch `(x, y)` batches.
- **References are first-class.** Any collection can declare named reference collections (`{"input": …, "output": …}` for FRFs computed from time-series; `{"reference": …, "damaged": …}` for indicators). References are embedded by copy into the consumer's HDF5 file, so the file is self-contained and traceable.
- **Disk I/O throughput matters.** Batch operations on collections are intended to be parallelisable.

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
│   ├── collection_parent.py         # _collection — HDF5 layout, names, labels, references
│   ├── collection_0d.py             # _collection_0d — items shaped (n_outputs, n_inputs)
│   ├── collection_1d.py             # _collection_1d — items shaped (D1, n_outputs, n_inputs)
│   ├── collection_2d.py             # _collection_2d — items shaped (D1, D2, n_outputs, n_inputs)
│   ├── collection_3d.py             # _collection_3d — items shaped (D1, D2, D3, n_outputs, n_inputs)
│   ├── timeseries.py                # timeseries(_collection_1d) — time-domain
│   ├── frf.py                       # frf(_collection_1d) — frequency-domain
│   ├── indicators_0d.py             # sci/drq/aigac/frfrms/frfsf/frfsm/ods_diff/r2_imag/unsigned_sci collections
│   ├── indicators_1d.py             # rvac/rvac_2d/gac/m2l collections
│   ├── indicators_2d.py             # cfdac/cfdac_a/fdac collections
│   └── hdf5_dataset.py              # HDF5Dataset — PyTorch Dataset wrapper
├── tests/
│   ├── test_0_change_resolution.py  # 48 parametric tests for change_domain_resolution
│   ├── test_1_collection_parents.py # Dimensional-parent contract: shape, layout, refs, append, split
│   ├── test_2_frf_timeseries.py     # frf and timeseries on _collection_1d
│   ├── test_3_indicators.py         # Typed indicator collections (0D/1D/2D)
│   ├── test_4_torch_dataset.py      # HDF5Dataset round-trips for variable-rank items
│   └── aux_test_utils.py            # Test helper for generating synthetic arrays
├── setup.py
├── requirements.txt
├── requirements-dev.txt
└── environment.yml
```

## Class Hierarchy

```
_collection  (collection_parent.py)        # HDF5 file, names, labels, references
├── _collection_0d   (collection_0d.py)    # items shaped (n_outputs, n_inputs)
│   └── *_collection (indicators_0d.py)    # sci, drq, frfrms, frfsf, frfsm, …
├── _collection_1d   (collection_1d.py)    # items shaped (D1, n_outputs, n_inputs)
│   ├── frf          (frf.py)              # 1-D over frequency
│   ├── timeseries   (timeseries.py)       # 1-D over time
│   └── *_collection (indicators_1d.py)    # rvac, rvac_2d, gac, m2l
├── _collection_2d   (collection_2d.py)    # items shaped (D1, D2, n_outputs, n_inputs)
│   └── *_collection (indicators_2d.py)    # cfdac, cfdac_a, fdac
└── _collection_3d   (collection_3d.py)    # reserved (e.g. time-resolved CFDAC)

HDF5Dataset  (hdf5_dataset.py)   ← torch.utils.data.Dataset
```

## Core Design Decisions

### Item array shape, by dimensionality
Item shape is `(D1?, D2?, D3?, n_outputs, n_inputs)` — channel axes are always last, and channel/DOF metadata is preserved at every dimensionality:

- 0-D: `(n_outputs, n_inputs)` — one scalar per channel pair
- 1-D: `(D1, n_outputs, n_inputs)` — `frf`, `timeseries`, vector indicators
- 2-D: `(D1, D2, n_outputs, n_inputs)` — matrix indicators (CFDAC, FDAC)
- 3-D: `(D1, D2, D3, n_outputs, n_inputs)` — reserved

The dimensionality is fixed by the class via `_n_domain_axes` and enforced in `_collection.__init__`.

### Method types
The `method` attribute controls how the channel axes are interpreted:
- `"SIMO"` — single input, multiple outputs
- `"MISO"` — multiple inputs, single output
- `"MIMO"` — multiple inputs and outputs (must be square)
- `"excitation"` — input-only signal; only valid for `timeseries`, not `frf`

### Units
Default units (set per concrete class):
- `frf.measurements_units` → `"millimeter / second ** 2 / newton"` (accelerance)
- `frf.freq_units` → `"hertz"`
- `timeseries.measurements_units` → `"meter / second ** 2"` (acceleration), or `"newton"` when `method="excitation"`
- `timeseries.time_units` → `"second"`

Units are stored as plain strings on the HDF5 attrs; pint is used for unit-aware computation when needed (e.g. dimensional analysis in `timeseries.to_FRF`).

### HDF5 layout
Every collection writes a file with this structure:

```
/measurements/
    .attrs: method, n_domain_axes, n_outputs, n_inputs, dof,
            coordinates, orientations, measurements_units, space_units
    _axes/
        axis_{k}/data         (one per domain axis, with .attrs["units"])
    {name}/
        data                  (rank = n_domain_axes + 2)
        label                 (optional scalar)
/references/
    {role}/
        .attrs: mode="embedded", index_map, collection_class, n_domain_axes
        measurements/         (full embedded snapshot of the linked collection)
```

The HDF5 file stays open for the lifetime of the collection. Call `.close()` to release; `.open()` to reacquire.

### Reference linking (embed-by-copy)
Any collection can register named reference collections, and each reference is **embedded by copy** into the consumer's HDF5 file under `/references/{role}/`. This makes the consumer self-contained — the source file can move, be deleted, or drift, and the consumer's view of its references stays intact (verified by `test_reference_survives_source_deletion`).

Standard roles emerge from the producer:

- `timeseries.to_FRF(excitation)` registers `{"input": excitation, "output": self}` on the resulting `frf`.
- Indicator methods on `frf` (e.g. `dmg.cfdac(reference)`) register `{"reference": reference, "damaged": self}` on the resulting indicator collection.

Use `coll.references[role]` for the in-memory linked object and `coll.get_reference_data(role, i)` to read item *i*'s reference array directly from the embedded snapshot.

### In-place mutation pattern
Collection edition methods (`change_domain_span`, `change_domain_resolution`, `__getitem__` with strings/slices, `append`) mutate the HDF5 file and return `self` for chaining. There is no longer a "deep copy on edit" path — single-item collections are just 1-item collections.

## Key Modules

### utils.py
Pure functions only. Grouped by concern:
- **Domain manipulation**: `change_domain_resolution`, `change_domain_span`
- **Plotting**: `lineplot`, `plot_control_chart`
- **Array I/O**: `save_array`, `load_array` (supports `.npy`, `.npz`, `.mat`)
- **FRF comparison metrics**: `value_CFDAC`, `value_CFDAC_A`, `value_FDAC`, `value_RVAC`, `value_RVAC_2d`, `value_GAC`
- **SHM scalar metrics**: `FRFRMS`, `FRFSF`, `FRFSM`, `ODS_diff`, `r2_imag`, `SCI`, `unsigned_SCI`, `DRQ`, `AIGAC`, `M2L`, `M2L_func`
- **Simulation**: `synthetic_FRF`, `modal_superposition`, `damping_coefficient`

These are the single source of truth — the indicator collection classes delegate to them.

### collection_parent.py — `_collection`
Never instantiated directly. Owns HDF5 file lifecycle, `name`/`labels` lists, channel metadata (coordinates/orientations/method), reference embedding and retrieval, item selection, `append`, `split`, `torch_dataset`, `open`/`close`. Subclasses declare `_n_domain_axes` and provide axis-specific edition methods.

### collection_0d.py / 1d.py / 2d.py / 3d.py
Dimensional parents. `_collection_1d`, `_collection_2d`, `_collection_3d` each provide `change_domain_resolution(axis=k)` and `change_domain_span(new_min, new_max, axis=k)` that mutate every item and the corresponding axis in `/measurements/_axes/`.

### frf.py — `frf` (subclass of `_collection_1d`)
1-D collection with frequency as the domain axis. Adds frequency aliases (`freq_array`, `freq_units`, `freq_start/end/span/resolution`), `change_freq_span`/`change_freq_resolution`, `_as_matrix(i)` for indicator math, `plot`/`waterfall`, and indicator methods (`cfdac`, `sci`, `rvac`, …) that return typed indicator collections with reference linking already wired.

### timeseries.py — `timeseries` (subclass of `_collection_1d`)
1-D collection with time as the domain axis. Adds time aliases (`time_array`, `sampling_rate`, `time_step`, …), `change_time_span`/`change_sampling_rate`, `to_FRF` (registers references), and `AddGaussianNoise` for augmentation.

### indicators_0d.py / 1d.py / 2d.py
One subclass per SHM indicator family. Each declares a `_indicator_op` (delegating to a pure function in `utils`); the shared `from_pair(reference, damaged)` classmethod on the family base computes the metric per item and stores the result with the correct dimensionality and embedded references.

### hdf5_dataset.py — `HDF5Dataset`
PyTorch `Dataset` wrapper. Reads `/measurements/{name}/data` and optional `/measurements/{name}/label`, skipping the `_axes` group. Supports lazy loading and an optional RAM cache. Item rank is preserved exactly as written, so 0-D / 1-D / 2-D collections all yield items of their natural rank.

## Testing

```bash
pytest tests/
```

Five test files, 84 tests total:
- `test_0_change_resolution.py` — 48 parametric tests for `change_domain_resolution`
- `test_1_collection_parents.py` — 14 tests for the dimensional-parent contract (shape, layout, references, item selection, append, split, embed-and-survive)
- `test_2_frf_timeseries.py` — 9 tests for `frf` and `timeseries` on `_collection_1d`, including `to_FRF` reference registration
- `test_3_indicators.py` — 9 tests for typed indicator collections (0-D / 1-D / 2-D)
- `test_4_torch_dataset.py` — 4 tests for `HDF5Dataset` round-trips across all dimensionalities

Tests pass on Python 3.9+. The CI workflow tests Python 3.9 and 3.10 on ubuntu, macos, and windows.

## Development Notes

- Python ≥ 3.9 required. Use `Optional[X]` from `typing` for optional type hints — **not** `X | None` (which requires Python 3.10+).
- The active development branch is `master`. The old `signal` branch has been deleted. The original pre-rewrite codebase is preserved as `legacy`.
- Indicator collections store results with channel shape `(1, 1)` even though their source FRFs may have many channels. Channel/DOF metadata is recovered from the embedded `reference` and `damaged` references.
- `HDF5_USE_FILE_LOCKING=FALSE` is set as an environment variable in `collection_parent.py` to prevent HDF5 file locking issues on some filesystems.
- The legacy `signal_parent`, `signal_collection_parent`, `frf_collection`, `timeseries_collection`, and `indicator_collection` modules have been removed. `frf_collection` and `timeseries_collection` no longer exist as separate classes — `frf` and `timeseries` accept either a single ndarray (1-item) or a list of ndarrays (multi-item).

## Git and Authorship

**NEVER appear as the commit author.** Every commit must be authored by the repository owner (Guillermo Reyes Carmenaty `<grcarmenaty@gmail.com>`). Before making any commit, verify that the git user identity is set to the owner's name and email — not to Claude, Anthropic, or any AI identity. If the local `user.name` / `user.email` config does not match the owner's identity, set it explicitly with:

```bash
git config user.name "Guillermo Reyes Carmenaty"
git config user.email "grcarmenaty@gmail.com"
```

This must be checked at the start of every session and before every commit.
