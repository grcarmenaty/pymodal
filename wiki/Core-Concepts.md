# Core Concepts

This page is the conceptual model behind every other page in the wiki. Read it once and the API will fall into place.

## 1. Collections are organised by domain dimensionality

Every concrete collection in pymodal is a subclass of one of four **dimensional parents**:

| Parent | Item shape | Domain axes | Example concrete classes |
|---|---|---|---|
| `_collection_0d` | `(n_outputs, n_inputs)` | 0 | `sci_collection`, `drq_collection`, `frfrms_collection`, … |
| `_collection_1d` | `(D1, n_outputs, n_inputs)` | 1 | `frf`, `timeseries`, `rvac_collection`, `gac_collection`, … |
| `_collection_2d` | `(D1, D2, n_outputs, n_inputs)` | 2 | `cfdac_collection`, `fdac_collection`, … |
| `_collection_3d` | `(D1, D2, D3, n_outputs, n_inputs)` | 3 | reserved (e.g. time-resolved CFDAC) |

The dimensionality is fixed by the class via `_n_domain_axes` and enforced in `_collection.__init__`. The **channel/DOF axes are always last** at every dimensionality, so channel metadata stays meaningful end-to-end.

## 2. `frf` and `timeseries` ARE collections

There is no separate `frf_collection` or `timeseries_collection` class. The same class accepts either a single `ndarray` (1-item collection) or a list of `ndarray`s (multi-item collection):

```python
import numpy as np
import pymodal

# 1-item collection
single = pymodal.frf(np.ones((100, 2, 1), dtype=complex), freq_resolution=1.0)

# multi-item collection (3 items, identical shape)
multi = pymodal.frf(
    [np.ones((100, 2, 1), dtype=complex) for _ in range(3)],
    freq_resolution=1.0,
)
```

Both subclass `_collection_1d`. Domain edition methods (`change_freq_span`, `change_freq_resolution`, …) operate in place on every item. There is no "deep copy on edit" path.

## 3. HDF5 is the persistence layer

Every collection has a backing `.h5` file that is created on construction and stays open for the lifetime of the object. The file is the canonical state — Python attributes mirror it but the disk is the source of truth.

The HDF5 layout is fixed and predictable:

```
/measurements/
    .attrs:   method, n_domain_axes, n_outputs, n_inputs, dof,
              coordinates, orientations,
              measurements_units, space_units
    _axes/
        axis_{k}/data         (one per domain axis, with .attrs["units"])
    {name}/
        data                  (rank = n_domain_axes + 2)
        label                 (optional scalar)
/references/
    {role}/
        .attrs:   mode = "embedded",
                  index_map (int64 array),
                  collection_class,
                  n_domain_axes
        measurements/         (full embedded snapshot of the linked collection)
/attached/
    .attrs:   {role}: <absolute path to a saved file, e.g. PyTorch state dict>
```

You can poke at any pymodal HDF5 file with `h5dump` or `h5py` directly without going through the library — the layout is portable and self-describing.

Call `.close()` to release the handle (deletes the file by default — pass `keep=True` to retain it). Call `.open()` to reacquire.

## 4. Items, channels, and the `method` attribute

Every item ends with two channel axes: `(n_outputs, n_inputs)`. The `method` attribute tells you how to interpret them:

| `method` | Meaning |
|---|---|
| `"SIMO"` | Single Input, Multiple Outputs — default. |
| `"MISO"` | Multiple Inputs, Single Output. |
| `"MIMO"` | Multiple Inputs, Multiple Outputs (must be square: `n_outputs == n_inputs`). |
| `"excitation"` | Force-input signal. **Only valid for `timeseries`.** Marks the collection as the input side for `timeseries.to_FRF`. |

Channel/DOF metadata travels with the collection:

- `coordinates` — `(dof, 3)` spatial positions.
- `orientations` — `(dof, 3)` unit-vector orientations (auto-normalised at construction).
- `dof` — number of distinct DOFs (defaults to `max(n_outputs, n_inputs)`).

Indicator collections store results with channel shape `(1, 1)` even when the source FRFs had many channels — the channel metadata is recovered via the embedded `reference` and `damaged` references rather than duplicated in the indicator values themselves.

## 5. References are first-class (embed-by-copy)

Any collection can register named **reference collections**, and each reference is **embedded by copy** into the consumer's HDF5 file under `/references/{role}/`. Once embedded:

- The consumer is **self-contained** — the source file can be moved, renamed, or deleted without affecting the consumer.
- Both the in-memory linked object (`coll.references[role]`) and the on-disk snapshot are available.
- Per-item linking is encoded in `reference_index_maps[role]` (one int per item, pointing into the referenced collection).

Standard roles emerge from the producer:

| Producer | Roles registered |
|---|---|
| `timeseries.to_FRF(excitation)` | `"input"` → excitation, `"output"` → self |
| `frf.cfdac(reference)` (and other indicator methods) | `"reference"` → reference, `"damaged"` → self |

Read references with:

```python
indicator.references["reference"]                  # in-memory frf object
indicator.get_reference_data("reference", i)       # ndarray for item i
indicator.reference_roles()                        # ["reference", "damaged"]
```

The embed-by-copy guarantee is verified by `test_reference_survives_source_deletion` in the test suite.

## 6. Labels are first-class

Every item can carry a numeric `label` (typically a damage-state class index). Labels are stored under `/measurements/{name}/label` in the HDF5 file, surfaced as `coll.labels` (a list of HDF5 datasets), and returned as the `y` in `(x, y)` PyTorch batches.

```python
frfs = pymodal.frf(
    [arr0, arr1, arr2],
    freq_resolution=1.0,
    labels=[0.0, 1.0, 1.0],
)
frfs.split(train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
# stratified split — frfs.train_indices / val_indices / test_indices
```

Pass `labels=None` (the default) to build an unlabelled collection. Appending an item with `label=` to a previously unlabelled collection promotes the whole collection to labelled (existing items get `NaN`).

## 7. In-place mutation pattern

Edition methods on collections (`change_freq_span`, `change_domain_resolution`, `__getitem__` with strings/slices, `append`, `AddGaussianNoise`) **mutate the HDF5 file and return `self`** — they are chainable:

```python
(timeseries_collection
    .change_sampling_rate(8192)
    .change_time_span(new_min_time=0.1, new_max_time=2.0)
    .AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01))
```

There is intentionally no "save a fresh copy" path on the collection itself — the file is the state. If you need a copy, point the constructor at a new `path=` and pass the data through.

## 8. Units

Units are stored as plain strings on the HDF5 attributes. Pint is used internally for unit-aware computation only where it matters (e.g. dimensional analysis in `timeseries.to_FRF`).

Defaults per concrete class:

| Class | `measurements_units` | Domain units |
|---|---|---|
| `frf` | `"millimeter / second ** 2 / newton"` (accelerance) | `"hertz"` |
| `timeseries` (response) | `"meter / second ** 2"` (acceleration) | `"second"` |
| `timeseries` (`method="excitation"`) | `"newton"` | `"second"` |
| Indicator collections | `""` (dimensionless) | DOF index for 1-D / 2-D variants |

Override any of these at construction time by passing the corresponding keyword argument.

## 9. PyTorch hand-off

`coll.torch_dataset()` closes the HDF5 file (keeping it on disk), constructs an `HDF5Dataset` over the same path, and binds it as `coll.dataset`. The dataset:

- Is a `torch.utils.data.Dataset` — works with any `DataLoader`.
- Lazy-loads each sample via a fresh read-only handle on `__getitem__`, so it is safe with `num_workers > 0`.
- Returns `(x, y)` pairs where `y` is the per-item label tensor.
- Optionally caches all arrays in RAM (`HDF5Dataset(path, load_data=True)`).
- Preserves item rank exactly as written — 0-D / 1-D / 2-D collections all yield items of their natural rank.

See [HDF5 Dataset](HDF5-Dataset) for the full API.

## What this looks like together

```python
# 1. Build a labelled response collection from raw arrays.
resp = pymodal.timeseries(
    measurements=[acc_pristine, acc_damaged_a, acc_damaged_b],
    time_step=1/8192,
    labels=[0.0, 1.0, 1.0],
)

# 2. Build the matching excitation collection.
exc = pymodal.timeseries(
    measurements=[force_pristine, force_damaged_a, force_damaged_b],
    time_step=1/8192,
    method="excitation",
)

# 3. Compute FRFs (writes a new HDF5 file with input/output references embedded).
frfs = resp.to_FRF(exc, FRF_type="H1")

# 4. Compute a SHM indicator vs the pristine baseline.
sci = frfs[1:].sci(reference=frfs[:1])

# 5. Hand off to PyTorch.
ds = sci.torch_dataset().dataset
x, y = ds[0]
```

That is the whole library in nine lines. Every later page in the wiki is detail and edge-case behaviour around this pattern.
