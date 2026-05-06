# Collections

Every persistent object in pymodal is a collection — an HDF5-backed, labelled, reference-aware set of items that share a single channel layout and a fixed number of domain axes. This page covers the behaviour shared by every collection class, regardless of whether the items are FRFs, time-series, or indicator results.

## Class hierarchy

```
_collection                       (collection_parent.py)
├── _collection_0d                 # items shaped (n_outputs, n_inputs)
│   └── sci_collection, drq_collection, frfrms_collection, frfsf_collection,
│       frfsm_collection, ods_diff_collection, r2_imag_collection,
│       aigac_collection, unsigned_sci_collection
├── _collection_1d                 # items shaped (D1, n_outputs, n_inputs)
│   ├── frf
│   ├── timeseries
│   └── rvac_collection, rvac_2d_collection, gac_collection, m2l_collection
├── _collection_2d                 # items shaped (D1, D2, n_outputs, n_inputs)
│   └── cfdac_collection, cfdac_a_collection, fdac_collection
└── _collection_3d                 # items shaped (D1, D2, D3, n_outputs, n_inputs)
                                   # reserved for future use
```

`_collection` is abstract; instantiate one of the four dimensional parents or — much more commonly — one of the concrete subclasses.

## Construction (general shape)

The dimensional parents all share the same constructor surface (with `domain_arrays` for 2-D / 3-D and `domain_array` for 1-D). The most common construction path is one of the public concrete classes — `frf`, `timeseries`, or one of the `*_collection` indicator classes (which are usually built via `frf.<indicator>()` rather than directly).

```python
import numpy as np
import pymodal

frfs = pymodal.frf(
    measurements=[arr_a, arr_b, arr_c],          # list → multi-item; single ndarray → 1-item
    freq_resolution=1.0,                         # or freq_array=...
    coordinates=np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]]),
    orientations=np.array([[0, 0, 1]] * 3),      # auto-normalised
    method="SIMO",
    measurements_units="millimeter / second ** 2 / newton",
    space_units="millimeter",
    names=["a", "b", "c"],
    labels=[0.0, 1.0, 1.0],
    references=None,
    reference_index_maps=None,
    path="frfs.h5",                              # auto-named if omitted
)
```

Item shape consistency is enforced at construction. All items must share the same shape (after the constructor adds trailing channel axes if needed) and `_n_domain_axes + 2` rank.

## Names and labels

- `coll.name` — list of strings, one per item. Auto-generated as `item_0`, `item_1`, …; duplicates get `_<n>` suffixes via `add_suffix`.
- `coll.labels` — `None` (no labels) or a list of HDF5 scalar datasets. Read with `coll.labels[i][()]`.

```python
print(coll.name)                                 # ['a', 'b', 'c']
print([lbl[()] for lbl in coll.labels])          # [0.0, 1.0, 1.0]
```

## Item selection (in-place)

`__getitem__` **mutates the active selection** and returns `self`. Supported keys:

| Key | Effect |
|---|---|
| `int` | Restrict to the item at that position. |
| `slice` | Restrict to the slice of names. |
| `str` | Restrict to the named item (raises `KeyError` if missing). |
| `list[str]` / `set[str]` / `tuple[str]` | Restrict to those names. |

```python
coll["item_b"]                                   # only "item_b"
coll[0:2]                                        # first two items
coll[["a", "c"]]                                 # named subset
coll.select_all()                                # restore full selection
```

> **In-place semantics.** `coll[1:3]` is destructive — it changes what `coll.name` and `coll.measurements` point at. The data on disk is untouched; call `coll.select_all()` to restore the full selection.

## Append

`append(item, name=None, label=None)` adds a single new item to the underlying HDF5 file. It enforces channel-shape compatibility, auto-names if you don't pass `name`, and gracefully promotes an unlabelled collection to labelled if you supply `label=`.

```python
coll.append(arr_d, name="d", label=1.0)
print(len(coll), coll.name[-1])                  # one more, "d"
```

Returns `self` for chaining.

## Stratified split

```python
train, val, test = coll.split(
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15,
    seed=42,
)
```

- Fractions must sum to 1.0.
- Stratification is done by `int(label)` — pass labels as integers (or close-to-integer floats) for the result to be meaningful.
- The indices are returned and also stored on the collection as `coll.train_indices`, `coll.val_indices`, `coll.test_indices`.

`split` requires labels. If you want a non-stratified random split, do it yourself with `numpy.random.default_rng(seed).permutation(len(coll))`.

## References (embed-by-copy)

References are how pymodal expresses provenance — every indicator collection carries the reference and damaged FRF collections that produced it; every FRF computed from time-series carries the input excitation and output response. Each link is **embedded by copy** into the consumer's HDF5 file: the source can move, drift, or be deleted without affecting the consumer.

API:

```python
coll.references                                  # {role: in-memory linked _collection}
coll.reference_index_maps                        # {role: list[int]}
coll.reference_roles()                           # ["reference", "damaged"]
coll.get_reference_data(role, i)                 # ndarray for item i's reference
```

Roles produced automatically:

| Producer | Roles |
|---|---|
| `timeseries.to_FRF(excitation)` | `"input"` (excitation), `"output"` (self) |
| `frf.<indicator>(reference)` | `"reference"`, `"damaged"` |

You can pass references explicitly to any concrete-class constructor via `references={...}` and `reference_index_maps={...}`. The `reference_index_maps` dict tells the collection which item of the reference each item corresponds to — defaults to `[0] * len(items)`.

The embed-by-copy guarantee is verified by `test_reference_survives_source_deletion`. The on-disk layout under `/references/{role}/` is documented in [Core Concepts](Core-Concepts).

## File lifecycle

```python
coll.path                                        # pathlib.Path of the HDF5 file
coll.file                                        # open h5py.File handle
coll.close()                                     # close handle AND delete file
coll.close(keep=True)                            # close handle, keep file on disk
coll.open()                                      # reopen the file (no-op if already open)
```

By default `close()` deletes the file — call it that way for short-lived working collections in scripts and notebooks. Pass `keep=True` to retain the file on disk (this is what `torch_dataset()` does internally).

## Attached files (model checkpoints, configs)

Collections can record an absolute path to an external file under a named role. The file itself is **not** copied — only the path is stored under `/attached/{role}` on the HDF5.

```python
import torch.nn as nn

model = nn.Linear(10, 2)
coll.save_model(model, role="detection")
print(coll.attached_roles())                     # ['detection']
print(coll.attached_path("detection"))           # absolute Path
restored = coll.load_model(nn.Linear, role="detection",
                            map_location="cpu", in_features=10, out_features=2)
```

Use this to bind a trained checkpoint to the dataset it was trained on, so future loads can recover the right state dict without the user having to remember its location.

## Domain edition

Each dimensional parent provides axis-specific edition methods:

| Class | Methods |
|---|---|
| `_collection_1d` | `change_domain_resolution(new_resolution)`, `change_domain_span(new_min=None, new_max=None)` |
| `_collection_2d` | same, plus `axis=0` or `axis=1` |
| `_collection_3d` | same, with `axis ∈ {0, 1, 2}` |

`frf` and `timeseries` add named aliases on top of these (`change_freq_span`, `change_freq_resolution`, `change_time_span`, `change_sampling_rate`).

All of these mutate the file in place and return `self`. Resampling that is not an integer multiple of the existing resolution triggers `numpy.interp` and emits a `UserWarning` (which the collection swallows internally).

## Channel and DOF metadata

```python
coll.method                                      # "SIMO", "MISO", "MIMO", "excitation"
coll.n_outputs, coll.n_inputs                    # int, int
coll.dof                                         # int (max of the two by default)
coll.coordinates                                 # (dof, 3) ndarray
coll.orientations                                # (dof, 3) unit-vector ndarray
coll.channel_shape()                             # (n_outputs, n_inputs)
coll.domain_shape()                              # (D1,), (D1, D2), () for 0-D, …
coll.measurements_units                          # str
coll.space_units                                 # str
coll.domain_arrays                               # list of 1-D ndarrays
coll.domain_units                                # list of str
```

These are populated at construction from the constructor arguments and the HDF5 attributes are kept in sync.

## PyTorch hand-off

```python
ds = coll.torch_dataset().dataset
```

`torch_dataset()` calls `close(keep=True)` on the HDF5 file, sets `coll.measurements = None` (Python-level handles are released), constructs a fresh `HDF5Dataset` over the same path, and binds it to `coll.dataset`. The returned `HDF5Dataset` is safe with `DataLoader(num_workers > 0)` because it opens a fresh file handle inside each `__getitem__` call. See [HDF5 Dataset](HDF5-Dataset) for the wrapper's full API.

## Common pitfalls

- **Empty `items` raises `ValueError`** — every collection must have at least one item.
- **MIMO requires `n_outputs == n_inputs`.** Use SIMO or MISO for non-square.
- **`method="excitation"` is `timeseries`-only.** `frf.__init__` raises if you pass it.
- **Item rank must match `_n_domain_axes + 2`.** `frf` items must be 3-D; CFDAC items must be 4-D; etc. The constructor will append trailing axes to lower-rank arrays but never strip them.
- **`__getitem__` is in-place.** Use `coll.select_all()` to restore full visibility.
- **Don't fork HDF5 file handles.** If you `import multiprocessing` and pass an open collection to a worker, reopen via `coll.open()` (or — preferred — let `HDF5Dataset` handle it for `DataLoader(num_workers > 0)`).
