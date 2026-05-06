# HDF5 Dataset

`pymodal.HDF5Dataset` is a `torch.utils.data.Dataset` over one or more pymodal HDF5 files. It is the bridge between an offline-built collection and a PyTorch training loop. Source: `pymodal/hdf5_dataset.py`.

## Construction

You almost never instantiate `HDF5Dataset` directly — call `coll.torch_dataset()` and read `coll.dataset`. But the underlying constructor is also public:

```python
import pymodal

ds = pymodal.HDF5Dataset(
    file_path="frfs.h5",     # single .h5/.hdf5 file or a directory
    recursive=False,         # search subdirectories when file_path is a directory
    load_data=False,         # pre-load all arrays into RAM at construction
    transform=None,          # callable(ndarray) -> torch.Tensor
)
```

`file_path` can be:

- A single `.h5` or `.hdf5` file written by a pymodal collection.
- A directory — every `.h5`/`.hdf5` file in it is included, with `recursive=True` walking subdirectories.

All files in the dataset must share the same HDF5 layout: items under `/measurements/{name}/data` and (optionally) `/measurements/{name}/label`. The `_axes` group at `/measurements/_axes/` is automatically skipped — it carries the collection's domain axes, not items.

## Returned tensors

```python
x, y = ds[i]
```

- `x` is a `torch.Tensor` of the item's natural rank — `(n_outputs, n_inputs)` for a 0-D collection, `(D1, n_outputs, n_inputs)` for `frf` / `timeseries`, `(D1, D2, n_outputs, n_inputs)` for CFDAC/FDAC, etc. The dtype mirrors what was written to disk (typically `float64` for indicators, `complex64` for raw FRFs).
- `y` is a `torch.Tensor` of the per-item label scalar.

Item rank is preserved exactly as written — 0-D / 1-D / 2-D collections all yield items of their natural rank. This is verified by `test_4_torch_dataset.py`.

If you supplied a `transform=` callable at construction, it is applied to the numpy array before the tensor conversion (and your callable is responsible for the conversion to `torch.Tensor`).

## Lazy vs. RAM-cached loading

By default the dataset is lazy: each `__getitem__` opens the HDF5 file in read-only mode (`"r"`), reads exactly the requested item, and closes the file again. This makes the dataset safe with `DataLoader(num_workers > 0)` because each worker opens its own file handle.

Pass `load_data=True` to pre-load every array into a Python dict at construction time:

```python
ds = pymodal.HDF5Dataset("frfs.h5", load_data=True)
```

Use this when the full dataset fits comfortably in RAM and you want to eliminate per-sample disk I/O. Use the default (lazy) when the dataset is large or you are running multi-GPU training where each process loads its own copy.

## DataLoader integration

```python
import torch
from torch.utils.data import DataLoader

loader = DataLoader(
    ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

for x, y in loader:
    ...
```

`HDF5Dataset` is safe with `num_workers > 0` because every worker opens its own file handle on demand.

## `pad_collate_fn` for variable-length items

If your collection contains items with different lengths along the last domain axis (e.g. after asymmetric `change_freq_span` calls), use the bundled collate function:

```python
loader = DataLoader(
    ds,
    batch_size=8,
    collate_fn=pymodal.HDF5Dataset.pad_collate_fn,
)
```

This stacks the batch by zero-padding the **last dimension** of every `x` to the longest length in the batch. The result has shape `(batch, ..., max_len)`. Labels are stacked as-is.

`pad_collate_fn` only works for collections where every item has the same rank and only the last axis varies. For more complex padding, write your own collate.

## Stratified train/val/test loaders

Use the indices returned by `coll.split(...)` together with `Subset`:

```python
from torch.utils.data import Subset, DataLoader

train_idx, val_idx, test_idx = coll.split(seed=42)
ds = coll.torch_dataset().dataset

train_ds = Subset(ds, train_idx)
val_ds   = Subset(ds, val_idx)
test_ds  = Subset(ds, test_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)
```

`coll.split` is stratified by `int(label)`, so each subset has the original class distribution preserved.

## Reading raw metadata

Bypassing the PyTorch wrapper:

```python
ds.data_info             # list of dicts: {file_path, name, type, shape, hdf5_path}
ds.get_data_infos("data")     # the data entries only
ds.get_data_infos("label")    # the label entries only
ds.get_data("data", i)        # ndarray of item i (uses RAM cache if loaded)
ds.get_data("label", i)       # scalar ndarray of label i
len(ds)                       # total number of items across all files
```

These are useful for debugging or for building custom indexing strategies.

## Multiple files in one dataset

```python
ds = pymodal.HDF5Dataset("collections_dir/", recursive=True)
```

This is the canonical pattern for very large datasets that cannot live in a single HDF5 file: write one `.h5` per logical chunk (a recording session, a damage class, a sensor location) and merge them at training time.

The dataset opens each file fresh on every access — there is no advantage in keeping all data in one giant `.h5` versus many smaller ones.

## Saving and reloading model checkpoints

`HDF5Dataset` itself does not store model weights, but the collection that produced the file does. Use `coll.save_model(model, role="detection")` to record an absolute path to a saved state dict under `/attached/{role}` on the HDF5, then `coll.load_model(ModelClass, role="detection", **init_kwargs)` to recover it. See [Collections](Collections#attached-files-model-checkpoints-configs) for the full pattern.

## Pitfalls

- **`HDF5Dataset` reads from `_axes` would break.** It explicitly skips the `_axes` group and any non-`Group` children of `/measurements/`. This means it ignores domain axes — those live on the collection itself, not in the dataset. If you need them, read the source HDF5 file directly with `h5py`.
- **Items must have the same rank inside one dataset.** A directory containing both 0-D and 1-D collections will technically work but will yield mixed-rank tensors that PyTorch can't batch.
- **Complex tensors require recent PyTorch.** Raw FRF collections store `complex64`; `torch.from_numpy` returns `torch.complex64`. PyTorch ≥ 2.0 (the declared minimum) supports this.
- **The collection's HDF5 file must remain on disk.** `coll.torch_dataset()` calls `close(keep=True)` for you, but a subsequent `coll.close()` (without `keep=True`) deletes the file and breaks the dataset. Don't close the collection after handing it off.
