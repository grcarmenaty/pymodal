# Quickstart

A 5-minute, end-to-end example: synthesise a small dataset of damaged-vs-pristine FRFs, compute an indicator collection, and hand the result off to a PyTorch `DataLoader`.

Every line is grounded in the public API. You can copy this script verbatim into a Python file or notebook.

## Setup

```python
import numpy as np
import pymodal
```

## 1. Synthesise some FRFs

`pymodal.synthetic_FRF` sums modal contributions of the form
`1 / (1 - (ω/ω_n)² + 2 j ζ ω/ω_n)` over the supplied modes:

```python
freq_resolution = 1.0
min_freq, max_freq = 0.0, 1024.0
n_freq = int((max_freq - min_freq) / freq_resolution) + 1

def make_frf(natural_freqs, damping):
    arr = pymodal.synthetic_FRF(
        min_freq=min_freq,
        max_freq=max_freq,
        resolution=freq_resolution,
        natural_frequencies=np.asarray(natural_freqs, dtype=float),
        damping=np.asarray(damping, dtype=float),
    )
    return arr.reshape(n_freq, 1, 1)        # (n_freq, n_outputs, n_inputs)
```

Build a small "pristine" baseline and three "damaged" realisations whose first mode drifts down by 1, 2, and 3 % respectively:

```python
pristine = make_frf([100.0, 250.0, 480.0], [0.01, 0.01, 0.01])
damaged = [
    make_frf([100.0 * (1 - p), 250.0, 480.0], [0.01, 0.01, 0.01])
    for p in (0.01, 0.02, 0.03)
]
```

## 2. Build labelled FRF collections

A single `ndarray` makes a 1-item collection; a list makes a multi-item collection. Pristine = label `0`, damaged = label `1`:

```python
ref = pymodal.frf(
    measurements=pristine,
    freq_resolution=freq_resolution,
    labels=[0.0],
    names=["pristine"],
    path="ref.h5",
)

dmg = pymodal.frf(
    measurements=damaged,
    freq_resolution=freq_resolution,
    labels=[1.0, 1.0, 1.0],
    names=["dmg_1pct", "dmg_2pct", "dmg_3pct"],
    path="dmg.h5",
)
```

## 3. Compute an SHM indicator

Every indicator method on `frf` returns a typed indicator collection with both inputs embedded as references. CFDAC produces a 2-D matrix per item:

```python
cfdac = dmg.cfdac(reference=ref, path="cfdac.h5")
print(type(cfdac).__name__)          # cfdac_collection
print(cfdac.measurements[0].shape)   # (n_freq, n_freq, 1, 1)
print(cfdac.reference_roles())       # ['reference', 'damaged']
```

A 0-D scalar indicator works the same way:

```python
sci = dmg.sci(reference=ref, path="sci.h5")
print(sci.measurements[0].shape)     # (1, 1)
print([m[()].item() for m in sci.measurements])
# [<sci value for 1% damage>, <2% damage>, <3% damage>]
```

## 4. Augment time-series (optional aside)

If you started in the time domain, you can append Gaussian-noise-augmented copies before computing FRFs:

```python
sr = 8192
t = np.arange(0, 1.0, 1 / sr)
acc = np.sin(2 * np.pi * 100 * t).reshape(-1, 1, 1)
force = np.random.randn(*acc.shape)

resp = pymodal.timeseries([acc], time_step=1 / sr, labels=[0.0])
exc  = pymodal.timeseries([force], time_step=1 / sr, method="excitation")

resp.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01)
print([n for n in resp.name])
# ['item_0', 'item_0_augmented']

frfs = resp.to_FRF(exc.append(force, name="force_aug"), FRF_type="H1")
print(frfs.reference_roles())
# ['input', 'output']
```

(The `to_FRF` step needs the excitation collection to have the same number of items as the response collection, which is why `force` is appended above.)

## 5. Hand off to PyTorch

`.torch_dataset()` closes the HDF5 file (keeping it on disk) and binds an `HDF5Dataset` to `.dataset`:

```python
import torch
from torch.utils.data import DataLoader

dataset = sci.torch_dataset().dataset
print(len(dataset))                  # 3
x, y = dataset[0]
print(x.shape, x.dtype)              # torch.Size([1, 1]) torch.float64
print(y.shape, y.dtype)              # torch.Size([]) torch.float64

loader = DataLoader(dataset, batch_size=2, num_workers=0)
for x, y in loader:
    print(x.shape, y.shape)
    break
```

For variable-length items (e.g. after asymmetric `change_freq_span` calls), use the bundled `pad_collate_fn`:

```python
loader = DataLoader(
    dataset, batch_size=2, collate_fn=pymodal.HDF5Dataset.pad_collate_fn
)
```

## 6. Stratified split

If your collection is labelled, you can request a deterministic train/val/test split:

```python
train, val, test = dmg.split(train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42)
print(train, val, test)              # lists of item indices
```

The indices are also stored on the collection as `train_indices`, `val_indices`, `test_indices` for later use.

## 7. Cleanup

If you don't want the on-disk HDF5 files to persist:

```python
ref.close()        # also unlinks ref.h5
dmg.close()
cfdac.close()
sci.close()
```

Pass `keep=True` (e.g. `coll.close(keep=True)`) to keep the files. `torch_dataset()` already calls `close(keep=True)` for you.

## What's next

- See [Collections](Collections) for the full lifecycle (`append`, indexing, `references`, file handles).
- See [Indicators](Indicators) for every available indicator and the formulas they implement.
- See [HDF5 Dataset](HDF5-Dataset) for `DataLoader` integration patterns.
- See the [MCP Server](MCP-Server) page if you want to drive this pipeline from an LLM agent.
