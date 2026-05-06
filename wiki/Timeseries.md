# `timeseries`

`pymodal.timeseries` is a 1-D collection over **time**. Items are shaped `(n_samples, n_outputs, n_inputs)`; the single domain axis carries the time vector. Source: `pymodal/timeseries.py`.

## Construction

```python
import numpy as np
import pymodal

# Single-item collection
acc = np.random.randn(8192, 1, 1)
ts = pymodal.timeseries(measurements=acc, time_step=1/8192)

# Multi-item collection (identical shapes required)
ts = pymodal.timeseries(
    measurements=[acc1, acc2, acc3],
    time_step=1/8192,
    labels=[0.0, 1.0, 1.0],
    names=["pristine", "dmg_a", "dmg_b"],
    coordinates=np.array([[0, 0, 0]]),
    orientations=np.array([[0, 0, 1]]),
    method="SIMO",
    measurements_units="meter / second ** 2",
    time_units="second",
    space_units="millimeter",
    path="response.h5",
)
```

You can define the time axis any of these ways (pick whichever you have):

| Pass | Effect |
|---|---|
| `time_array=np.linspace(0, 1, n)` | explicit time vector |
| `time_step=1/sr` | sampling period in seconds (most common) |
| `time_end=...` (with optional `time_start=0`) | inferred from end and item length |
| `time_span=...` | inferred from span and item length |

`time_start` defaults to `0.0`. If both `time_step` and `time_end` are omitted, construction raises `ValueError`.

### `method="excitation"`

Mark a force-input collection by passing `method="excitation"`. This also flips the default `measurements_units` from `"meter / second ** 2"` to `"newton"`. Excitation collections are the input side of `to_FRF`; they cannot be passed to `frf` (which rejects `method="excitation"`).

## Time aliases

Convenience properties wrap the underlying domain attributes:

| Property | Returns |
|---|---|
| `ts.time_array` | 1-D ndarray of time samples |
| `ts.time_units` | unit string (default `"second"`) |
| `ts.time_start`, `ts.time_end`, `ts.time_span` | floats |
| `ts.time_step` | sampling period |
| `ts.sampling_rate` | `1 / time_step` (Hz) |

## Resampling and cropping

```python
ts.change_sampling_rate(4096)                    # resample every signal
ts.change_time_span(new_min_time=0.1, new_max_time=2.0)   # crop or pad with zeros
```

Both mutate the HDF5 file and return `self` for chaining. `change_sampling_rate` is a thin wrapper around `change_domain_resolution(new_resolution=1/sr)`; `change_time_span` wraps `change_domain_span`. Resampling that is not an integer ratio of the existing rate triggers `numpy.interp` per channel. Cropping to a span larger than the existing one zero-pads at the appropriate end.

## Plotting

`ts.plot(ax=None, index=0, color="blue", title=None, xlabel=None, ylabel=None)` plots one item in the time domain. Channels are flattened — every output × input column is overlaid. Returns the matplotlib `Axes`.

```python
import matplotlib.pyplot as plt
ts.plot(index=0)
plt.show()
```

## Augmentation: `AddGaussianNoise`

```python
ts.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, sample=1.0)
```

For each selected item, a noisy copy is **appended** to the collection with the suffix `_augmented`; the originals are preserved. Backed by `audiomentations.AddGaussianNoise` per channel.

| Argument | Meaning |
|---|---|
| `min_amplitude` / `max_amplitude` | noise amplitude range (passed straight to audiomentations) |
| `sample` | which items to augment: `float` (fraction, default 1.0), or list of names |

After this call, `ts.name` includes `<original>_augmented` entries; labels carry over from the original.

## Computing FRFs: `to_FRF`

`to_FRF(excitation, FRF_type="H1", resp_delay=0, path=None)` pairs the response collection with an excitation collection and returns a `pymodal.frf` whose two reference roles are pre-wired:

- `"input"` → `excitation`
- `"output"` → `self`

```python
resp = pymodal.timeseries([acc1, acc2], time_step=1/8192, labels=[0.0, 1.0])
exc  = pymodal.timeseries([force1, force2], time_step=1/8192, method="excitation")

frfs = resp.to_FRF(exc, FRF_type="H1")
print(frfs.reference_roles())    # ['input', 'output']
print(frfs.freq_resolution)      # 1/time_span
```

Accepted `FRF_type` values (passed through to pyFRF): `"H1"`, `"H2"`, `"Hv"`, `"vector"`, `"ODS"`.

### Requirements and unit handling

- `excitation.method` must be `"excitation"`.
- `len(excitation) == len(self)` — items are paired by index.
- `self.method` must be `"SIMO"`, `"MISO"`, or `"MIMO"`. Each is wired to pyFRF differently:
  - **SIMO**: one excitation channel, `n_outputs` response channels.
  - **MISO**: `n_inputs` excitation channels, one response channel.
  - **MIMO**: square `(n_outputs, n_inputs)`; one pyFRF per output × input pair.
- Response and excitation `measurements_units` are parsed by Pint and mapped to pyFRF's `resp_type`/`exc_type` codes. Acceleration → `"a"`, velocity → `"v"`, displacement → `"d"`, force → `"f"`, otherwise `"e"` (electrical / unknown). The resulting `frf.measurements_units` is `"(<resp_units>) / (<exc_units>)"`.

The frequency axis of the returned FRF starts at `0 Hz` with resolution `1 / time_span`.

## File and selection lifecycle

`timeseries` inherits the full `_collection` API: `__getitem__` (in-place), `select_all`, `append`, `split`, `open`/`close`, `attach_file`/`save_model`/`load_model`, `references` / `get_reference_data`, and `torch_dataset`. See [Collections](Collections) for details.

## Common patterns

### Build a labelled response collection from a list of measurements

```python
ts = pymodal.timeseries(
    measurements=[acc[i] for i in range(N)],
    time_step=1/sr,
    labels=labels,
    names=[f"trial_{i:03d}" for i in range(N)],
    path="response.h5",
)
```

### Crop, augment, and convert in one chain

```python
frfs = (
    pymodal.timeseries(measurements=signals, time_step=1/sr, labels=labels)
        .change_time_span(new_min_time=0.05, new_max_time=2.0)
        .AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.01)
        .to_FRF(excitation_collection, FRF_type="H1")
)
```

### Hand off straight to PyTorch (without going through FRFs)

```python
ds = pymodal.timeseries(...).torch_dataset().dataset
x, y = ds[0]                    # x.shape == (n_samples, n_outputs, n_inputs)
```

## Pitfalls

- **`method="excitation"` only on inputs.** Don't tag a response collection as excitation; `to_FRF` will refuse to compute against it.
- **Sampling rates must agree** between response and excitation. Resample one of them first (`change_sampling_rate`) if they differ.
- **`AddGaussianNoise` requires `audiomentations`.** Listed as a required dependency in `setup.py`.
- **Augmented copies are appended, not replaced.** If you don't want the originals, build the collection from scratch with the augmented arrays.
