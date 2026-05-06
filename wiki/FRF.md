# `frf`

`pymodal.frf` is a 1-D collection over **frequency**. Items are shaped `(n_freq, n_outputs, n_inputs)`; the single domain axis carries the frequency vector. Source: `pymodal/frf.py`.

## Construction

```python
import numpy as np
import pymodal

# Single-item collection (1 FRF)
H = np.zeros((1024, 4, 1), dtype=complex)        # 4 outputs, 1 input, SIMO
single = pymodal.frf(measurements=H, freq_resolution=1.0)

# Multi-item collection
multi = pymodal.frf(
    measurements=[H_a, H_b, H_c],
    freq_resolution=1.0,
    coordinates=np.array([[0,0,0], [10,0,0], [20,0,0], [30,0,0]]),
    orientations=np.array([[0,0,1]] * 4),
    method="SIMO",
    measurements_units="millimeter / second ** 2 / newton",
    freq_units="hertz",
    space_units="millimeter",
    labels=[0.0, 1.0, 1.0],
    names=["pristine", "dmg_a", "dmg_b"],
    path="frfs.h5",
)
```

You can define the frequency axis any of these ways:

| Pass | Effect |
|---|---|
| `freq_array=np.arange(...)` | explicit frequency vector |
| `freq_resolution=Δf` (with optional `freq_start=0.0`) | uniform grid (most common) |
| `freq_end=...` (with optional `freq_start=0.0`) | inferred from end and item length |
| `freq_span=...` | inferred from span and item length |

Construction with `method="excitation"` is rejected — that method is reserved for `timeseries`.

## Frequency aliases

| Property | Returns |
|---|---|
| `frfs.freq_array` | 1-D ndarray of frequency samples |
| `frfs.freq_units` | unit string (default `"hertz"`) |
| `frfs.freq_start`, `frfs.freq_end`, `frfs.freq_span` | floats |
| `frfs.freq_resolution` | resolution Δf |

## Resampling and cropping

```python
frfs.change_freq_resolution(2.0)                 # resample every FRF
frfs.change_freq_span(new_min_freq=10.0, new_max_freq=500.0)
```

Both mutate the HDF5 file and return `self`. Aliases for `change_domain_resolution` / `change_domain_span` from `_collection_1d`. Resolution changes that are not integer ratios of the existing resolution interpolate via `numpy.interp` per channel.

## Plotting

### `plot(format="mod", ax=None, index=0, color="blue", title=None, xlabel=None, ylabel=None, log_y=None)`

Plot one item in the requested format. Channels are flattened and overlaid.

| `format` | What is plotted | Default `log_y` |
|---|---|---|
| `"mod"` | `np.abs(H)` | `True` |
| `"phase"` | `np.angle(H)` | `False` |
| `"real"` | `H.real` | `False` |
| `"imag"` | `H.imag` | `False` |

```python
import matplotlib.pyplot as plt
frfs.plot(format="mod", index=0)
plt.show()
```

### `waterfall(ax=None, format="mod", dof_idx=0, colormap=None, alpha=0.7)`

3-D waterfall over every item along the y-axis (one DOF column shown per slice). Useful for visualising how a damage class evolves across realisations.

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
frfs.waterfall(ax=ax, format="mod", dof_idx=0)
plt.show()
```

## SHM indicator methods

Every indicator method on `frf` takes a `reference` (another `frf`) and returns a typed indicator collection with both `"reference"` and `"damaged"` embedded automatically. You never instantiate the indicator collection class directly — use these methods.

### 0-D (one scalar per item)

| Method | Resulting class | Description |
|---|---|---|
| `frfs.sci(reference)` | `sci_collection` | Signed Structural Change Indicator |
| `frfs.unsigned_sci(reference)` | `unsigned_sci_collection` | Unsigned variant of SCI |
| `frfs.drq(reference)` | `drq_collection` | Damage Residual Quantifier (mean of RVAC) |
| `frfs.aigac(reference)` | `aigac_collection` | Average Index from GAC |
| `frfs.frfrms(reference)` | `frfrms_collection` | FRF RMS deviation (log scale) |
| `frfs.frfsf(reference)` | `frfsf_collection` | FRF Scale Factor |
| `frfs.frfsm(reference, std=6.0)` | `frfsm_collection` | FRF Similarity Metric (Gaussian, dB) |
| `frfs.ods_diff(reference)` | `ods_diff_collection` | Operating Deflection Shape difference |
| `frfs.r2_imag(reference)` | `r2_imag_collection` | R² of the imaginary part |

### 1-D (one vector per item, indexed over DOF)

| Method | Resulting class | Description |
|---|---|---|
| `frfs.rvac(reference)` | `rvac_collection` | Response Vector Assurance Criterion |
| `frfs.rvac_2d(reference)` | `rvac_2d_collection` | Curvature variant (second-difference) |
| `frfs.gac(reference)` | `gac_collection` | Global Amplitude Criterion |
| `frfs.m2l(reference)` | `m2l_collection` | Mode-shape-to-Local indicator |

### 2-D (one matrix per item, indexed over DOF × DOF)

| Method | Resulting class | Description |
|---|---|---|
| `frfs.cfdac(reference)` | `cfdac_collection` | Complex Frequency Domain Assurance Criterion |
| `frfs.cfdac_a(reference)` | `cfdac_a_collection` | CFDAC alternative formulation |
| `frfs.fdac(reference)` | `fdac_collection` | Frequency Domain Assurance Criterion (real) |

Every one of these accepts a `path=` argument that controls where the resulting indicator HDF5 file is written. If omitted, a unique file path is auto-generated.

See [Indicators](Indicators) for the formulas and reference papers.

### Reference index mapping

When `len(reference) == len(self)`, items are paired by index (item `i` of `self` is compared against item `i` of `reference`). When the lengths differ — for example a single pristine baseline vs. many damaged realisations — every damaged item is paired against item `0` of the reference. Custom mappings are possible by constructing the indicator collection directly via its `from_pair`/`from_pair_op` classmethod and passing `reference_index_maps={...}`.

## Internal helper: `_as_matrix(i)`

```python
H = frfs._as_matrix(0)            # (n_dof, n_freq) complex matrix
```

Reshapes item `i` into the `(n_dof, n_freq)` convention used by the SHM-indicator pure functions in `pymodal.utils`. You normally don't need to call this directly; it is the bridge that lets the indicator methods delegate to `utils.value_CFDAC`, `utils.value_RVAC`, etc.

## Inheriting `_collection` behaviour

`frf` inherits the full `_collection` API: `__getitem__` (in-place), `select_all`, `append`, `split`, `open`/`close`, `attach_file`/`save_model`/`load_model`, `references` / `get_reference_data`, and `torch_dataset`. See [Collections](Collections) for details.

## Common patterns

### Pristine vs. damaged in one expression

```python
frfs = pymodal.frf(measurements=all_arrays, freq_resolution=1.0,
                    labels=[0]*1 + [1]*N_dmg, path="frfs.h5")
sci = frfs[1:].sci(reference=frfs[:1])
```

### Computing several indicators on the same pair

```python
ref, dmg = pymodal.frf(...), pymodal.frf(...)
cfdac = dmg.cfdac(ref, path="cfdac.h5")
rvac  = dmg.rvac(ref,  path="rvac.h5")
sci   = dmg.sci(ref,   path="sci.h5")
```

Each call writes a fresh HDF5 file with `ref` and `dmg` embedded as references — the four files are independent and self-contained.

### Frequency-band restriction before indicators

```python
dmg.change_freq_span(new_min_freq=20.0, new_max_freq=400.0)
ref.change_freq_span(new_min_freq=20.0, new_max_freq=400.0)
sci = dmg.sci(ref)
```

Both collections must share the same frequency axis for the math to make sense. The `value_CFDAC`/`value_RVAC` functions in `utils` operate on the `(n_dof, n_freq)` matrices directly and expect aligned frequency grids.

## Pitfalls

- **Construction with `method="excitation"` raises.** Use `timeseries` for force-input.
- **MIMO requires `n_outputs == n_inputs`.** SIMO/MISO are unrestricted.
- **Indicator math expects matching frequency axes.** If the reference and damaged collections have different `freq_array`s, crop or resample first.
- **Items with `n_dof == 1` make most 1-D / 2-D indicators degenerate.** Vector indicators reduce to a length-1 vector and FDAC/CFDAC to a 1×1 matrix; that is mathematically expected but rarely useful.
- **Complex dtype is preserved.** `frf` items can be `complex64` or `complex128`; the parent constructor downcasts `complex128` → `complex64` on disk for storage parity with legacy datasets.
