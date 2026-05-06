# FAQ

Common questions and pitfalls. If something tripped you up, it probably belongs here.

## General

### Why HDF5 instead of `numpy.savez` or pickle?

Three reasons:

1. **Streaming.** HDF5 lets pymodal read and write items one at a time without ever materialising the full dataset in RAM. This is the entire reason the library exists — the design target is collections too large to fit in memory.
2. **Self-describing.** Every collection's HDF5 file carries channel metadata, domain axes, units, names, labels, and embedded reference snapshots in a fixed, predictable layout. You can poke at it with `h5dump` or `h5py` directly without going through pymodal.
3. **PyTorch-friendly.** `HDF5Dataset` opens a fresh read-only handle inside each `__getitem__`, which makes it safe with `DataLoader(num_workers > 0)`. `pickle`-backed datasets are not.

### What's the difference between `frf` and `frf_collection`?

There is no `frf_collection`. The class was removed in the rewrite; `frf` now plays both roles. A single `ndarray` constructs a 1-item collection; a list of `ndarray`s constructs a multi-item collection. Same for `timeseries`. See [Core Concepts § 2](Core-Concepts#2-frf-and-timeseries-are-collections).

### Why do indicator collections store `(1, 1)` channel shape?

Because the channel/DOF metadata is recovered from the embedded `reference` and `damaged` collections rather than duplicated on each indicator value. An `sci_collection` is a per-item scalar — the original `n_outputs × n_inputs` layout doesn't apply to it. If you need the channel coordinates, read them off `coll.references["reference"]`.

## File handling

### When does `close()` delete the file?

By default. `coll.close()` closes the HDF5 handle **and** unlinks the file. Pass `coll.close(keep=True)` to retain the file on disk. `coll.torch_dataset()` already calls `close(keep=True)` internally so that the `HDF5Dataset` can keep reading from it.

This default exists because most pymodal collections in scripts and notebooks are short-lived working state. Persistent datasets are typically built once with an explicit `path=` and kept around with `keep=True`.

### What if I want to reopen a collection later?

Construct it fresh from the same arrays — the constructor is the only public path that builds the HDF5 file. Or use the loader functions in `pymodal.mcp.loaders` (`load_frf`, `load_timeseries`) which read the file and reconstruct the in-memory object. Note that those loaders are intended primarily for the MCP server; they are part of the package but not exported from the top-level `pymodal` namespace.

### I'm getting `OSError: file already open` / locking errors on a network filesystem.

`pymodal/collection_parent.py` sets `os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")` at import time precisely to avoid this. If it still happens, set the env var explicitly **before** Python starts:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
python your_script.py
```

### Is the HDF5 file thread-safe?

No — `h5py` itself is not thread-safe. But `HDF5Dataset` is **multi-process-safe**: every `__getitem__` opens a fresh read-only handle, which is what `DataLoader(num_workers > 0)` needs. If you need threaded access during training, use `num_workers > 0` (separate processes) instead of threads.

## Construction

### What goes in `coordinates` and `orientations`?

`coordinates` is a `(dof, 3)` array of XYZ positions for each DOF; `orientations` is a `(dof, 3)` array of unit vectors. Defaults are `(0, 0, 0)`-line for coordinates and `+z` for orientations. The library auto-normalises orientations and tolerates non-unit input.

### How do I pick `n_outputs`, `n_inputs`, and `dof`?

The constructor infers `n_outputs` and `n_inputs` from the last two axes of your item arrays. `dof` defaults to `max(n_outputs, n_inputs)` and is purely a metadata attribute — it controls the length of `coordinates` / `orientations` and the indicator output sizes (e.g. RVAC returns one value per DOF), but does not change how items are stored.

### My items have different shapes — how do I build one collection?

You can't, at construction. `_collection.__init__` enforces uniform shape across items. Either pad your items to a common shape before constructing, build separate collections (one per shape), or use `change_domain_span` / `change_domain_resolution` after construction to bring everything to a common axis. For PyTorch loading of variable-length items along the last axis, use `pymodal.HDF5Dataset.pad_collate_fn` — but the *items as stored* still need a uniform shape.

### Can I construct a `frf` from a `(n_freq, n_dof)` array?

The constructor will append a trailing axis to bring the rank up to `n_domain_axes + 2 = 3`, so a 2-D `(n_freq, n_dof)` input becomes a `(n_freq, n_dof, 1)` SIMO 1-input collection. If you intended `(n_freq, n_outputs, n_inputs)` and the constructor's auto-padding does the wrong thing, reshape explicitly before passing in.

### I passed `method="excitation"` to `frf` and got an error.

Correct — `method="excitation"` is reserved for `timeseries`. Force-input collections live in the time domain; FRFs are always response/excitation ratios. If you're trying to build an FRF from a time-domain pair, use `timeseries.to_FRF(excitation)`.

## Pre-processing

### `change_freq_resolution` printed a warning.

The underlying `pymodal.change_domain_resolution` warns when the requested resolution is not an integer multiple of the existing one — in that case it interpolates with `numpy.interp` instead of decimating. The warning is captured and silenced by `change_freq_resolution` / `change_domain_resolution`, so you usually don't see it; if you do, it's coming from manual calls to `pymodal.change_domain_resolution`.

### `change_freq_span` is supposed to crop, why is the file getting bigger?

If `new_max_freq > current freq_end`, the function **extends** the axis with zeros at the high-frequency end (and likewise at the low end if `new_min_freq < freq_start`). To strictly crop, pass values inside the existing range.

### `AddGaussianNoise` doubled my collection size.

By design — augmented copies are *appended*, not replacing the originals. Each augmented copy gets the suffix `_augmented`. If you want only the augmented data, build a fresh collection from the augmented arrays.

## FRFs from time-series

### What `FRF_type` should I use?

The estimators are passed straight through to pyFRF. Usual choices:

- `"H1"` — least-squares, robust to output noise. **Default.**
- `"H2"` — robust to input noise.
- `"Hv"` — geometric mean of H1 and H2.
- `"vector"` — raw FFT division (no averaging).
- `"ODS"` — operating deflection shape.

When in doubt, start with `"H1"`.

### `to_FRF` raised `ValueError("excitation.method must be 'excitation'")`.

Build the excitation collection with `method="excitation"`. That marks it as the input side and switches the default `measurements_units` to `"newton"`.

### The frequency resolution of my computed FRF surprised me.

`to_FRF` sets `freq_resolution = 1 / time_span`. If you cropped the time axis before computing FRFs, the new span determines the new frequency grid.

## Indicators

### Which indicator should I use?

See the picking-an-indicator section in [Indicators](Indicators#picking-an-indicator). Short version:

- One scalar per realisation → 0-D (`sci`, `drq`, `frfrms`, …).
- Localise along DOFs → 1-D (`rvac`, `gac`, `m2l`).
- Full off-diagonal coupling → 2-D (`cfdac`, `fdac`).

### Reference and damaged have different numbers of items — what happens?

`from_pair` falls back to comparing every damaged item against item `0` of the reference. If you want a custom mapping, build the indicator collection directly via its `from_pair`/`from_pair_op` classmethod and pass `reference_index_maps={"reference": [...]}` explicitly.

### My CFDAC matrices are mostly zero.

Either the reference and damaged FRFs are nearly identical (rare in real datasets), or the input matrices' frequency grids don't align (so the matrix products at any given pair of frequencies see uncorrelated noise). Check `freq_array` matches between the two collections.

## PyTorch hand-off

### Why does `torch_dataset()` close the file?

So that the `HDF5Dataset` can open its own read-only handles per worker. Once you call `torch_dataset()`, treat the collection as "frozen on disk" — you can still call `coll.open()` to reacquire write access, but the `HDF5Dataset` returned by the call is the supported read path.

### `DataLoader` is dropping samples / hangs with `num_workers > 0`.

Most likely you forked the process with the HDF5 file already open. The `HDF5Dataset` opens its own handle inside `__getitem__`, so as long as the file is closed when the workers are spawned this is fine. The pattern that works:

```python
ds = coll.torch_dataset().dataset
# ↑ closes the original handle internally before constructing the dataset
loader = DataLoader(ds, batch_size=32, num_workers=4)
```

If you keep the original handle open in the parent process and then start workers, you get undefined behaviour.

### Can `HDF5Dataset` load complex tensors?

Yes — PyTorch ≥ 2.0 supports `complex64` and `complex128` tensors natively, and pymodal stores raw `frf` items as `complex64`. If your model expects real input, take `.real`, `.imag`, or `.abs()` in the model or via the `transform=` callable.

## Pinning and reproducibility

### How do I make a build reproducible?

Pass `seed=` to `coll.split(...)`. Pass `seed=` and a `np.random.default_rng(seed)` into your `ParameterVariation` callbacks if you're using `pymodal.scenarios`. Set numpy and PyTorch seeds globally if your model has stochastic layers.

The HDF5 file does **not** record the random seed automatically. If you need provenance, save it as a string under `/attached/` via `coll.attach_file` or attach a small JSON config alongside.

### How do I version my collections?

There is no built-in versioning. The collection's HDF5 layout is stable, but pymodal pre-1.0 occasionally adds attributes. For reproducibility, pin a specific `pymodal` version in your environment and store the version string alongside your collection (e.g. as an attached config file).

## Misc

### Do I need ANSYS to use pymodal?

No. The README mentions ANSYS for legacy reasons; the current codebase has no hard ANSYS dependency. FEA-driven dataset construction is handled externally via the mesh-agnostic `pymodal.scenarios.build_frf_collection` builder, which calls a user-supplied `modal_provider` callable to produce each FRF tensor. That callable can run ANSYS, OpenSees, an internal solver, or none of the above.

### Why is `Optional[X]` used everywhere instead of `X | None`?

Compatibility with Python 3.9. The PEP 604 `X | None` syntax requires Python 3.10+. pymodal targets Python ≥ 3.9.

### Where do I report bugs or request features?

The repository is at <https://github.com/grcarmenaty/pymodal>. Open an issue with a minimal reproducer; PRs are welcome.
