# API Reference

This page indexes every public symbol exported from the top-level `pymodal` namespace, grouped by module of origin. Every entry is a real export from `pymodal/__init__.py`.

For the conceptual model behind the API, see [Core Concepts](Core-Concepts).

## Concrete classes

| Name | Module | Purpose |
|---|---|---|
| `frf` | `pymodal.frf` | 1-D collection over frequency. See [FRF](FRF). |
| `timeseries` | `pymodal.timeseries` | 1-D collection over time. See [Timeseries](Timeseries). |
| `HDF5Dataset` | `pymodal.hdf5_dataset` | `torch.utils.data.Dataset` over pymodal HDF5 files. See [HDF5 Dataset](HDF5-Dataset). |

## Dimensional parents

| Name | Module | `_n_domain_axes` | Item shape |
|---|---|---|---|
| `_collection` | `pymodal.collection_parent` | abstract | — |
| `_collection_0d` | `pymodal.collection_0d` | 0 | `(n_outputs, n_inputs)` |
| `_collection_1d` | `pymodal.collection_1d` | 1 | `(D1, n_outputs, n_inputs)` |
| `_collection_2d` | `pymodal.collection_2d` | 2 | `(D1, D2, n_outputs, n_inputs)` |
| `_collection_3d` | `pymodal.collection_3d` | 3 | `(D1, D2, D3, n_outputs, n_inputs)` |

The leading underscore signals that these are abstract scaffolding; you usually instantiate one of the concrete subclasses below. They are public so users can write their own indicator collections.

## Indicator collections (by rank)

### 0-D — `pymodal.indicators_0d`

`sci_collection`, `unsigned_sci_collection`, `drq_collection`, `aigac_collection`, `frfrms_collection`, `frfsf_collection`, `frfsm_collection`, `ods_diff_collection`, `r2_imag_collection`

### 1-D — `pymodal.indicators_1d`

`rvac_collection`, `rvac_2d_collection`, `gac_collection`, `m2l_collection`

### 2-D — `pymodal.indicators_2d`

`cfdac_collection`, `cfdac_a_collection`, `fdac_collection`

Use the matching method on `frf` (`frfs.sci(ref)`, `frfs.cfdac(ref)`, …) instead of constructing these directly. See [Indicators](Indicators).

## Pure functions — `pymodal.utils`

### Domain manipulation

| Function | Signature | What it does |
|---|---|---|
| `change_domain_resolution` | `(domain_array, measurements_array, new_resolution)` | Resample along the first axis. Returns `(new_domain, new_measurements)`. |
| `change_domain_span` | `(domain_array, measurements_array, new_min_domain=None, new_max_domain=None)` | Crop or zero-pad along the first axis. Returns `(new_domain, new_measurements)`. |

### FRF comparison metrics

`(n_dof, n_freq)` complex matrices in, indicator out:

| Function | Output shape |
|---|---|
| `value_CFDAC(ref, frf)` | `(n_freq, n_freq)` complex |
| `value_CFDAC_A(ref, frf)` | `(n_freq, n_freq)` complex |
| `value_FDAC(ref, frf)` | `(n_freq, n_freq)` real |
| `value_RVAC(ref, frf)` | `(n_dof,)` real |
| `value_RVAC_2d(ref, frf)` | `(n_dof,)` real (curvature variant) |
| `value_GAC(ref, frf)` | `(n_dof,)` real |

### SHM scalar metrics

| Function | Signature |
|---|---|
| `FRFRMS(ref, frf)` | `float` |
| `FRFSF(ref, frf)` | `float` |
| `FRFSM(ref, frf, std)` | `float` (typically `std=6.0` dB) |
| `ODS_diff(ref, frf)` | `float` |
| `r2_imag(ref, frf)` | `float` |
| `SCI(CFDAC_pristine, CFDAC_altered)` | `float` (signed) |
| `unsigned_SCI(CFDAC_pristine, CFDAC_altered)` | `float` |
| `DRQ(RVAC)` | `float` (mean of RVAC) |
| `AIGAC(GAC)` | `float` (mean of GAC) |
| `M2L(CFDAC)` | `(n_dof,)` array |
| `M2L_func(x, i)` | helper used by `M2L` |

### Simulation

| Function | Signature | Description |
|---|---|---|
| `synthetic_FRF(min_freq, max_freq, resolution, natural_frequencies, damping)` | sums `1 / (1 − (ω/ω_n)² + 2 j ζ ω/ω_n)` over modes | returns a 1-D complex ndarray |
| `modal_superposition(min_freq, max_freq, resolution, modal_frequencies, damping, mode_shapes, mass_matrix, rovings, drivings)` | full multi-DOF modal superposition | returns a `pymodal.frf` |
| `damping_coefficient(omega, mass_multiplier, stiffness_multiplier)` | Rayleigh damping coefficient | returns a float / ndarray |

### Plotting

| Function | Signature |
|---|---|
| `lineplot(y, x=None, ax=None, **style)` | Heavily-formatted matplotlib line plot. |
| `plot_control_chart(di_array, di_train, threshold, colors, method_name, n_pcs)` | SHM control-chart with training/test split. |

### Array I/O

| Function | Description |
|---|---|
| `save_array(array, path)` | Save an ndarray to `.npy`, `.npz`, or `.mat`. |
| `load_array(path)` | Load an ndarray from `.npy`, `.npz`, or `.mat`. |

## Scenario builder — `pymodal.scenarios`

| Name | Type | Description |
|---|---|---|
| `Scenario` | dataclass | `(label, name, apply)` triple — one labelled damage class. |
| `ParameterVariation` | dataclass | Per-attribute relative Gaussian sigmas applied to a geometry. |
| `sample(scenario, seed, base_factory, variation=None)` | function | Single perturbed realisation of a scenario. |
| `build_frf_collection(scenarios, n_per_scenario, modal_provider, freq_array, inputs, outputs, path, base_factory, variation=None, seed=0, progress=False, method="SIMO")` | function | Assemble one labelled multi-item `pymodal.frf` collection by looping over `(scenario, sample_index)`. |
| `load_nodes_json(path)` | function | Read `{node_id: [x,y,z]}` JSON → `(ids, coords)`. |
| `closest_node(point, node_ids, coords)` | function | Resolve `(x,y,z)` to the nearest mesh node. |

The `scenarios` module is intentionally mesh-agnostic — it only knows that a *scenario* is a labelled callback that mutates an opaque "geometry" object, and that an *FRF batch* is built by repeatedly perturbing such a geometry, asking a user-supplied `modal_provider` for the corresponding FRF tensor, and persisting each result as one labelled item of an `frf` collection.

## Collection method index

The methods every collection inherits from `_collection` (parent of all four dimensional parents):

| Method | Description |
|---|---|
| `__len__()` | Number of currently-active items. |
| `__getitem__(key)` | **In-place** restriction by `int`, `slice`, `str`, list/set/tuple of strs. Returns `self`. |
| `select_all()` | Reset the active selection to every item present in the file. |
| `append(item, name=None, label=None)` | Append a single new item. |
| `split(train_frac, val_frac, test_frac, seed)` | Stratified train/val/test indices (requires labels). |
| `torch_dataset()` | Close the file (`keep=True`) and bind an `HDF5Dataset` to `self.dataset`. |
| `open()` / `close(keep=False)` | HDF5 file lifecycle. `close()` deletes by default. |
| `attach_file(path, role="model")` | Record an absolute path under `/attached/{role}`. |
| `attached_path(role="model")` | Return the path attached under `role`. |
| `attached_roles()` | List of attached roles. |
| `save_model(model, path=None, role="model")` | Save a PyTorch state dict and attach it under `role`. |
| `load_model(model_class, role="model", map_location=None, **init_kwargs)` | Reconstruct a model from the attached state dict. |
| `_embed_reference(role, ref_coll, index_map)` | Embed a referenced collection by copy. |
| `get_reference_data(role, i)` | Read the raw ndarray of item `i`'s reference under `role`. |
| `reference_roles()` | List of registered reference roles. |
| `channel_shape()` | `(n_outputs, n_inputs)`. |
| `domain_shape()` | Per-item domain-axis lengths. |

Methods specific to each dimensional parent (`change_domain_resolution`, `change_domain_span`) are documented in [Collections](Collections).

## Collection attribute index

Available on any `_collection` after construction:

| Attribute | Type | Description |
|---|---|---|
| `path` | `pathlib.Path` | HDF5 file path. |
| `file` | `h5py.File` | Open file handle (or closed after `close()`). |
| `name` | `list[str]` | Per-item names (deduped with `_<n>` suffixes). |
| `measurements` | `list[h5py.Dataset]` | Per-item HDF5 datasets, lazy-readable with `[()]`. |
| `labels` | `list[h5py.Dataset]` or `None` | Per-item label datasets. |
| `method` | `str` | `"SIMO"`, `"MISO"`, `"MIMO"`, or `"excitation"`. |
| `n_outputs`, `n_inputs`, `dof` | `int` | Channel counts. |
| `coordinates` | `(dof, 3) ndarray` | Spatial DOF positions. |
| `orientations` | `(dof, 3) ndarray` | Unit-vector DOF orientations. |
| `measurements_units`, `space_units` | `str` | Units. |
| `domain_arrays`, `domain_units` | lists | Per-axis domain arrays / unit strings. |
| `references` | `dict` | `{role: in-memory _collection}` for the linked references. |
| `reference_index_maps` | `dict` | `{role: list[int]}`. |

## Module map

```
pymodal/__init__.py                ← public API surface
pymodal/utils.py                   ← pure functions (domain, metrics, simulation, plotting, I/O)
pymodal/collection_parent.py       ← _collection (HDF5, names, labels, references, lifecycle)
pymodal/collection_0d.py           ← _collection_0d
pymodal/collection_1d.py           ← _collection_1d (parent of frf, timeseries, 1-D indicators)
pymodal/collection_2d.py           ← _collection_2d (parent of cfdac, fdac)
pymodal/collection_3d.py           ← _collection_3d (reserved)
pymodal/timeseries.py              ← timeseries(_collection_1d)
pymodal/frf.py                     ← frf(_collection_1d) and indicator methods
pymodal/indicators_0d.py           ← scalar indicator collections
pymodal/indicators_1d.py           ← per-DOF vector indicator collections
pymodal/indicators_2d.py           ← DOF×DOF matrix indicator collections
pymodal/hdf5_dataset.py            ← HDF5Dataset (PyTorch Dataset)
pymodal/scenarios.py               ← Scenario, ParameterVariation, build_frf_collection
pymodal/mcp/                       ← FastMCP server (optional [mcp] extra)
```
