"""Base class for HDF5-backed, reference-aware, dimensionality-typed collections.

Every concrete collection in pymodal inherits from one of the four
dimensional parents:

- :class:`_collection_0d` — items shaped ``(n_outputs, n_inputs)``
- :class:`_collection_1d` — items shaped ``(D1, n_outputs, n_inputs)``
- :class:`_collection_2d` — items shaped ``(D1, D2, n_outputs, n_inputs)``
- :class:`_collection_3d` — items shaped ``(D1, D2, D3, n_outputs, n_inputs)``

All of those inherit the storage, naming, labelling, reference-linking, and
PyTorch-pipeline behaviour defined in this module's :class:`_collection`.

HDF5 layout
-----------
::

    /measurements/
        .attrs:
            method, n_domain_axes, n_outputs, n_inputs, dof,
            coordinates, orientations,
            measurements_units, space_units
        _axes/
            axis_{k}/data         (one per domain axis)
            axis_{k}.attrs.units
        {name}/
            data                  (rank = n_domain_axes + 2)
            label                 (optional scalar)
    /references/
        {role}/
            .attrs:
                mode             = "embedded"
                index_map        = int[N]
                collection_class = "frf" | "timeseries" | "cfdac" | ...
                n_domain_axes    = int
            measurements/
                ... full embedded snapshot of the linked collection
"""

from __future__ import annotations

import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional
from warnings import catch_warnings, filterwarnings

import h5py
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# ── Helpers ───────────────────────────────────────────────────────────────────

def add_suffix(strings: list[str]) -> list[str]:
    """Return a copy of ``strings`` with duplicates disambiguated by ``_<n>`` suffixes."""
    counter: dict[str, int] = {}
    result: list[str] = []
    for s in strings:
        if s in counter:
            new = s
            while new in counter:
                count = counter[s]
                counter[s] = count + 1
                new = f"{s}_{count}"
            counter[new] = 1
            result.append(new)
        else:
            counter[s] = 1
            result.append(s)
    return result


def _as_str(value) -> str:
    """Coerce numpy bytes/str scalars from HDF5 attrs to plain str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


# ── Base class ────────────────────────────────────────────────────────────────

class _collection:
    """Abstract HDF5-backed collection with reference linking and labels.

    Subclasses fix the number of domain axes via the class attribute
    ``_n_domain_axes`` (0, 1, 2, or 3). All items share a single channel
    structure of shape ``(n_outputs, n_inputs)`` appended to the domain axes.

    Parameters
    ----------
    items : list of numpy.ndarray
        Per-item measurement arrays, each of rank ``n_domain_axes + 2``.
    domain_arrays : list of numpy.ndarray, optional
        One 1-D array per domain axis. Required for ``n_domain_axes > 0``.
    domain_units : list of str, optional
        One unit string per domain axis. Required for ``n_domain_axes > 0``.
    coordinates : numpy.ndarray, optional
        Spatial coordinates for the channel DOFs; shape ``(dof, 3)``.
    orientations : numpy.ndarray, optional
        Unit-vector orientations for the channel DOFs; shape ``(dof, 3)``.
    dof : int, optional
        Number of distinct DOFs; inferred from item shape if omitted.
    method : str, optional
        ``"SIMO"`` (default), ``"MISO"``, ``"MIMO"``, or ``"excitation"``.
    space_units : str, optional
        Coordinate units, default ``"millimeter"``.
    measurements_units : str, optional
        Unit string for the measurement values; subclass-specific default.
    names : list of str, optional
        Per-item identifying name. Auto-generated if omitted.
    labels : list of float, optional
        Numeric label per item (e.g. damage state).
    references : dict, optional
        ``{role_name: _collection}`` mapping. Each linked collection is embedded
        into the new file under ``/references/{role}/``.
    reference_index_maps : dict, optional
        ``{role_name: list[int]}`` mapping each item index to a position in the
        referenced collection. Defaults to ``[0] * len(items)`` per role.
    path : Path or str, optional
        Filesystem path for the HDF5 file. Auto-generated if omitted.
    """

    _n_domain_axes: int = 0          # overridden by 0d/1d/2d/3d subclasses
    _default_units: str = ""         # overridden by domain-specific subclasses

    def __init__(
        self,
        items: list[np.ndarray],
        domain_arrays: Optional[list[np.ndarray]] = None,
        domain_units: Optional[list[str]] = None,
        coordinates: Optional[np.ndarray] = None,
        orientations: Optional[np.ndarray] = None,
        dof: Optional[int] = None,
        method: str = "SIMO",
        space_units: str = "millimeter",
        measurements_units: Optional[str] = None,
        names: Optional[list[str]] = None,
        labels: Optional[list[float]] = None,
        references: Optional[dict] = None,
        reference_index_maps: Optional[dict] = None,
        path: Optional[Path] = None,
    ):
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError("`items` must be a non-empty list of arrays.")

        n_axes = self._n_domain_axes
        expected_rank = n_axes + 2

        # ── normalise items to expected rank ─────────────────────────────────
        normed_items: list[np.ndarray] = []
        for i, arr in enumerate(items):
            a = np.asarray(arr)
            while a.ndim < expected_rank:
                a = a[..., np.newaxis]
            if a.ndim != expected_rank:
                raise ValueError(
                    f"Item {i} has rank {a.ndim}, expected {expected_rank} "
                    f"({n_axes} domain axes + 2 channel axes)."
                )
            normed_items.append(a)

        # ── shape consistency across items ───────────────────────────────────
        ref_shape = normed_items[0].shape
        for i, a in enumerate(normed_items[1:], start=1):
            if a.shape != ref_shape:
                raise ValueError(
                    f"Item {i} shape {a.shape} differs from item 0 shape {ref_shape}."
                )

        n_outputs = ref_shape[-2]
        n_inputs = ref_shape[-1]
        domain_shape = ref_shape[:-2]

        # ── method/channel sanity ────────────────────────────────────────────
        if method not in ("SIMO", "MISO", "MIMO", "excitation"):
            raise ValueError(f"Unknown method '{method}'.")
        if method == "MIMO" and n_outputs != n_inputs:
            raise ValueError("MIMO requires n_outputs == n_inputs.")

        if dof is None:
            dof = max(n_outputs, n_inputs)

        # ── coordinates / orientations defaults ─────────────────────────────
        if coordinates is None:
            coordinates = np.column_stack(
                (np.arange(dof, dtype=float), np.zeros(dof), np.zeros(dof))
            )
        else:
            coordinates = np.asarray(coordinates, dtype=float)

        if orientations is None:
            orientations = np.column_stack(
                (np.zeros(dof), np.zeros(dof), np.ones(dof))
            )
        else:
            orientations = np.asarray(orientations, dtype=float)
        norms = np.linalg.norm(orientations, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        orientations = orientations / norms

        # ── domain axes ──────────────────────────────────────────────────────
        if n_axes > 0:
            if domain_arrays is None or len(domain_arrays) != n_axes:
                raise ValueError(
                    f"Subclass requires {n_axes} domain_arrays."
                )
            domain_arrays = [np.asarray(d, dtype=float) for d in domain_arrays]
            for k, (arr, expected_len) in enumerate(zip(domain_arrays, domain_shape)):
                if arr.ndim != 1 or arr.shape[0] != expected_len:
                    raise ValueError(
                        f"domain_arrays[{k}] must be 1-D of length {expected_len}; "
                        f"got shape {arr.shape}."
                    )
            if domain_units is None:
                domain_units = [self._default_units] * n_axes
            elif len(domain_units) != n_axes:
                raise ValueError(f"domain_units must have length {n_axes}.")
        else:
            domain_arrays = []
            domain_units = []

        # ── names ────────────────────────────────────────────────────────────
        if names is None:
            names = [f"item_{i}" for i in range(len(normed_items))]
        if len(names) != len(normed_items):
            raise ValueError("len(names) must equal len(items).")
        names = add_suffix(list(names))

        # ── labels ───────────────────────────────────────────────────────────
        if labels is not None and len(labels) != len(normed_items):
            raise ValueError("len(labels) must equal len(items).")

        # ── path ─────────────────────────────────────────────────────────────
        if path is None:
            path = Path(f"{int(time.time() * 1000)}_{id(self):x}.h5")
        else:
            path = Path(path)
        if path.exists():
            path.unlink()

        # ── persist all collection-level attributes on self ──────────────────
        self.path = path
        self.name = list(names)
        self.method = method
        self.dof = int(dof)
        self.n_outputs = int(n_outputs)
        self.n_inputs = int(n_inputs)
        self.coordinates = coordinates
        self.orientations = orientations
        self.space_units = str(space_units)
        self.measurements_units = (
            str(measurements_units)
            if measurements_units is not None
            else self._default_measurements_units(method)
        )
        self.domain_units = [str(u) for u in domain_units]
        self.domain_arrays = domain_arrays

        # ── write file ───────────────────────────────────────────────────────
        with h5py.File(self.path, "w") as f:
            grp = f.create_group("measurements")
            grp.attrs["method"] = self.method
            grp.attrs["n_domain_axes"] = n_axes
            grp.attrs["n_outputs"] = self.n_outputs
            grp.attrs["n_inputs"] = self.n_inputs
            grp.attrs["dof"] = self.dof
            grp.attrs["coordinates"] = self.coordinates
            grp.attrs["orientations"] = self.orientations
            grp.attrs["measurements_units"] = self.measurements_units
            grp.attrs["space_units"] = self.space_units

            axes_grp = grp.create_group("_axes")
            for k, (arr, units) in enumerate(zip(self.domain_arrays, self.domain_units)):
                ds = axes_grp.create_dataset(f"axis_{k}", data=arr)
                ds.attrs["units"] = units

            for i, (n, arr) in enumerate(zip(self.name, normed_items)):
                item_grp = grp.create_group(n)
                # downcast complex128 → complex64 for storage parity with legacy
                if np.iscomplexobj(arr) and arr.dtype == np.complex128:
                    arr = arr.astype(np.complex64)
                item_grp.create_dataset("data", data=arr)
                if labels is not None:
                    item_grp.create_dataset("label", data=labels[i])

        # ── open and bind handles ────────────────────────────────────────────
        self.file = h5py.File(self.path, "a")
        self.measurements = [
            self.file[f"measurements/{n}/data"] for n in self.name
        ]
        self.labels = (
            [self.file[f"measurements/{n}/label"] for n in self.name]
            if labels is not None
            else None
        )

        # ── references (embed-by-copy) ───────────────────────────────────────
        self.references: dict = {}
        self.reference_index_maps: dict = {}
        if references:
            for role, ref_coll in references.items():
                idx_map = (
                    list(reference_index_maps[role])
                    if (reference_index_maps and role in reference_index_maps)
                    else [0] * len(normed_items)
                )
                self._embed_reference(role, ref_coll, idx_map)

    # ── subclass hooks ────────────────────────────────────────────────────────

    @classmethod
    def _default_measurements_units(cls, method: str) -> str:
        """Default measurements_units string. Overridden by domain subclasses."""
        return ""

    # ── reference handling ────────────────────────────────────────────────────

    def _embed_reference(
        self,
        role: str,
        ref_coll: "_collection",
        index_map: list[int],
    ) -> None:
        """Embed-by-copy: snapshot ``ref_coll``'s ``measurements`` tree into our file."""
        if len(index_map) != len(self):
            raise ValueError(
                f"reference_index_maps['{role}'] must have length {len(self)}."
            )
        if max(index_map) >= len(ref_coll) or min(index_map) < 0:
            raise ValueError(
                f"reference_index_maps['{role}'] has out-of-range indices."
            )

        ref_path = ref_coll.path
        with h5py.File(ref_path, "r") as src:
            ref_grp = self.file.require_group(f"references/{role}")
            ref_grp.attrs["mode"] = "embedded"
            ref_grp.attrs["index_map"] = np.asarray(index_map, dtype=np.int64)
            ref_grp.attrs["collection_class"] = type(ref_coll).__name__
            ref_grp.attrs["n_domain_axes"] = ref_coll._n_domain_axes
            if "measurements" in ref_grp:
                del ref_grp["measurements"]
            src.copy("measurements", ref_grp, name="measurements")

        self.references[role] = ref_coll
        self.reference_index_maps[role] = list(index_map)

    def get_reference_data(self, role: str, i: int) -> np.ndarray:
        """Return the raw numpy array for the reference of item ``i`` under ``role``."""
        if role not in self.reference_index_maps:
            raise KeyError(f"No reference role '{role}' on this collection.")
        idx = self.reference_index_maps[role][i]
        ref_grp = self.file[f"references/{role}/measurements"]
        # find the i-th item name (deterministic ordering by insertion)
        ref_names = [
            k for k in ref_grp.keys() if k != "_axes"
        ]
        return ref_grp[f"{ref_names[idx]}/data"][()]

    def reference_roles(self) -> list[str]:
        """Return the list of reference roles attached to this collection."""
        return list(self.references.keys())

    # ── basic dunder / iteration ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.name)

    def __getitem__(self, key):
        """Restrict the active selection to a subset of items.

        Accepts a single int, slice, str, list of strs, or set of strs.
        Returns ``self`` after mutating the active selection.
        """
        all_stored = {
            k for k in self.file["measurements"].keys() if k != "_axes"
        }
        if isinstance(key, str):
            if key not in all_stored:
                raise KeyError(key)
            self.name = [key]
        elif isinstance(key, (list, set, tuple)) and all(
            isinstance(k, str) for k in key
        ):
            for k in key:
                if k not in all_stored:
                    raise KeyError(k)
            self.name = list(key)
        elif isinstance(key, int):
            self.name = [self.name[key]]
        elif isinstance(key, slice):
            self.name = self.name[key]
        else:
            raise TypeError(
                f"Unsupported key type {type(key).__name__} for collection indexing."
            )
        self.measurements = [
            self.file[f"measurements/{n}/data"] for n in self.name
        ]
        if self.labels is not None:
            self.labels = [self.file[f"measurements/{n}/label"] for n in self.name]
        return self

    def select_all(self):
        """Reset the active selection to every item present in the HDF5 file."""
        all_names = [
            k for k in self.file["measurements"].keys() if k != "_axes"
        ]
        return self[all_names]

    # ── append ────────────────────────────────────────────────────────────────

    def append(
        self,
        item: np.ndarray,
        name: Optional[str] = None,
        label: Optional[float] = None,
    ):
        """Append a single new item to the collection.

        Parameters
        ----------
        item : numpy.ndarray
            Measurement array of rank ``n_domain_axes + 2`` and matching shape.
        name : str, optional
            Identifying name; defaults to ``item_<n>``.
        label : float, optional
            Numeric label for the new item.
        """
        a = np.asarray(item)
        expected_rank = self._n_domain_axes + 2
        while a.ndim < expected_rank:
            a = a[..., np.newaxis]
        if a.ndim != expected_rank:
            raise ValueError(
                f"item has rank {a.ndim}, expected {expected_rank}."
            )
        if a.shape[-2:] != (self.n_outputs, self.n_inputs):
            raise ValueError(
                f"item channel shape {a.shape[-2:]} does not match collection "
                f"({self.n_outputs}, {self.n_inputs})."
            )
        if name is None:
            name = f"item_{len(self)}"
        new_name = add_suffix(self.name + [name])[-1]
        if np.iscomplexobj(a) and a.dtype == np.complex128:
            a = a.astype(np.complex64)
        item_grp = self.file["measurements"].create_group(new_name)
        item_grp.create_dataset("data", data=a)
        if label is not None:
            item_grp.create_dataset("label", data=label)
            if self.labels is None:
                # promote previously-unlabelled collection
                self.labels = []
                for n in self.name:
                    self.file[f"measurements/{n}/label"] = np.nan
                    self.labels.append(self.file[f"measurements/{n}/label"])
            self.labels.append(self.file[f"measurements/{new_name}/label"])
        elif self.labels is not None:
            self.file[f"measurements/{new_name}/label"] = np.nan
            self.labels.append(self.file[f"measurements/{new_name}/label"])
        self.name.append(new_name)
        self.measurements.append(self.file[f"measurements/{new_name}/data"])
        return self

    # ── train/val/test split ─────────────────────────────────────────────────

    def split(self, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42):
        """Stratified train/val/test index split, stored on ``self``."""
        if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
            raise ValueError("train_frac + val_frac + test_frac must equal 1.0.")
        rng = np.random.default_rng(seed)
        labels_list = self.labels if self.labels is not None else []
        groups: dict = {}
        for i, lbl in enumerate(labels_list):
            key = int(lbl[()] if hasattr(lbl, "__getitem__") else lbl)
            groups.setdefault(key, []).append(i)
        train_idx, val_idx, test_idx = [], [], []
        for key in sorted(groups):
            arr = np.array(groups[key])
            rng.shuffle(arr)
            n = len(arr)
            n_train = int(round(n * train_frac))
            n_val = int(round(n * val_frac))
            train_idx.extend(arr[:n_train].tolist())
            val_idx.extend(arr[n_train:n_train + n_val].tolist())
            test_idx.extend(arr[n_train + n_val:].tolist())
        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx
        return train_idx, val_idx, test_idx

    # ── PyTorch handoff ───────────────────────────────────────────────────────

    def torch_dataset(self):
        """Close the file and return a PyTorch Dataset wrapper bound to ``self.dataset``."""
        from pymodal.hdf5_dataset import HDF5Dataset
        self.close(keep=True)
        self.measurements = None
        self.labels = None
        self.dataset = HDF5Dataset(self.path)
        return self

    # ── file lifecycle ────────────────────────────────────────────────────────

    def open(self):
        """Reopen the HDF5 file and rebind ``measurements``/``labels`` handles."""
        if hasattr(self, "file") and self.file:
            try:
                if self.file.id.valid:
                    return self
            except Exception:
                pass
        self.file = h5py.File(self.path, "a")
        self.measurements = [
            self.file[f"measurements/{n}/data"] for n in self.name
        ]
        try:
            self.labels = [
                self.file[f"measurements/{n}/label"] for n in self.name
            ]
        except KeyError:
            self.labels = None
        return self

    def close(self, keep: bool = False):
        """Close the HDF5 file. By default the file on disk is also deleted.

        Parameters
        ----------
        keep : bool, optional
            Pass True to retain the file on disk (e.g. before ``torch_dataset()``).
        """
        try:
            self.file.close()
        except Exception:
            pass
        if not keep:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    # ── utility ───────────────────────────────────────────────────────────────

    def channel_shape(self) -> tuple:
        """Return the ``(n_outputs, n_inputs)`` channel shape."""
        return (self.n_outputs, self.n_inputs)

    def domain_shape(self) -> tuple:
        """Return the per-item domain-axis lengths, e.g. ``(n_freq,)`` or ``(n_freq1, n_freq2)``."""
        if self._n_domain_axes == 0:
            return ()
        return tuple(d.shape[0] for d in self.domain_arrays)
