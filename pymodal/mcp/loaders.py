"""Helpers to round-trip pymodal collections through HDF5 files on disk.

Pymodal collections are HDF5-backed by design: every collection has a backing
``.h5`` file that contains the data, the channel/DOF metadata, the domain
axes, and any embedded reference collections. The MCP layer treats those
HDF5 files as the *only* persistent identifier for a collection — every tool
takes file paths in and writes file paths out, with no in-memory state shared
between calls.

To make pymodal's mutation methods (``change_freq_span``, ``to_FRF``, the
indicator factories, …) usable from a stateless tool API, we need to be able
to reconstruct a typed collection (``frf``, ``timeseries``) from a previously
written file. The functions in this module do exactly that: they read the
HDF5 layout described in :mod:`pymodal.collection_parent` and call the
appropriate constructor with the data and metadata they recovered, optionally
writing the rebuilt collection to a new path so the source file stays
untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

import pymodal


def _coerce(value):
    """Decode bytes/numpy scalars from HDF5 attrs into plain Python objects."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def read_collection_payload(path: str | Path) -> dict:
    """Read everything needed to reconstruct a 1-D collection from ``path``.

    Returns a dict with raw numpy arrays for items, the domain axis,
    coordinates, orientations, and the channel/measurement metadata. Used by
    :func:`load_frf` and :func:`load_timeseries`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No HDF5 file at {path}")
    with h5py.File(path, "r") as f:
        meas = f["measurements"]
        attrs = {k: _coerce(v) for k, v in meas.attrs.items()}
        n_axes = int(attrs.get("n_domain_axes", 0))
        names = [k for k in meas.keys() if k != "_axes"]
        items = [meas[f"{n}/data"][()] for n in names]
        labels: list[Optional[float]] = []
        any_label = False
        for n in names:
            grp = meas[n]
            if "label" in grp:
                labels.append(float(np.asarray(grp["label"]).item()))
                any_label = True
            else:
                labels.append(None)
        domain_arrays: list[np.ndarray] = []
        domain_units: list[str] = []
        for k in range(n_axes):
            ax = meas[f"_axes/axis_{k}"]
            domain_arrays.append(np.asarray(ax[()], dtype=float))
            domain_units.append(_coerce(ax.attrs.get("units", "")))
    return {
        "items": items,
        "names": names,
        "labels": labels if any_label else None,
        "method": attrs.get("method", "SIMO"),
        "n_outputs": int(attrs.get("n_outputs", 1)),
        "n_inputs": int(attrs.get("n_inputs", 1)),
        "dof": int(attrs.get("dof", max(int(attrs.get("n_outputs", 1)), int(attrs.get("n_inputs", 1))))),
        "coordinates": np.asarray(attrs.get("coordinates")),
        "orientations": np.asarray(attrs.get("orientations")),
        "space_units": attrs.get("space_units", "millimeter"),
        "measurements_units": attrs.get("measurements_units", ""),
        "n_domain_axes": n_axes,
        "domain_arrays": domain_arrays,
        "domain_units": domain_units,
    }


def load_frf(path: str | Path, output_path: Optional[str | Path] = None) -> pymodal.frf:
    """Reconstruct a :class:`pymodal.frf` from an HDF5 file.

    Pass ``output_path`` to write the rebuilt collection to a new location;
    otherwise pymodal will auto-generate a fresh path so the source file is
    not overwritten.
    """
    p = read_collection_payload(path)
    if p["n_domain_axes"] != 1:
        raise ValueError(
            f"{path} stores a {p['n_domain_axes']}-D collection; "
            f"expected a 1-D collection (frf or timeseries)."
        )
    return pymodal.frf(
        measurements=p["items"],
        coordinates=p["coordinates"],
        orientations=p["orientations"],
        dof=p["dof"],
        freq_array=p["domain_arrays"][0],
        freq_units=p["domain_units"][0],
        measurements_units=p["measurements_units"],
        space_units=p["space_units"],
        method=p["method"],
        names=p["names"],
        labels=p["labels"],
        path=output_path,
    )


def load_timeseries(
    path: str | Path,
    output_path: Optional[str | Path] = None,
) -> pymodal.timeseries:
    """Reconstruct a :class:`pymodal.timeseries` from an HDF5 file."""
    p = read_collection_payload(path)
    if p["n_domain_axes"] != 1:
        raise ValueError(
            f"{path} stores a {p['n_domain_axes']}-D collection; "
            f"expected a 1-D collection."
        )
    return pymodal.timeseries(
        measurements=p["items"],
        coordinates=p["coordinates"],
        orientations=p["orientations"],
        dof=p["dof"],
        time_array=p["domain_arrays"][0],
        time_units=p["domain_units"][0],
        measurements_units=p["measurements_units"],
        space_units=p["space_units"],
        method=p["method"],
        names=p["names"],
        labels=p["labels"],
        path=output_path,
    )


def describe_collection(path: str | Path) -> dict:
    """Return a JSON-serialisable summary of the collection stored at ``path``.

    Reads the HDF5 file directly without instantiating a collection class —
    safe to call on any file written by pymodal regardless of which subclass
    produced it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No HDF5 file at {path}")
    with h5py.File(path, "r") as f:
        meas = f["measurements"]
        attrs = {k: _coerce(v) for k, v in meas.attrs.items()}
        n_axes = int(attrs.get("n_domain_axes", 0))
        names = [k for k in meas.keys() if k != "_axes"]
        first_data = meas[f"{names[0]}/data"]
        item_shape = tuple(int(s) for s in first_data.shape)
        dtype = str(first_data.dtype)
        axes = []
        for k in range(n_axes):
            ax = meas[f"_axes/axis_{k}"]
            arr = np.asarray(ax[()], dtype=float)
            axes.append({
                "index": k,
                "length": int(arr.shape[0]),
                "min": float(arr[0]),
                "max": float(arr[-1]),
                "resolution": float(np.mean(np.diff(arr))) if arr.shape[0] > 1 else 0.0,
                "units": _coerce(ax.attrs.get("units", "")),
            })
        labels: list[Optional[float]] = []
        for n in names:
            grp = meas[n]
            if "label" in grp:
                labels.append(float(np.asarray(grp["label"]).item()))
            else:
                labels.append(None)
        reference_roles = []
        if "references" in f:
            for role in f["references"].keys():
                ref_grp = f[f"references/{role}"]
                ref_attrs = {k: _coerce(v) for k, v in ref_grp.attrs.items()}
                reference_roles.append({
                    "role": role,
                    "collection_class": ref_attrs.get("collection_class"),
                    "n_domain_axes": int(ref_attrs.get("n_domain_axes", 0)),
                    "mode": ref_attrs.get("mode", "embedded"),
                })
        attached = {}
        if "attached" in f:
            attached = {k: _coerce(v) for k, v in f["attached"].attrs.items()}
    return {
        "path": str(Path(path).resolve()),
        "n_items": len(names),
        "names": names,
        "item_shape": list(item_shape),
        "dtype": dtype,
        "method": attrs.get("method"),
        "n_outputs": int(attrs.get("n_outputs", 0)),
        "n_inputs": int(attrs.get("n_inputs", 0)),
        "dof": int(attrs.get("dof", 0)),
        "n_domain_axes": n_axes,
        "domain_axes": axes,
        "measurements_units": attrs.get("measurements_units"),
        "space_units": attrs.get("space_units"),
        "labels": labels,
        "reference_roles": reference_roles,
        "attached_files": attached,
    }
