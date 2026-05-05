"""1-D collection: items shaped ``(D1, n_outputs, n_inputs)``.

Holds collections with a single functional domain axis: time-series, FRFs,
and per-DOF vector indicators (RVAC, GAC, M2L, RVAC-2D).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Union
from warnings import catch_warnings, filterwarnings

import h5py
import numpy as np

import pymodal
from pymodal.collection_parent import _collection


class _collection_1d(_collection):
    """Parent for collections with one functional domain axis."""

    _n_domain_axes = 1

    def __init__(
        self,
        items: list[np.ndarray],
        domain_array: Optional[np.ndarray] = None,
        domain_start: Optional[float] = None,
        domain_end: Optional[float] = None,
        domain_resolution: Optional[float] = None,
        domain_units: Optional[str] = None,
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
        path=None,
    ):
        # Accept either an explicit domain_array or (start, resolution) inferred from
        # the first item's leading axis length.
        if domain_array is None:
            n = np.asarray(items[0]).shape[0]
            if domain_resolution is None and domain_end is None:
                raise ValueError(
                    "Provide either `domain_array` or "
                    "`domain_resolution` (with optional `domain_start`, `domain_end`)."
                )
            start = float(domain_start) if domain_start is not None else 0.0
            if domain_resolution is not None:
                domain_array = np.arange(n, dtype=float) * float(domain_resolution) + start
            else:
                domain_array = np.linspace(start, float(domain_end), n)
        else:
            domain_array = np.asarray(domain_array, dtype=float)

        super().__init__(
            items=items,
            domain_arrays=[domain_array],
            domain_units=[domain_units] if domain_units is not None else None,
            coordinates=coordinates,
            orientations=orientations,
            dof=dof,
            method=method,
            space_units=space_units,
            measurements_units=measurements_units,
            names=names,
            labels=labels,
            references=references,
            reference_index_maps=reference_index_maps,
            path=path,
        )

        self.domain_array = self.domain_arrays[0]
        self.domain_units_str = self.domain_units[0]
        self.domain_start = float(self.domain_array[0])
        self.domain_end = float(self.domain_array[-1])
        self.domain_span = self.domain_end - self.domain_start
        self.domain_resolution = (
            float(np.mean(np.diff(self.domain_array)))
            if len(self.domain_array) > 1
            else 0.0
        )

    # ── domain edition ────────────────────────────────────────────────────────

    def change_domain_resolution(self, new_resolution: float):
        """Resample every item to ``new_resolution`` along the domain axis."""
        new_domain = None
        for i, name in enumerate(self.name):
            arr = self.measurements[i][()]
            with catch_warnings():
                filterwarnings("ignore", category=UserWarning)
                new_domain, new_arr = pymodal.change_domain_resolution(
                    domain_array=self.domain_array,
                    measurements_array=arr,
                    new_resolution=new_resolution,
                )
            del self.file[f"measurements/{name}/data"]
            self.file[f"measurements/{name}/data"] = new_arr
        # update axis dataset
        del self.file["measurements/_axes/axis_0"]
        ds = self.file["measurements/_axes"].create_dataset("axis_0", data=new_domain)
        ds.attrs["units"] = self.domain_units_str
        self.domain_array = new_domain
        self.domain_arrays = [new_domain]
        self.domain_start = float(new_domain[0])
        self.domain_end = float(new_domain[-1])
        self.domain_span = self.domain_end - self.domain_start
        self.domain_resolution = float(np.mean(np.diff(new_domain))) if len(new_domain) > 1 else 0.0
        self.measurements = [
            self.file[f"measurements/{n}/data"] for n in self.name
        ]
        return self

    def change_domain_span(
        self,
        new_min: Optional[float] = None,
        new_max: Optional[float] = None,
    ):
        """Crop or extend every item along the domain axis."""
        new_domain = None
        for i, name in enumerate(self.name):
            arr = self.measurements[i][()]
            with catch_warnings():
                filterwarnings("ignore", category=UserWarning)
                new_domain, new_arr = pymodal.change_domain_span(
                    domain_array=self.domain_array,
                    measurements_array=arr,
                    new_min_domain=new_min,
                    new_max_domain=new_max,
                )
            del self.file[f"measurements/{name}/data"]
            self.file[f"measurements/{name}/data"] = new_arr
        del self.file["measurements/_axes/axis_0"]
        ds = self.file["measurements/_axes"].create_dataset("axis_0", data=new_domain)
        ds.attrs["units"] = self.domain_units_str
        self.domain_array = new_domain
        self.domain_arrays = [new_domain]
        self.domain_start = float(new_domain[0])
        self.domain_end = float(new_domain[-1])
        self.domain_span = self.domain_end - self.domain_start
        self.domain_resolution = float(np.mean(np.diff(new_domain))) if len(new_domain) > 1 else 0.0
        self.measurements = [
            self.file[f"measurements/{n}/data"] for n in self.name
        ]
        return self
