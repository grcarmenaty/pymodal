"""3-D collection: items shaped ``(D1, D2, D3, n_outputs, n_inputs)``.

Reserved for future use (e.g. time-resolved CFDAC). Mechanically identical
to :class:`_collection_2d`, generalised to three domain axes.
"""

from __future__ import annotations

from typing import Optional
from warnings import catch_warnings, filterwarnings

import numpy as np

import pymodal
from pymodal.collection_parent import _collection


class _collection_3d(_collection):
    """Parent for collections with three functional domain axes."""

    _n_domain_axes = 3

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
        path=None,
    ):
        if domain_arrays is None:
            shape = np.asarray(items[0]).shape
            domain_arrays = [
                np.arange(shape[k], dtype=float) for k in range(3)
            ]
        super().__init__(
            items=items,
            domain_arrays=domain_arrays,
            domain_units=domain_units,
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

    def change_domain_resolution(self, new_resolution: float, axis: int = 0):
        """Resample every item along the chosen domain axis."""
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2.")
        old_axis = self.domain_arrays[axis]
        new_axis = None
        for i, name in enumerate(self.name):
            arr = self.measurements[i][()]
            arr_moved = np.moveaxis(arr, axis, 0)
            with catch_warnings():
                filterwarnings("ignore", category=UserWarning)
                new_axis, new_moved = pymodal.change_domain_resolution(
                    domain_array=old_axis,
                    measurements_array=arr_moved,
                    new_resolution=new_resolution,
                )
            new_arr = np.moveaxis(new_moved, 0, axis)
            del self.file[f"measurements/{name}/data"]
            self.file[f"measurements/{name}/data"] = new_arr
        del self.file[f"measurements/_axes/axis_{axis}"]
        ds = self.file["measurements/_axes"].create_dataset(
            f"axis_{axis}", data=new_axis
        )
        ds.attrs["units"] = self.domain_units[axis]
        self.domain_arrays[axis] = new_axis
        self.measurements = [
            self.file[f"measurements/{n}/data"] for n in self.name
        ]
        return self

    def change_domain_span(
        self,
        new_min: Optional[float] = None,
        new_max: Optional[float] = None,
        axis: int = 0,
    ):
        """Crop or extend every item along the chosen domain axis."""
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2.")
        old_axis = self.domain_arrays[axis]
        new_axis = None
        for i, name in enumerate(self.name):
            arr = self.measurements[i][()]
            arr_moved = np.moveaxis(arr, axis, 0)
            with catch_warnings():
                filterwarnings("ignore", category=UserWarning)
                new_axis, new_moved = pymodal.change_domain_span(
                    domain_array=old_axis,
                    measurements_array=arr_moved,
                    new_min_domain=new_min,
                    new_max_domain=new_max,
                )
            new_arr = np.moveaxis(new_moved, 0, axis)
            del self.file[f"measurements/{name}/data"]
            self.file[f"measurements/{name}/data"] = new_arr
        del self.file[f"measurements/_axes/axis_{axis}"]
        ds = self.file["measurements/_axes"].create_dataset(
            f"axis_{axis}", data=new_axis
        )
        ds.attrs["units"] = self.domain_units[axis]
        self.domain_arrays[axis] = new_axis
        self.measurements = [
            self.file[f"measurements/{n}/data"] for n in self.name
        ]
        return self
