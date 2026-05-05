"""2-D collection: items shaped ``(D1, D2, n_outputs, n_inputs)``.

Holds collections with two functional domain axes — for example CFDAC
(freq × freq) and FDAC (DOF × DOF).
"""

from __future__ import annotations

from typing import Optional
from warnings import catch_warnings, filterwarnings

import numpy as np

import pymodal
from pymodal.collection_parent import _collection


class _collection_2d(_collection):
    """Parent for collections with two functional domain axes."""

    _n_domain_axes = 2

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
                np.arange(shape[0], dtype=float),
                np.arange(shape[1], dtype=float),
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

    # ── domain edition (per-axis) ─────────────────────────────────────────────

    def change_domain_resolution(self, new_resolution: float, axis: int = 0):
        """Resample every item along the chosen domain axis."""
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1.")
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
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1.")
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

    @classmethod
    def from_pair_op(
        cls,
        reference,
        damaged,
        op,
        role_pair=("reference", "damaged"),
        path=None,
        domain_arrays: Optional[list[np.ndarray]] = None,
        domain_units: Optional[list[str]] = None,
        measurements_units: Optional[str] = None,
    ):
        """Build a 2-D collection by applying ``op(ref_i, dmg_i)`` per item.

        ``op`` returns either ``(D1, D2)`` or ``(D1, D2, n_outputs, n_inputs)``.
        """
        if len(reference) != len(damaged):
            ref_idx = [0] * len(damaged)
        else:
            ref_idx = list(range(len(damaged)))

        n_out, n_in = damaged.n_outputs, damaged.n_inputs
        items: list[np.ndarray] = []
        for i in range(len(damaged)):
            ref_arr = reference.measurements[ref_idx[i]][()]
            dmg_arr = damaged.measurements[i][()]
            mat = np.asarray(op(ref_arr, dmg_arr))
            if mat.ndim == 2:
                mat = np.broadcast_to(
                    mat[..., None, None], mat.shape + (n_out, n_in)
                ).copy()
            elif mat.ndim != 4:
                raise ValueError(
                    f"op returned array of rank {mat.ndim}; expected 2 or 4."
                )
            items.append(mat)

        labels = (
            [lbl[()] for lbl in damaged.labels]
            if damaged.labels is not None
            else None
        )

        return cls(
            items=items,
            domain_arrays=domain_arrays,
            domain_units=domain_units,
            coordinates=damaged.coordinates,
            orientations=damaged.orientations,
            dof=damaged.dof,
            method=damaged.method,
            space_units=damaged.space_units,
            measurements_units=measurements_units,
            names=list(damaged.name),
            labels=labels,
            references={role_pair[0]: reference, role_pair[1]: damaged},
            reference_index_maps={
                role_pair[0]: ref_idx,
                role_pair[1]: list(range(len(damaged))),
            },
            path=path,
        )
