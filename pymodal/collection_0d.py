"""0-D collection: items shaped ``(n_outputs, n_inputs)``, no domain axis.

Holds collections of scalar-per-channel quantities such as SCI, DRQ, FRFRMS,
FRFSF, FRFSM, ODS-diff, R²-imag, AIGAC, unsigned-SCI. Channel/DOF metadata
remains meaningful even though there is no functional domain.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from pymodal.collection_parent import _collection


class _collection_0d(_collection):
    """Parent for scalar-per-channel collections."""

    _n_domain_axes = 0

    def __init__(
        self,
        items: list[np.ndarray],
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
        super().__init__(
            items=items,
            domain_arrays=None,
            domain_units=None,
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

    @classmethod
    def from_pair_op(
        cls,
        reference,
        damaged,
        op,
        role_pair=("reference", "damaged"),
        path=None,
        measurements_units: Optional[str] = None,
    ):
        """Build a 0-D collection by applying ``op(ref_i, dmg_i)`` per item.

        Parameters
        ----------
        reference, damaged : _collection_1d
            Source collections (same length, same channel shape).
        op : callable
            ``op(ref_array, damaged_array) -> scalar or (n_outputs, n_inputs) array``.
        role_pair : tuple of (str, str)
            Reference role names to register, default ``("reference", "damaged")``.
        path : Path or str, optional
        measurements_units : str, optional
        """
        if len(reference) != len(damaged):
            ref_idx = [0] * len(damaged)
        else:
            ref_idx = list(range(len(damaged)))

        items: list[np.ndarray] = []
        for i in range(len(damaged)):
            ref_arr = reference.measurements[ref_idx[i]][()]
            dmg_arr = damaged.measurements[i][()]
            scalar = op(ref_arr, dmg_arr)
            arr = np.asarray(scalar)
            if arr.ndim == 0:
                arr = np.full((damaged.n_outputs, damaged.n_inputs), arr)
            elif arr.shape != (damaged.n_outputs, damaged.n_inputs):
                arr = np.broadcast_to(
                    arr, (damaged.n_outputs, damaged.n_inputs)
                ).copy()
            items.append(arr.astype(np.float64))

        labels = (
            [lbl[()] for lbl in damaged.labels]
            if damaged.labels is not None
            else None
        )

        return cls(
            items=items,
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
