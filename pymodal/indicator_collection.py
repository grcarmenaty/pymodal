"""IndicatorCollection — HDF5-backed SHM indicator collection.

Each item stores an indicator matrix/vector/scalar derived by comparing one
signal from a source FRF collection against a reference FRF.  The collection
persists the reference data inside its own HDF5 file so that every item can
retrieve the exact reference FRF used to produce it.

Edition methods
---------------
select_dofs(indices)
    For matrix indicators (cfdac, cfdac_a, fdac): slices both the row- and
    column-DOF axes simultaneously so the result remains square.
    For vector indicators (rvac, rvac_2d, gac, m2l): slices the DOF axis.
    For scalar indicators: no-op.
change_freq_span / change_freq_resolution
    Inherited from frf_collection.  For vector indicators these operate on
    the DOF axis as expected.  For matrix indicators they are redirected to
    select_dofs to keep the result square.
"""

from __future__ import annotations

import h5py
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union
from warnings import catch_warnings, filterwarnings

from pymodal import frf as _frf
from pymodal import frf_collection as _frf_collection


# ── Indicator-type constants ──────────────────────────────────────────────────

MATRIX_INDICATORS = frozenset(("cfdac", "cfdac_a", "fdac"))
VECTOR_INDICATORS = frozenset(("rvac", "rvac_2d", "gac", "m2l"))
SCALAR_INDICATORS = frozenset((
    "sci", "unsigned_sci", "drq", "aigac",
    "frfrms", "frfsf", "frfsm", "ods_diff", "r2_imag",
))


# ── Module-level worker (must be picklable for future multiprocessing) ────────

def _select_dofs_worker(var):
    """Load one indicator from HDF5, apply DOF selection, write back.

    Parameters
    ----------
    var : tuple
        ``(collection, i, indices, indicator_type)``

    Returns
    -------
    _frf
        A fresh frf instance built from the selected data, used only for
        attribute extraction.  Its ``measurements`` attribute is deleted
        before returning.
    """
    collection, i, indices, indicator_type = var
    name = collection.name[i]

    with h5py.File(collection.path, "r") as f:
        data = f[f"measurements/{name}/data"][()]

    if indicator_type in MATRIX_INDICATORS:
        new_data = data[indices][:, indices]   # (n_sel, n_sel, 1)
    elif indicator_type in VECTOR_INDICATORS:
        new_data = data[np.array(indices)]      # (n_sel, 1, 1)
    else:
        new_data = data                          # scalar: no-op

    with h5py.File(collection.path, "a") as f:
        del f[f"measurements/{name}/data"]
        f[f"measurements/{name}/data"] = new_data

    with catch_warnings():
        filterwarnings("ignore", message="The unit of the quantity is stripped")
        filterwarnings("ignore", message="Coordinates will be assumed")
        filterwarnings("ignore", message="Orientations will be assumed")
        updated = _frf(
            measurements=new_data,
            freq_resolution=1.0,
            measurements_units="",
            method="SIMO",
            name=name,
        )
    del updated.measurements
    return updated


# ── IndicatorCollection ───────────────────────────────────────────────────────

class IndicatorCollection(_frf_collection):
    """HDF5-backed SHM indicator collection with per-item reference tracking.

    Inherits all storage and PyTorch-pipeline infrastructure from
    :class:`frf_collection`.  Additionally stores one or more reference FRFs
    inside the same HDF5 file (under ``references/``) and records, for each
    item, which reference was used.

    Parameters
    ----------
    frfs_list : list of frf
        Indicator-wrapped frf objects (as produced by
        :meth:`frf_collection._indicator_frf`).
    labels : list of float, optional
        Numeric label per item, forwarded to the parent.
    ref_frfs : list of frf
        One or more reference FRF objects.  Pass a single-element list when
        all items share the same reference; pass one per item when each item
        has its own reference.
    ref_indices : list of int, optional
        ``ref_indices[i]`` is the index into ``ref_frfs`` of the reference
        used to compute item ``i``.  Defaults to ``[0] * len(frfs_list)``.
    indicator_type : str, optional
        One of the strings in ``MATRIX_INDICATORS``, ``VECTOR_INDICATORS``,
        or ``SCALAR_INDICATORS``.
    path : Path or str, optional
        HDF5 file for the collection.  Auto-generated if None.

    Attributes
    ----------
    indicator_type : str
    ref_indices : list of int
    """

    def __init__(
        self,
        frfs_list: list,
        labels: Optional[list] = None,
        ref_frfs: Optional[list] = None,
        ref_indices: Optional[list] = None,
        indicator_type: Optional[str] = None,
        path: Optional[Path] = None,
    ):
        super().__init__(frfs_list, labels=labels, path=path)

        self.indicator_type = indicator_type
        self.ref_indices = (
            ref_indices if ref_indices is not None else [0] * len(frfs_list)
        )

        # ── persist reference FRFs in the same HDF5 file ─────────────────────
        if ref_frfs:
            with catch_warnings():
                filterwarnings(
                    "ignore", message="The unit of the quantity is stripped"
                )
                for j, ref_frf in enumerate(ref_frfs):
                    mag = ref_frf.measurements.magnitude
                    self.file[f"references/ref_{j}/data"] = mag

        # ── persist metadata as HDF5 attributes ───────────────────────────────
        self.file["measurements"].attrs["_ref_indices"] = self.ref_indices
        if indicator_type is not None:
            self.file["measurements"].attrs["_indicator_type"] = indicator_type

    # ── reference retrieval ───────────────────────────────────────────────────

    @property
    def n_references(self) -> int:
        """Number of distinct reference FRFs stored in this collection."""
        if "references" not in self.file:
            return 0
        return len(self.file["references"])

    def get_reference(self, i: int) -> _frf:
        """Return the reference :class:`frf` used to compute item *i*.

        The reference data is read from the ``references/`` group of the
        collection's own HDF5 file.

        Parameters
        ----------
        i : int
            Index of the indicator item.

        Returns
        -------
        frf
        """
        if "references" not in self.file:
            raise ValueError("No reference FRFs are stored in this collection.")
        ref_idx = self.ref_indices[i]
        data = self.file[f"references/ref_{ref_idx}/data"][()]
        with catch_warnings():
            filterwarnings("ignore", message="Coordinates will be assumed")
            filterwarnings("ignore", message="Orientations will be assumed")
            return _frf(
                measurements=data,
                freq_resolution=self.freq_resolution.magnitude,
                measurements_units=str(self.measurements_units.units),
                method=self.method,
                name=f"ref_{ref_idx}",
            )

    # ── DOF-axis edition ──────────────────────────────────────────────────────

    def select_dofs(
        self, indices: Union[list, slice]
    ) -> "IndicatorCollection":
        """Select a subset of DOF indices from every indicator measurement.

        For **matrix** indicators (cfdac, cfdac_a, fdac) both row and column
        DOF axes are sliced simultaneously so the result remains square.

        For **vector** indicators (rvac, rvac_2d, gac, m2l) only the single
        DOF axis is sliced.

        For **scalar** indicators this is a no-op.

        Parameters
        ----------
        indices : list of int or slice
            DOF positions to keep.  At least 2 indices are required for the
            underlying frf storage to remain valid.

        Returns
        -------
        self
        """
        if self.indicator_type in SCALAR_INDICATORS:
            return self

        n_dof = self.measurements[0].shape[0]
        if isinstance(indices, slice):
            indices = list(range(*indices.indices(n_dof)))

        vars_ = [
            (self, i, indices, self.indicator_type) for i in range(len(self))
        ]
        self.file.close()
        del self.file
        del self.measurements

        for var in vars_:
            working_instance = _select_dofs_worker(var)

        attrs_to_update = deepcopy(self.attributes)
        attrs_to_update.remove("measurements")
        attrs_to_update.remove("name")

        self.file = h5py.File(self.path, "a")
        self.measurements = [
            self.file[f"measurements/{name}/data"] for name in self.name
        ]
        for attr in attrs_to_update:
            val = getattr(working_instance, attr)
            try:
                self.file["measurements"].attrs[attr] = val
            except Exception:
                pass
            setattr(self, attr, val)

        del working_instance
        return self

    # ── override freq-span/resolution for matrix indicators ──────────────────

    def change_freq_span(
        self, new_min_freq=None, new_max_freq=None
    ) -> "IndicatorCollection":
        """For matrix indicators: redirect to :meth:`select_dofs`.

        For vector and scalar indicators: delegates to the parent
        :class:`frf_collection` implementation which operates on the DOF
        (domain) axis directly.
        """
        if self.indicator_type in MATRIX_INDICATORS:
            n_dof = self.measurements[0].shape[0]
            lo = int(round(new_min_freq)) if new_min_freq is not None else 0
            hi = int(round(new_max_freq)) + 1 if new_max_freq is not None else n_dof
            return self.select_dofs(list(range(lo, min(hi, n_dof))))
        return super().change_freq_span(new_min_freq, new_max_freq)

    def change_freq_resolution(
        self, freq_resolution=None
    ) -> "IndicatorCollection":
        """For matrix indicators: redirect to :meth:`select_dofs` (stride).

        Selects every *k*-th DOF where *k = round(freq_resolution)*.
        For vector/scalar indicators delegates to the parent.
        """
        if self.indicator_type in MATRIX_INDICATORS:
            n_dof = self.measurements[0].shape[0]
            stride = max(1, int(round(freq_resolution)))
            return self.select_dofs(list(range(0, n_dof, stride)))
        return super().change_freq_resolution(freq_resolution)
