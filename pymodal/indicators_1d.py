"""1-D indicator collections — one (n_dof,) vector per item.

Each class is a thin subclass of :class:`pymodal.collection_1d._collection_1d`
that pairs a reference :class:`pymodal.frf.frf` with a damaged ``frf``,
applies the matching pure function from :mod:`pymodal.utils`, and stores
the resulting vector as a 1-item domain axis.

Item shape is ``(n_dof, 1, 1)``. The single domain axis represents the DOF
index inherited from the source FRFs.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

from pymodal import utils
from pymodal.collection_1d import _collection_1d


def _as_matrix(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], -1)  # (n_freq, n_dof)


def _pair_indices(reference, damaged) -> list[int]:
    if len(reference) == len(damaged):
        return list(range(len(damaged)))
    return [0] * len(damaged)


class _VectorIndicatorCollection(_collection_1d):
    """Base for 1-D indicator collections built from a (reference, damaged) pair."""

    _indicator_op: Optional[Callable] = None
    _default_units = "dimensionless"

    @classmethod
    def from_pair(cls, reference, damaged, path=None, **op_kwargs):
        op = cls._indicator_op
        if op is None:
            raise NotImplementedError(
                f"{cls.__name__} must define _indicator_op."
            )
        ref_idx = _pair_indices(reference, damaged)
        items: list[np.ndarray] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(len(damaged)):
                ref_arr = reference.measurements[ref_idx[i]][()]
                dmg_arr = damaged.measurements[i][()]
                vec = op(_as_matrix(ref_arr), _as_matrix(dmg_arr), **op_kwargs)
                vec = np.asarray(vec).astype(float).reshape(-1)
                items.append(vec[:, None, None])  # (n_dof, 1, 1)

        n_dof = items[0].shape[0]
        labels = (
            [lbl[()] for lbl in damaged.labels]
            if damaged.labels is not None
            else None
        )
        return cls(
            items=items,
            domain_array=np.arange(n_dof, dtype=float),
            domain_units="dof_index",
            method="SIMO",
            measurements_units="",
            names=list(damaged.name),
            labels=labels,
            references={"reference": reference, "damaged": damaged},
            reference_index_maps={
                "reference": ref_idx,
                "damaged": list(range(len(damaged))),
            },
            path=path,
        )


class rvac_collection(_VectorIndicatorCollection):
    """Response Vector Assurance Criterion — one value per DOF."""

    @staticmethod
    def _op(ref_H, dmg_H):
        return utils.value_RVAC(ref_H, dmg_H)

    _indicator_op = _op


class rvac_2d_collection(_VectorIndicatorCollection):
    """Curvature-based RVAC (second finite difference) per DOF."""

    @staticmethod
    def _op(ref_H, dmg_H):
        return utils.value_RVAC_2d(ref_H, dmg_H)

    _indicator_op = _op


class gac_collection(_VectorIndicatorCollection):
    """Global Assurance Criterion per DOF."""

    @staticmethod
    def _op(ref_H, dmg_H):
        return utils.value_GAC(ref_H, dmg_H)

    _indicator_op = _op


class m2l_collection(_VectorIndicatorCollection):
    """Mode-to-Location indicator per DOF."""

    @staticmethod
    def _op(ref_H, dmg_H):
        cfdac_mat = np.abs(utils.value_CFDAC(ref_H, dmg_H))
        return utils.M2L(cfdac_mat)

    _indicator_op = _op
