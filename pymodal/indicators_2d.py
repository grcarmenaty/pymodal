"""2-D indicator collections — one (n_dof, n_dof) matrix per item.

Each class is a thin subclass of :class:`pymodal.collection_2d._collection_2d`
that pairs a reference :class:`pymodal.frf.frf` with a damaged ``frf``, applies
the matching pure function from :mod:`pymodal.utils`, and stores the resulting
matrix with two domain axes.

Item shape is ``(n_freq, n_freq, 1, 1)``. The two domain axes are
frequency-line indices inherited from the source FRFs.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

from pymodal import utils
from pymodal.collection_2d import _collection_2d


def _as_matrix(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(arr.shape[0], -1)  # (n_freq, n_dof)


def _pair_indices(reference, damaged) -> list[int]:
    if len(reference) == len(damaged):
        return list(range(len(damaged)))
    return [0] * len(damaged)


class _MatrixIndicatorCollection(_collection_2d):
    """Base for 2-D indicator collections built from a (reference, damaged) pair."""

    _indicator_op: Optional[Callable] = None
    _result_dtype = complex

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
                mat = op(_as_matrix(ref_arr), _as_matrix(dmg_arr), **op_kwargs)
                mat = np.asarray(mat).astype(cls._result_dtype)
                if mat.ndim != 2:
                    raise ValueError(
                        f"{cls.__name__}._op returned rank {mat.ndim}; expected 2."
                    )
                items.append(mat[..., None, None])  # (D1, D2, 1, 1)

        n_dof_ref, n_dof_dmg = items[0].shape[0], items[0].shape[1]
        labels = (
            [lbl[()] for lbl in damaged.labels]
            if damaged.labels is not None
            else None
        )
        return cls(
            items=items,
            domain_arrays=[
                np.arange(n_dof_ref, dtype=float),
                np.arange(n_dof_dmg, dtype=float),
            ],
            domain_units=["freq_index", "freq_index"],
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


class cfdac_collection(_MatrixIndicatorCollection):
    """Complex Frequency Domain Assurance Criterion."""

    @staticmethod
    def _op(ref_H, dmg_H):
        return utils.value_CFDAC(ref_H, dmg_H)

    _indicator_op = _op
    _result_dtype = complex


class cfdac_a_collection(_MatrixIndicatorCollection):
    """CFDAC alternative formulation."""

    @staticmethod
    def _op(ref_H, dmg_H):
        return utils.value_CFDAC_A(ref_H, dmg_H)

    _indicator_op = _op
    _result_dtype = complex


class fdac_collection(_MatrixIndicatorCollection):
    """Frequency Domain Assurance Criterion (real-valued)."""

    @staticmethod
    def _op(ref_H, dmg_H):
        return utils.value_FDAC(ref_H, dmg_H)

    _indicator_op = _op
    _result_dtype = float
