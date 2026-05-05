"""0-D indicator collections — one scalar per item.

Each class is a thin subclass of :class:`pymodal.collection_0d._collection_0d`
that pairs a reference :class:`pymodal.frf.frf` with a damaged ``frf`` and
applies the matching pure function from :mod:`pymodal.utils` to produce one
scalar per item. The reference and damaged collections are embedded into
the resulting indicator collection so the provenance is fully traceable.

Item shape is ``(1, 1)`` — channel/DOF metadata is preserved on the
embedded references rather than duplicated in the indicator values.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

from pymodal import utils
from pymodal.collection_0d import _collection_0d


# ── Internal helpers ─────────────────────────────────────────────────────────

def _as_matrix(arr: np.ndarray) -> np.ndarray:
    """Reshape a 3-D ``(n_freq, n_outputs, n_inputs)`` FRF into ``(n_dof, n_freq)``."""
    return arr.reshape(arr.shape[0], -1).T


def _pair_indices(reference, damaged) -> list[int]:
    """Per-item reference index map: identity if lengths match, else all-zero."""
    if len(reference) == len(damaged):
        return list(range(len(damaged)))
    return [0] * len(damaged)


class _ScalarIndicatorCollection(_collection_0d):
    """Base for 0-D indicator collections built from a (reference, damaged) pair.

    Subclasses set the class attribute ``_indicator_op`` to a callable
    ``op(ref_matrix, dmg_matrix) -> float`` where each matrix is
    ``(n_dof, n_freq)``.
    """

    _indicator_op: Optional[Callable] = None

    @classmethod
    def from_pair(
        cls,
        reference,
        damaged,
        path=None,
        **op_kwargs,
    ):
        """Construct the indicator collection by applying the op per item."""
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
                value = op(_as_matrix(ref_arr), _as_matrix(dmg_arr), **op_kwargs)
                items.append(np.array([[float(np.nan_to_num(value))]]))

        labels = (
            [lbl[()] for lbl in damaged.labels]
            if damaged.labels is not None
            else None
        )
        return cls(
            items=items,
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


# ── Concrete indicator classes ───────────────────────────────────────────────

class sci_collection(_ScalarIndicatorCollection):
    """Structural Change Indicator — signed, vs reference FRF."""

    @staticmethod
    def _op(ref_H: np.ndarray, dmg_H: np.ndarray) -> float:
        cfdac_ref = np.abs(utils.value_CFDAC(ref_H, ref_H))
        cfdac_dmg = np.abs(utils.value_CFDAC(ref_H, dmg_H))
        return float(utils.SCI(cfdac_ref, cfdac_dmg))

    _indicator_op = _op


class unsigned_sci_collection(_ScalarIndicatorCollection):
    """Unsigned Structural Change Indicator vs reference FRF."""

    @staticmethod
    def _op(ref_H: np.ndarray, dmg_H: np.ndarray) -> float:
        cfdac_ref = np.abs(utils.value_CFDAC(ref_H, ref_H))
        cfdac_dmg = np.abs(utils.value_CFDAC(ref_H, dmg_H))
        return float(utils.unsigned_SCI(cfdac_ref, cfdac_dmg))

    _indicator_op = _op


class drq_collection(_ScalarIndicatorCollection):
    """Damage Residual Quantifier — mean of RVAC vector."""

    @staticmethod
    def _op(ref_H: np.ndarray, dmg_H: np.ndarray) -> float:
        return float(utils.DRQ(utils.value_RVAC(ref_H, dmg_H)))

    _indicator_op = _op


class aigac_collection(_ScalarIndicatorCollection):
    """Average Index from the Global Assurance Criterion."""

    @staticmethod
    def _op(ref_H: np.ndarray, dmg_H: np.ndarray) -> float:
        return float(utils.AIGAC(utils.value_GAC(ref_H, dmg_H)))

    _indicator_op = _op


class frfrms_collection(_ScalarIndicatorCollection):
    """FRF RMS metric (log scale)."""

    @staticmethod
    def _op(ref_H: np.ndarray, dmg_H: np.ndarray) -> float:
        return float(utils.FRFRMS(ref_H, dmg_H))

    _indicator_op = _op


class frfsf_collection(_ScalarIndicatorCollection):
    """FRF Scale Factor."""

    @staticmethod
    def _op(ref_H: np.ndarray, dmg_H: np.ndarray) -> float:
        return float(utils.FRFSF(ref_H, dmg_H))

    _indicator_op = _op


class frfsm_collection(_ScalarIndicatorCollection):
    """FRF Similarity Measure (Gaussian, parameter ``std`` in dB)."""

    @staticmethod
    def _op(ref_H: np.ndarray, dmg_H: np.ndarray, std: float = 6.0) -> float:
        return float(utils.FRFSM(ref_H, dmg_H, std=std))

    _indicator_op = _op


class ods_diff_collection(_ScalarIndicatorCollection):
    """Summed absolute Operating Deflection Shape difference."""

    @staticmethod
    def _op(ref_H: np.ndarray, dmg_H: np.ndarray) -> float:
        return float(utils.ODS_diff(ref_H, dmg_H))

    _indicator_op = _op


class r2_imag_collection(_ScalarIndicatorCollection):
    """Coefficient of determination of the imaginary part."""

    @staticmethod
    def _op(ref_H: np.ndarray, dmg_H: np.ndarray) -> float:
        return float(utils.r2_imag(ref_H, dmg_H))

    _indicator_op = _op
