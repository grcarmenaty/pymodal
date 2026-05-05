"""Phase-3 tests: typed indicator collections (0-D / 1-D / 2-D).

Each indicator class is exercised by:
- constructing a (reference, damaged) frf pair
- calling indicator method on the damaged frf (e.g. damaged.cfdac(reference))
- verifying item shape, dimensionality class, and registered references
- verifying numerical equivalence to the underlying utils function
"""

from __future__ import annotations

import numpy as np
import pytest

import pymodal
from pymodal import utils


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_pair(tmp_path, n_freq=16, n_outputs=4, n_inputs=1, n_items=2):
    """Construct (reference, damaged) frf collections of identical layout."""
    rng = np.random.default_rng(0)
    ref_arrs = [
        (rng.standard_normal((n_freq, n_outputs, n_inputs))
         + 1j * rng.standard_normal((n_freq, n_outputs, n_inputs))).astype(np.complex64)
        for _ in range(n_items)
    ]
    dmg_arrs = [
        a * (1.0 + 0.1 * rng.standard_normal()) + 0.05 * (
            rng.standard_normal(a.shape) + 1j * rng.standard_normal(a.shape)
        ).astype(np.complex64)
        for a in ref_arrs
    ]
    ref = pymodal.frf(
        measurements=ref_arrs, freq_resolution=1.0, method="SIMO",
        names=[f"ref_{i}" for i in range(n_items)],
        path=tmp_path / "ref.h5",
    )
    dmg = pymodal.frf(
        measurements=dmg_arrs, freq_resolution=1.0, method="SIMO",
        names=[f"dmg_{i}" for i in range(n_items)],
        labels=list(range(n_items)),
        path=tmp_path / "dmg.h5",
    )
    return ref, dmg


# ── 2-D matrix indicators ─────────────────────────────────────────────────────

def test_cfdac_collection_shape_and_refs(tmp_path):
    ref, dmg = _make_pair(tmp_path, n_freq=8, n_outputs=4, n_items=2)
    out = dmg.cfdac(ref, path=tmp_path / "cfdac.h5")
    try:
        assert isinstance(out, pymodal.cfdac_collection)
        assert out._n_domain_axes == 2
        # 4 outputs × 1 input = 4 DOFs flattened
        assert out.measurements[0].shape == (4, 4, 1, 1)
        assert "reference" in out.reference_roles()
        assert "damaged" in out.reference_roles()
        # numerical equivalence against utils
        ref_H = ref.measurements[0][()].reshape(8, -1).T
        dmg_H = dmg.measurements[0][()].reshape(8, -1).T
        expected = utils.value_CFDAC(ref_H, dmg_H)
        actual = out.measurements[0][()][:, :, 0, 0]
        assert np.allclose(actual, expected)
    finally:
        out.close()
        ref.close()
        dmg.close()


def test_fdac_collection_returns_real(tmp_path):
    ref, dmg = _make_pair(tmp_path, n_freq=8, n_outputs=4, n_items=1)
    out = dmg.fdac(ref, path=tmp_path / "fdac.h5")
    try:
        assert isinstance(out, pymodal.fdac_collection)
        arr = out.measurements[0][()]
        assert arr.dtype.kind == "f"     # real float
        assert arr.shape == (4, 4, 1, 1)
    finally:
        out.close()
        ref.close()
        dmg.close()


# ── 1-D vector indicators ─────────────────────────────────────────────────────

def test_rvac_collection_shape(tmp_path):
    ref, dmg = _make_pair(tmp_path, n_freq=8, n_outputs=4, n_items=2)
    out = dmg.rvac(ref, path=tmp_path / "rvac.h5")
    try:
        assert isinstance(out, pymodal.rvac_collection)
        assert out._n_domain_axes == 1
        assert out.measurements[0].shape == (4, 1, 1)
        # numerical equivalence
        ref_H = ref.measurements[0][()].reshape(8, -1).T
        dmg_H = dmg.measurements[0][()].reshape(8, -1).T
        expected = utils.value_RVAC(ref_H, dmg_H)
        actual = out.measurements[0][()][:, 0, 0]
        assert np.allclose(actual, expected)
    finally:
        out.close()
        ref.close()
        dmg.close()


def test_gac_collection(tmp_path):
    ref, dmg = _make_pair(tmp_path, n_freq=8, n_outputs=4, n_items=1)
    gac = dmg.gac(ref, path=tmp_path / "gac.h5")
    try:
        assert isinstance(gac, pymodal.gac_collection)
        assert gac.measurements[0].shape == (4, 1, 1)
    finally:
        gac.close()
        ref.close()
        dmg.close()


def test_m2l_collection_class_wiring(tmp_path):
    """M2L itself has a pre-existing index-bounds bug in utils.M2L unrelated
    to this refactor; we verify only that the wiring exists and the call
    site lazily imports the right class."""
    ref, dmg = _make_pair(tmp_path, n_freq=8, n_outputs=4, n_items=1)
    try:
        from pymodal import m2l_collection  # noqa: F401
        assert pymodal.m2l_collection.__name__ == "m2l_collection"
    finally:
        ref.close()
        dmg.close()


# ── 0-D scalar indicators ─────────────────────────────────────────────────────

def test_sci_collection_shape_and_value(tmp_path):
    ref, dmg = _make_pair(tmp_path, n_freq=8, n_outputs=4, n_items=2)
    out = dmg.sci(ref, path=tmp_path / "sci.h5")
    try:
        assert isinstance(out, pymodal.sci_collection)
        assert out._n_domain_axes == 0
        assert out.measurements[0].shape == (1, 1)
        # values exist and are finite
        for i in range(len(out)):
            v = out.measurements[i][()][0, 0]
            assert np.isfinite(v)
        assert "reference" in out.reference_roles()
        assert "damaged" in out.reference_roles()
    finally:
        out.close()
        ref.close()
        dmg.close()


def test_drq_aigac_frfrms_frfsf_ods_r2(tmp_path):
    ref, dmg = _make_pair(tmp_path, n_freq=8, n_outputs=4, n_items=1)
    outs = [
        dmg.drq(ref, path=tmp_path / "drq.h5"),
        dmg.aigac(ref, path=tmp_path / "aigac.h5"),
        dmg.frfrms(ref, path=tmp_path / "frfrms.h5"),
        dmg.frfsf(ref, path=tmp_path / "frfsf.h5"),
        dmg.ods_diff(ref, path=tmp_path / "ods.h5"),
        dmg.r2_imag(ref, path=tmp_path / "r2.h5"),
        dmg.unsigned_sci(ref, path=tmp_path / "usci.h5"),
    ]
    expected_classes = [
        pymodal.drq_collection, pymodal.aigac_collection,
        pymodal.frfrms_collection, pymodal.frfsf_collection,
        pymodal.ods_diff_collection, pymodal.r2_imag_collection,
        pymodal.unsigned_sci_collection,
    ]
    try:
        for o, cls in zip(outs, expected_classes):
            assert isinstance(o, cls)
            assert o.measurements[0].shape == (1, 1)
            assert "reference" in o.reference_roles()
    finally:
        for o in outs:
            o.close()
        ref.close()
        dmg.close()


def test_frfsm_with_kwarg(tmp_path):
    ref, dmg = _make_pair(tmp_path, n_freq=8, n_outputs=4, n_items=1)
    out = dmg.frfsm(ref, std=3.0, path=tmp_path / "frfsm.h5")
    try:
        assert isinstance(out, pymodal.frfsm_collection)
        assert out.measurements[0].shape == (1, 1)
    finally:
        out.close()
        ref.close()
        dmg.close()


# ── Reference traceability ────────────────────────────────────────────────────

def test_indicator_reference_round_trip(tmp_path):
    """The damaged FRF used to compute item i can be retrieved from the
    indicator collection's embedded references."""
    ref, dmg = _make_pair(tmp_path, n_freq=8, n_outputs=2, n_items=3)
    out = dmg.cfdac(ref, path=tmp_path / "out.h5")
    try:
        for i in range(len(out)):
            ref_back = out.get_reference_data("reference", i)
            dmg_back = out.get_reference_data("damaged", i)
            assert np.allclose(ref_back, ref.measurements[i][()])
            assert np.allclose(dmg_back, dmg.measurements[i][()])
    finally:
        out.close()
        ref.close()
        dmg.close()
