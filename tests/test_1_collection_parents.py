"""Phase-1 smoke tests for the dimensionality-typed collection parents.

Covers _collection_0d / _collection_1d / _collection_2d:
- Construction and HDF5 layout
- Item shape contract per dimensionality
- DOF/channel preservation across dimensionalities
- Item selection, append, split
- Domain edition for 1-D and 2-D
- Reference embedding (embed-by-copy default), retrieval, and round-trip
- from_pair_op factories on 0-D and 2-D
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from pymodal.collection_0d import _collection_0d
from pymodal.collection_1d import _collection_1d
from pymodal.collection_2d import _collection_2d


# ── 1-D parent ────────────────────────────────────────────────────────────────

def _make_1d(n_items=3, n_freq=8, n_outputs=2, n_inputs=1, path=None):
    items = [
        (np.random.rand(n_freq, n_outputs, n_inputs)
         + 1j * np.random.rand(n_freq, n_outputs, n_inputs)).astype(np.complex64)
        for _ in range(n_items)
    ]
    return _collection_1d(
        items=items,
        domain_resolution=1.0,
        domain_start=0.0,
        domain_units="hertz",
        method="SIMO" if n_outputs > 1 else "MISO",
        names=[f"sig_{i}" for i in range(n_items)],
        labels=[i for i in range(n_items)],
        path=path,
    )


def test_1d_construction_and_shape(tmp_path):
    coll = _make_1d(path=tmp_path / "c.h5")
    try:
        assert len(coll) == 3
        assert coll.measurements[0].shape == (8, 2, 1)
        assert coll.channel_shape() == (2, 1)
        assert coll.domain_shape() == (8,)
        assert coll._n_domain_axes == 1
        assert coll.domain_array.shape == (8,)
        assert coll.domain_units_str == "hertz"
    finally:
        coll.close()


def test_1d_hdf5_layout(tmp_path):
    coll = _make_1d(path=tmp_path / "c.h5")
    try:
        attrs = coll.file["measurements"].attrs
        assert int(attrs["n_domain_axes"]) == 1
        assert int(attrs["n_outputs"]) == 2
        assert int(attrs["n_inputs"]) == 1
        assert attrs["measurements_units"] == ""
        assert attrs["space_units"] == "millimeter"
        assert "_axes/axis_0" in coll.file["measurements"]
        assert coll.file["measurements/_axes/axis_0"].attrs["units"] == "hertz"
        for n in coll.name:
            assert f"measurements/{n}/data" in coll.file
            assert f"measurements/{n}/label" in coll.file
    finally:
        coll.close()


def test_1d_item_selection(tmp_path):
    coll = _make_1d(path=tmp_path / "c.h5")
    try:
        coll[["sig_0", "sig_2"]]
        assert coll.name == ["sig_0", "sig_2"]
        assert len(coll) == 2
        coll.select_all()
        assert set(coll.name) == {"sig_0", "sig_1", "sig_2"}
        coll[1:]
        assert len(coll) == 2
    finally:
        coll.close()


def test_1d_change_domain_span_and_resolution(tmp_path):
    coll = _make_1d(n_freq=11, path=tmp_path / "c.h5")
    try:
        coll.change_domain_span(new_min=0.0, new_max=5.0)
        assert coll.domain_array[-1] == pytest.approx(5.0)
        assert coll.measurements[0].shape[0] == coll.domain_array.shape[0]
        # change_domain_resolution should resample on the active axis
        coll.change_domain_resolution(new_resolution=0.5)
        assert coll.domain_resolution == pytest.approx(0.5)
    finally:
        coll.close()


def test_1d_append(tmp_path):
    coll = _make_1d(path=tmp_path / "c.h5")
    try:
        new_item = np.random.rand(8, 2, 1).astype(np.complex64)
        coll.append(new_item, name="sig_new", label=99)
        assert len(coll) == 4
        assert coll.name[-1] == "sig_new"
        assert coll.labels[-1][()] == 99
    finally:
        coll.close()


def test_1d_split_stratified(tmp_path):
    items = [np.random.rand(4, 1, 1) for _ in range(10)]
    coll = _collection_1d(
        items=items,
        domain_resolution=1.0,
        method="MISO",
        labels=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        path=tmp_path / "c.h5",
    )
    try:
        tr, va, te = coll.split(0.6, 0.2, 0.2, seed=0)
        assert len(tr) + len(va) + len(te) == 10
        assert sorted(tr + va + te) == list(range(10))
    finally:
        coll.close()


# ── References (embed-by-copy) ────────────────────────────────────────────────

def test_1d_reference_embedding_and_retrieval(tmp_path):
    ref = _make_1d(n_items=2, path=tmp_path / "ref.h5")
    dmg = _collection_1d(
        items=[m[()] for m in ref.measurements],
        domain_array=ref.domain_array.copy(),
        domain_units="hertz",
        method=ref.method,
        names=[f"d_{i}" for i in range(len(ref))],
        labels=[1, 2],
        references={"reference": ref},
        reference_index_maps={"reference": [0, 1]},
        path=tmp_path / "dmg.h5",
    )
    try:
        assert "reference" in dmg.reference_roles()
        # the reference data is now embedded inside dmg.h5
        assert "references/reference/measurements" in dmg.file
        attrs = dmg.file["references/reference"].attrs
        assert attrs["mode"] == "embedded"
        assert list(attrs["index_map"]) == [0, 1]
        assert attrs["collection_class"] == "_collection_1d"
        # round-trip data check
        for i in range(len(dmg)):
            ref_back = dmg.get_reference_data("reference", i)
            assert np.allclose(ref_back, ref.measurements[i][()])
    finally:
        dmg.close()
        ref.close()


def test_reference_survives_source_deletion(tmp_path):
    """Embed-by-copy means the consumer keeps working after the source is gone."""
    ref = _make_1d(n_items=2, path=tmp_path / "ref.h5")
    ref_arrays = [m[()] for m in ref.measurements]
    dmg = _collection_1d(
        items=ref_arrays,
        domain_array=ref.domain_array.copy(),
        domain_units="hertz",
        method=ref.method,
        references={"reference": ref},
        reference_index_maps={"reference": [0, 1]},
        path=tmp_path / "dmg.h5",
    )
    try:
        # nuke the reference's file completely
        ref.close(keep=False)
        # consumer still resolves references from its own file
        for i in range(len(dmg)):
            arr = dmg.get_reference_data("reference", i)
            assert np.allclose(arr, ref_arrays[i])
    finally:
        dmg.close()


# ── 0-D parent ────────────────────────────────────────────────────────────────

def test_0d_construction_preserves_dofs(tmp_path):
    items = [np.array([[0.5], [0.7]]) for _ in range(3)]  # (n_outputs=2, n_inputs=1)
    coll = _collection_0d(
        items=items,
        method="SIMO",
        path=tmp_path / "0d.h5",
    )
    try:
        assert coll._n_domain_axes == 0
        assert coll.measurements[0].shape == (2, 1)
        assert coll.channel_shape() == (2, 1)
        assert coll.domain_shape() == ()
        # _axes group exists but is empty
        assert "measurements/_axes" in coll.file
        assert len(coll.file["measurements/_axes"].keys()) == 0
    finally:
        coll.close()


def test_0d_from_pair_op(tmp_path):
    """SCI-like factory: applying op(ref_i, dmg_i) per item, registering both as references."""
    ref = _make_1d(n_items=2, path=tmp_path / "ref.h5")
    dmg = _make_1d(n_items=2, path=tmp_path / "dmg.h5")

    def mean_abs_diff(r, d):
        return float(np.mean(np.abs(r - d)))

    sci_like = _collection_0d.from_pair_op(
        reference=ref,
        damaged=dmg,
        op=mean_abs_diff,
        path=tmp_path / "sci.h5",
    )
    try:
        assert len(sci_like) == 2
        assert sci_like.measurements[0].shape == (ref.n_outputs, ref.n_inputs)
        assert "reference" in sci_like.reference_roles()
        assert "damaged" in sci_like.reference_roles()
    finally:
        sci_like.close()
        dmg.close()
        ref.close()


# ── 2-D parent ────────────────────────────────────────────────────────────────

def test_2d_construction_preserves_dofs(tmp_path):
    items = [np.random.rand(5, 5, 2, 1) for _ in range(3)]  # (D1, D2, n_out, n_in)
    coll = _collection_2d(
        items=items,
        domain_units=["hertz", "hertz"],
        method="SIMO",
        path=tmp_path / "2d.h5",
    )
    try:
        assert coll._n_domain_axes == 2
        assert coll.measurements[0].shape == (5, 5, 2, 1)
        assert coll.channel_shape() == (2, 1)
        assert coll.domain_shape() == (5, 5)
        assert "measurements/_axes/axis_0" in coll.file
        assert "measurements/_axes/axis_1" in coll.file
    finally:
        coll.close()


def test_2d_change_domain_per_axis(tmp_path):
    items = [np.random.rand(8, 6, 1, 1) for _ in range(2)]
    coll = _collection_2d(
        items=items,
        domain_arrays=[np.linspace(0, 7, 8), np.linspace(0, 5, 6)],
        domain_units=["hertz", "hertz"],
        method="MISO",
        path=tmp_path / "2d.h5",
    )
    try:
        coll.change_domain_span(new_max=3.0, axis=0)
        assert coll.measurements[0].shape[0] == coll.domain_arrays[0].shape[0]
        assert coll.domain_arrays[0][-1] == pytest.approx(3.0)
        coll.change_domain_span(new_max=2.0, axis=1)
        assert coll.measurements[0].shape[1] == coll.domain_arrays[1].shape[0]
    finally:
        coll.close()


def test_2d_from_pair_op(tmp_path):
    """CFDAC-like factory: returns a (D1, D2) matrix per item."""
    ref = _make_1d(n_items=2, n_freq=4, n_outputs=1, n_inputs=1, path=tmp_path / "ref.h5")
    dmg = _make_1d(n_items=2, n_freq=4, n_outputs=1, n_inputs=1, path=tmp_path / "dmg.h5")

    def fake_cfdac(r, d):
        # squeeze channels so we hand back a (D1, D2) matrix for the factory
        r2 = r.reshape(r.shape[0], -1)
        d2 = d.reshape(d.shape[0], -1)
        return d2 @ r2.conj().T

    out = _collection_2d.from_pair_op(
        reference=ref,
        damaged=dmg,
        op=fake_cfdac,
        domain_arrays=[ref.domain_array.copy(), ref.domain_array.copy()],
        domain_units=["hertz", "hertz"],
        path=tmp_path / "cfdac.h5",
    )
    try:
        assert len(out) == 2
        assert out.measurements[0].shape == (4, 4, 1, 1)
        assert "reference" in out.reference_roles()
        assert "damaged" in out.reference_roles()
    finally:
        out.close()
        dmg.close()
        ref.close()


# ── Channel preservation guarantees across dimensionalities ──────────────────

def test_channel_axes_preserved_across_dimensionalities(tmp_path):
    n_out, n_in = 3, 1
    a0 = np.zeros((n_out, n_in))
    a1 = np.zeros((6, n_out, n_in))
    a2 = np.zeros((6, 6, n_out, n_in))

    c0 = _collection_0d([a0], method="SIMO", path=tmp_path / "c0.h5")
    c1 = _collection_1d(
        [a1], domain_resolution=1.0, domain_units="second",
        method="SIMO", path=tmp_path / "c1.h5",
    )
    c2 = _collection_2d(
        [a2], domain_units=["hertz", "hertz"],
        method="SIMO", path=tmp_path / "c2.h5",
    )
    try:
        for c in (c0, c1, c2):
            assert c.channel_shape() == (n_out, n_in)
            assert c.coordinates.shape == (n_out, 3)
            assert c.orientations.shape == (n_out, 3)
    finally:
        c0.close(); c1.close(); c2.close()
