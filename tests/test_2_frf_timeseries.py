"""Phase-2 tests: frf and timeseries as subclasses of _collection_1d.

Covers single-array and list-of-arrays construction, frequency/time aliases,
domain-edition aliases, and timeseries.to_FRF reference registration.
"""

from __future__ import annotations

import numpy as np
import pytest

import pymodal


# ── frf ───────────────────────────────────────────────────────────────────────

def test_frf_single_array_constructor(tmp_path):
    arr = (np.random.rand(16, 2, 1) + 1j * np.random.rand(16, 2, 1)).astype(np.complex64)
    f = pymodal.frf(
        measurements=arr, freq_resolution=0.5, name="single",
        path=tmp_path / "f.h5",
    )
    try:
        assert len(f) == 1
        assert f.measurements[0].shape == (16, 2, 1)
        assert f.freq_units == "hertz"
        assert f.freq_resolution == pytest.approx(0.5)
        assert f.freq_array.shape == (16,)
        assert f.name == ["single"]
    finally:
        f.close()


def test_frf_list_constructor(tmp_path):
    arrs = [
        (np.random.rand(8, 1, 1) + 1j * np.random.rand(8, 1, 1)).astype(np.complex64)
        for _ in range(3)
    ]
    f = pymodal.frf(
        measurements=arrs, freq_resolution=1.0, method="MISO",
        names=["a", "b", "c"], labels=[0, 1, 2],
        path=tmp_path / "f.h5",
    )
    try:
        assert len(f) == 3
        assert f.measurements[0].shape == (8, 1, 1)
        assert f.name == ["a", "b", "c"]
        assert [lbl[()] for lbl in f.labels] == [0, 1, 2]
    finally:
        f.close()


def test_frf_change_freq_aliases(tmp_path):
    arr = np.random.rand(11, 1, 1)
    f = pymodal.frf(
        measurements=arr, freq_resolution=1.0, method="MISO",
        path=tmp_path / "f.h5",
    )
    try:
        f.change_freq_span(new_min_freq=0.0, new_max_freq=5.0)
        assert f.freq_array[-1] == pytest.approx(5.0)
        f.change_freq_resolution(0.5)
        assert f.freq_resolution == pytest.approx(0.5)
    finally:
        f.close()


def test_frf_dof_and_method(tmp_path):
    arr = np.random.rand(8, 4, 1)  # 4 outputs, 1 input
    f = pymodal.frf(
        measurements=arr, freq_resolution=1.0, method="SIMO",
        path=tmp_path / "f.h5",
    )
    try:
        assert f.n_outputs == 4
        assert f.n_inputs == 1
        assert f.dof == 4
        assert f.coordinates.shape == (4, 3)
        assert f.orientations.shape == (4, 3)
    finally:
        f.close()


def test_frf_excitation_method_rejected(tmp_path):
    with pytest.raises(ValueError):
        pymodal.frf(
            measurements=np.zeros((4, 1, 1)), freq_resolution=1.0,
            method="excitation", path=tmp_path / "f.h5",
        )


# ── timeseries ───────────────────────────────────────────────────────────────

def test_timeseries_single_array_and_aliases(tmp_path):
    sig = np.sin(np.linspace(0, 2 * np.pi, 100))[:, None, None]
    ts = pymodal.timeseries(
        measurements=sig, time_step=0.01, name="sin",
        path=tmp_path / "t.h5",
    )
    try:
        assert len(ts) == 1
        assert ts.time_units == "second"
        assert ts.time_step == pytest.approx(0.01)
        assert ts.sampling_rate == pytest.approx(100.0)
        assert ts.measurements_units == "meter / second ** 2"
    finally:
        ts.close()


def test_timeseries_list_constructor_and_excitation_default_units(tmp_path):
    sigs = [np.random.rand(50, 1, 1) for _ in range(3)]
    exc = pymodal.timeseries(
        measurements=sigs, time_step=0.01, method="excitation",
        path=tmp_path / "t.h5",
    )
    try:
        assert len(exc) == 3
        assert exc.method == "excitation"
        assert exc.measurements_units == "newton"
    finally:
        exc.close()


def test_timeseries_change_aliases(tmp_path):
    sig = np.sin(np.linspace(0, 1, 101))[:, None, None]
    ts = pymodal.timeseries(
        measurements=sig, time_step=0.01,
        path=tmp_path / "t.h5",
    )
    try:
        ts.change_time_span(new_min_time=0.0, new_max_time=0.5)
        assert ts.time_array[-1] == pytest.approx(0.5)
        ts.change_sampling_rate(new_sampling_rate=200.0)
        assert ts.sampling_rate == pytest.approx(200.0, rel=1e-2)
    finally:
        ts.close()


def test_timeseries_to_FRF_registers_references(tmp_path):
    fs = 1024
    duration = 1.0
    n = int(fs * duration)
    t = np.arange(n) / fs
    response = np.sin(2 * np.pi * 50 * t)[:, None, None]
    excitation_arr = np.sin(2 * np.pi * 50 * t)[:, None, None]

    resp = pymodal.timeseries(
        measurements=[response, response],
        time_step=1.0 / fs,
        method="SIMO",
        names=["r0", "r1"],
        labels=[0, 1],
        path=tmp_path / "resp.h5",
    )
    exc = pymodal.timeseries(
        measurements=[excitation_arr, excitation_arr],
        time_step=1.0 / fs,
        method="excitation",
        names=["e0", "e1"],
        path=tmp_path / "exc.h5",
    )
    try:
        f = resp.to_FRF(exc, FRF_type="H1", path=tmp_path / "f.h5")
        try:
            assert isinstance(f, pymodal.frf)
            assert len(f) == 2
            assert "input" in f.reference_roles()
            assert "output" in f.reference_roles()
            # references survive round-trip
            in_arr = f.get_reference_data("input", 0)
            out_arr = f.get_reference_data("output", 0)
            assert np.allclose(in_arr, excitation_arr)
            assert np.allclose(out_arr, response)
            # labels carried through from response
            assert [lbl[()] for lbl in f.labels] == [0, 1]
        finally:
            f.close()
    finally:
        resp.close()
        exc.close()
