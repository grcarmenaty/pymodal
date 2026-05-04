"""Tests for frf, timeseries, collection classes, and HDF5Dataset."""
import h5py
import numpy as np
import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from pymodal import frf, timeseries, frf_collection, timeseries_collection
from pymodal.hdf5_dataset import HDF5Dataset


# ── Helpers ──────────────────────────────────────────────────────────────────

FS = 100.0
DT = 1.0 / FS
N = 512
T = N * DT
DF = 1.0 / T


def make_timeseries(n=N, dt=DT, method="SIMO", name=None):
    t = np.arange(n) * dt
    sig = np.sin(2 * np.pi * 5 * t).reshape(n, 1, 1)
    return timeseries(
        measurements=sig,
        time_step=dt,
        measurements_units="meter / second**2",
        method=method,
        name=name,
    )


def make_frf(n_freq=None, df=DF, name=None):
    if n_freq is None:
        n_freq = N // 2 + 1
    freqs = np.arange(n_freq) * df
    H = 1.0 / (100.0 - (2 * np.pi * freqs) ** 2 + 1j * 0.5 * (2 * np.pi * freqs))
    return frf(
        measurements=H[:, np.newaxis, np.newaxis],
        freq_resolution=df,
        measurements_units="millimeter / second**2 / newton",
        method="SIMO",
        name=name,
    )


# ── timeseries ───────────────────────────────────────────────────────────────

class TestTimeseries:
    def test_time_step_stored_correctly(self):
        ts = make_timeseries()
        assert ts.time_step.magnitude == pytest.approx(DT)

    def test_sampling_rate_is_reciprocal(self):
        ts = make_timeseries()
        assert ts.sampling_rate.magnitude == pytest.approx(FS)

    def test_sampling_rate_reciprocal_of_time_step(self):
        ts = make_timeseries()
        assert ts.sampling_rate.magnitude == pytest.approx(
            1.0 / ts.time_step.magnitude
        )

    def test_change_sampling_rate_takes_hz(self):
        ts = make_timeseries()
        ts_half = ts.change_sampling_rate(new_sampling_rate=FS / 2)
        assert ts_half.sampling_rate.magnitude == pytest.approx(FS / 2, rel=1e-3)
        assert ts_half.time_step.magnitude == pytest.approx(2 * DT, rel=1e-3)
        assert len(ts_half) == pytest.approx(N // 2, abs=2)

    def test_to_frf_frequency_axis(self):
        """to_FRF must produce an FRF whose freq_end ≈ fs/2."""
        ts = make_timeseries()
        exc = make_timeseries(method="excitation")
        result = ts.to_FRF(exc)
        expected_nyquist = FS / 2
        assert result.freq_end.magnitude == pytest.approx(expected_nyquist, rel=0.05)

    def test_to_frf_freq_resolution(self):
        ts = make_timeseries()
        exc = make_timeseries(method="excitation")
        result = ts.to_FRF(exc)
        # freq_resolution = 1 / time_span; time_span = (N-1)*DT, not N*DT
        expected_df = 1.0 / ts.time_span.magnitude
        assert result.freq_resolution.magnitude == pytest.approx(expected_df, rel=1e-3)


# ── frf ──────────────────────────────────────────────────────────────────────

class TestFrf:
    def test_construction(self):
        f = make_frf()
        assert f.measurements.shape[0] == N // 2 + 1
        assert f.dof == 1

    def test_change_freq_span_returns_copy(self):
        f = make_frf()
        original_len = len(f)
        f_trimmed = f.change_freq_span(new_max_freq=20.0)
        assert len(f) == original_len
        assert len(f_trimmed) < original_len

    def test_change_freq_resolution(self):
        f = make_frf()
        f_coarse = f.change_freq_resolution(new_resolution=2 * DF)
        assert f_coarse.freq_resolution.magnitude == pytest.approx(2 * DF, rel=0.05)


# ── frf_collection ───────────────────────────────────────────────────────────

@pytest.fixture
def frf_coll(tmp_path):
    frfs = [make_frf(name=f"sig_{i}") for i in range(4)]
    col = frf_collection(frfs, labels=[float(i % 2) for i in range(4)],
                         path=tmp_path / "test.h5")
    yield col
    col.close()


class TestFrfCollection:
    def test_length(self, frf_coll):
        assert len(frf_coll) == 4

    def test_labels_stored(self, frf_coll):
        vals = [l[()] for l in frf_coll.labels]
        assert vals == [0.0, 1.0, 0.0, 1.0]

    def test_append(self, frf_coll):
        frf_coll.append(make_frf(name="extra"), label=1.0)
        assert len(frf_coll) == 5

    def test_change_freq_span(self, frf_coll):
        original_lines = frf_coll.measurements[0].shape[0]
        frf_coll.change_freq_span(new_max_freq=20.0)
        assert frf_coll.measurements[0].shape[0] < original_lines

    def test_complex64_storage(self, frf_coll):
        arr = frf_coll.measurements[0][()]
        assert arr.dtype == np.complex64

    def test_torch_dataset(self, frf_coll):
        frf_coll.torch_dataset()
        ds = frf_coll.dataset
        assert len(ds) == 4
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)


# ── timeseries_collection ─────────────────────────────────────────────────────

@pytest.fixture
def ts_coll(tmp_path):
    signals = [make_timeseries(name=f"ts_{i}") for i in range(3)]
    col = timeseries_collection(signals, labels=[float(i) for i in range(3)],
                                path=tmp_path / "ts_test.h5")
    yield col
    col.close()


class TestTimeseriesCollection:
    def test_length(self, ts_coll):
        assert len(ts_coll) == 3

    def test_change_sampling_rate(self, ts_coll):
        original_samples = ts_coll.measurements[0].shape[0]
        ts_coll.change_sampling_rate(new_sampling_rate=FS / 2)
        assert ts_coll.measurements[0].shape[0] == pytest.approx(
            original_samples // 2, abs=2
        )
        # sampling_rate on the collection may be a pint Quantity or plain float
        sr = ts_coll.sampling_rate
        sr_val = sr.magnitude if hasattr(sr, 'magnitude') else float(sr)
        assert sr_val == pytest.approx(FS / 2, rel=1e-3)

    def test_add_gaussian_noise(self, ts_coll):
        ts_coll.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.02)
        assert len(ts_coll) == 6


# ── HDF5Dataset ───────────────────────────────────────────────────────────────

@pytest.fixture
def hdf5_ds(tmp_path):
    frfs = [make_frf(name=f"s{i}") for i in range(6)]
    col = frf_collection(frfs, labels=[float(i % 3) for i in range(6)],
                         path=tmp_path / "ds_test.h5")
    col.torch_dataset()
    yield col.dataset


class TestHDF5Dataset:
    def test_len(self, hdf5_ds):
        assert len(hdf5_ds) == 6

    def test_getitem_returns_tensors(self, hdf5_ds):
        x, y = hdf5_ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_labels_correct(self, hdf5_ds):
        labels = [float(hdf5_ds[i][1]) for i in range(6)]
        assert labels == [float(i % 3) for i in range(6)]

    def test_transform_applied(self, hdf5_ds):
        hdf5_ds.transform = lambda x: torch.from_numpy(np.abs(x).astype(np.float32))
        x, _ = hdf5_ds[0]
        assert isinstance(x, torch.Tensor)
        assert x.dtype == torch.float32

    def test_dataloader_num_workers(self, hdf5_ds):
        hdf5_ds.transform = lambda x: torch.from_numpy(
            np.abs(x).astype(np.float32).reshape(-1)
        )
        loader = DataLoader(hdf5_ds, batch_size=2, num_workers=2)
        batch = next(iter(loader))
        assert batch[0].shape[0] == 2

    def test_pad_collate_fn_same_length(self, hdf5_ds):
        hdf5_ds.transform = lambda x: torch.from_numpy(
            np.abs(x).astype(np.float32).reshape(1, -1)
        )
        loader = DataLoader(hdf5_ds, batch_size=3,
                            collate_fn=HDF5Dataset.pad_collate_fn)
        xs, ys = next(iter(loader))
        assert xs.shape[0] == 3

    def test_pad_collate_fn_variable_length(self, tmp_path):
        # Build an HDF5 file manually with two groups of different length.
        h5_path = str(tmp_path / "var.h5")
        with h5py.File(h5_path, "w") as f:
            meas = f.create_group("measurements")
            for i, length in enumerate([50, 50, 80, 80]):
                g = meas.create_group(f"sig_{i}")
                g["data"] = np.ones((length, 1, 1), dtype=np.complex64)
                g["label"] = float(i % 2)

        ds = HDF5Dataset(h5_path)
        ds.transform = lambda x: torch.from_numpy(
            np.abs(x).astype(np.float32).reshape(1, -1)
        )
        loader = DataLoader(ds, batch_size=4,
                            collate_fn=HDF5Dataset.pad_collate_fn)
        xs, ys = next(iter(loader))
        assert xs.shape == (4, 1, 80)
        assert xs[0, 0, 50:].sum() == 0   # short samples zero-padded
