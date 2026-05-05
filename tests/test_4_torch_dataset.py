"""Phase-4 tests: HDF5Dataset round-trips for variable-rank items.

The dimensional-collection refactor introduces an ``_axes`` group inside
``/measurements/`` and items of variable rank (rank 2 for 0-D collections,
rank 3 for 1-D, rank 4 for 2-D). HDF5Dataset must skip ``_axes`` and pass
items through unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import pymodal


def test_hdf5_dataset_skips_axes_group(tmp_path):
    """The _axes group is collection metadata, not an item — must not appear
    in the dataset's item list."""
    f = pymodal.frf(
        measurements=[np.random.rand(8, 1, 1) for _ in range(3)],
        freq_resolution=1.0,
        method="MISO",
        names=["a", "b", "c"],
        labels=[0, 1, 0],
        path=tmp_path / "f.h5",
    )
    f.torch_dataset()
    ds = f.dataset
    assert len(ds) == 3
    names = {info["name"] for info in ds.get_data_infos("data")}
    assert names == {"a", "b", "c"}
    assert "_axes" not in names


def test_torch_dataset_1d_round_trip(tmp_path):
    arrs = [np.random.rand(16, 2, 1).astype(np.float32) for _ in range(4)]
    f = pymodal.frf(
        measurements=arrs, freq_resolution=1.0, method="SIMO",
        labels=[0, 1, 0, 1], path=tmp_path / "f.h5",
    )
    f.torch_dataset()
    ds = f.dataset
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (2, 16, 2, 1)
    assert batch_y.shape == (2,)


def test_torch_dataset_2d_round_trip(tmp_path):
    """A 2-D indicator collection must produce rank-4 items via the dataset."""
    rng = np.random.default_rng(0)
    n_freq, n_outputs, n_items = 8, 3, 2
    ref_arrs = [
        (rng.standard_normal((n_freq, n_outputs, 1))
         + 1j * rng.standard_normal((n_freq, n_outputs, 1))).astype(np.complex64)
        for _ in range(n_items)
    ]
    dmg_arrs = [a + 0.05 * rng.standard_normal(a.shape).astype(np.complex64) for a in ref_arrs]
    ref = pymodal.frf(measurements=ref_arrs, freq_resolution=1.0, method="SIMO",
                     names=[f"r{i}" for i in range(n_items)], path=tmp_path / "ref.h5")
    dmg = pymodal.frf(measurements=dmg_arrs, freq_resolution=1.0, method="SIMO",
                     names=[f"d{i}" for i in range(n_items)],
                     labels=[0, 1], path=tmp_path / "dmg.h5")
    cfdac = dmg.cfdac(ref, path=tmp_path / "cfdac.h5")
    cfdac.torch_dataset()
    ds = cfdac.dataset
    info = ds.get_data_infos("data")[0]
    # rank-4: (n_dof, n_dof, 1, 1) where n_dof = n_outputs * n_inputs = 3
    assert info["shape"] == (3, 3, 1, 1)
    arr = ds.get_data("data", 0)
    assert arr.shape == (3, 3, 1, 1)
    ref.close()
    dmg.close()


def test_torch_dataset_0d_round_trip(tmp_path):
    """A 0-D scalar indicator collection must produce rank-2 items."""
    rng = np.random.default_rng(1)
    arrs = [
        (rng.standard_normal((8, 2, 1))
         + 1j * rng.standard_normal((8, 2, 1))).astype(np.complex64)
        for _ in range(3)
    ]
    dmg_arrs = [a + 0.05 * rng.standard_normal(a.shape).astype(np.complex64) for a in arrs]
    ref = pymodal.frf(measurements=arrs, freq_resolution=1.0, method="SIMO",
                     names=[f"r{i}" for i in range(3)], path=tmp_path / "ref.h5")
    dmg = pymodal.frf(measurements=dmg_arrs, freq_resolution=1.0, method="SIMO",
                     names=[f"d{i}" for i in range(3)], labels=[0, 1, 0],
                     path=tmp_path / "dmg.h5")
    sci = dmg.sci(ref, path=tmp_path / "sci.h5")
    sci.torch_dataset()
    ds = sci.dataset
    info = ds.get_data_infos("data")[0]
    assert info["shape"] == (1, 1)
    ref.close()
    dmg.close()
