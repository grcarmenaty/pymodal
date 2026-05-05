import numpy as np
import h5py
from pathlib import Path
import time
import warnings
from typing import Optional


def compute_cfdac(H_test, H_ref):
    """Compute CFDAC matrix between test and reference FRF arrays.

    Parameters
    ----------
    H_test : np.ndarray, shape (n_freq, n_dof), complex
    H_ref  : np.ndarray, shape (n_freq, n_dof), complex

    Returns
    -------
    np.ndarray, shape (n_dof, n_dof, 1), float64, values in [0, 1]
    """
    n_dof = H_test.shape[1]
    ref_norms = np.sum(np.abs(H_ref) ** 2, axis=0)
    test_norms = np.sum(np.abs(H_test) ** 2, axis=0)
    cfdac = np.zeros((n_dof, n_dof), dtype=np.float64)
    for i in range(n_dof):
        for j in range(n_dof):
            num = np.abs(np.dot(H_test[:, i], np.conj(H_ref[:, j]))) ** 2
            den = test_norms[i] * ref_norms[j]
            cfdac[i, j] = num / den if den > 1e-30 else 0.0
    return cfdac[:, :, np.newaxis]


class IndicatorCollection:
    """Collection of indicator matrices (e.g. CFDAC) stored in an HDF5 file.

    The on-disk layout matches the format expected by HDF5Dataset so that
    ``torch_dataset()`` works identically to other collection classes:

        measurements/<name>/data   – indicator matrix
        measurements/<name>/label  – scalar float label
        references/<idx>/data      – reference FRF magnitude array
        references/<idx>/name      – reference name string
    """

    def __init__(self, path, matrices, labels, names, references):
        self.path = Path(path)
        if self.path.exists():
            self.path.unlink()

        self.name = list(names)
        self.n_references = len(references)
        self._references = list(references)

        with h5py.File(self.path, "w") as f:
            for mat, name, label in zip(matrices, names, labels):
                f[f"measurements/{name}/data"] = mat
                f[f"measurements/{name}/label"] = float(label)
            for idx, ref in enumerate(references):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    f[f"references/{idx}/data"] = ref.measurements.magnitude
                ref_name = ref.name if ref.name is not None else f"reference_{idx}"
                f[f"references/{idx}/name"] = ref_name

        self.file = h5py.File(self.path, "a")
        self.measurements = [self.file[f"measurements/{n}/data"] for n in self.name]
        self.labels = [self.file[f"measurements/{n}/label"] for n in self.name]

    def __len__(self):
        return len(self.name)

    def get_reference(self, idx):
        return self._references[idx]

    def torch_dataset(self):
        from pymodal import HDF5Dataset
        self.file.close()
        self.dataset = HDF5Dataset(self.path)
        return self

    def close(self, keep=False):
        self.file.close()
        if not keep:
            self.path.unlink()
