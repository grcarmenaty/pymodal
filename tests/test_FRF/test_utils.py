import numpy as np
import os
from pathlib import Path
import pymodal
import pytest
import scipy.io as sio


def test_unpack_frf():
    frf_baseline = sio.loadmat(os.path.normpath(path))
    frf_info = sio.whosmat(os.path.normpath(path))
    frf_info = frf_info[0]
    frf_baseline = frf_baseline[frf_info[0]]
    frf = pymodal.unpack_FRF_mat(path=(
        Path(__file__).parent.parent / 'data' / 'mat_FRFs' / '00B_Ref_8.MAT'))
    assert (frf == frf_baseline).all()
    assert isinstance(frf, np.ndarray)
    assert frf.shape[0] == 6401
    assert frf.shape[1] == 81
    assert isinstance(frf[0, 0], np.complex64)
