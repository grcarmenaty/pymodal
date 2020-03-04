import numpy as np
from pathlib import Path
import pymodal
import scipy.io as sio


def test_unpack_frf():
    path = Path(__file__).parent.parent / 'data' / 'mat_FRFs' / '00B_Ref_8.MAT'
    frf_baseline = sio.loadmat(path)
    frf_info = sio.whosmat(path)
    frf_info = frf_info[0]
    frf_baseline = frf_baseline[frf_info[0]]
    frf = pymodal.unpack_FRF_mat(path=path)
    assert (frf == frf_baseline).all()
    assert isinstance(frf, np.ndarray)
    assert frf.shape[0] == 6401
    assert frf.shape[1] == 81
    assert isinstance(frf[0, 0], np.complex64)
