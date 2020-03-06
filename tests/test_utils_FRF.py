import numpy as np
from numpy.core import defchararray
from pathlib import Path
import pymodal
import scipy.io as sio

# Build a list of all FRF.mat files in the data folder. All of the used FRF
# files have 81 lines of 0Hz to 3200Hz bands with a resolution of 0.5Hz, which
# implies the shape of the arrays is (6401, 81)
path = (Path(__file__).parent / 'data' / 'FRF').glob('**/*')
files = [file for file in path if file.is_file()]
length = len(files)
array_list = [pymodal.load_array(file) for file in files]


def test_load_frf():
    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5)
    array_list_decimals = [np.around(item, 4) for item in array_list]
    frf_decimals = pymodal.frf.FRF(frf=array_list_decimals, resolution=0.5)
    file_path = Path(__file__).parent / 'data' / 'save' / 'FRF' / 'test.zip'
    decimals_file_path = file_path.parent / 'test_decimals.zip'
    assert pymodal.frf.load(file_path) == frf
    assert pymodal.frf.load(decimals_file_path) == frf_decimals
    file_path.unlink()
    decimals_file_path.unlink()
