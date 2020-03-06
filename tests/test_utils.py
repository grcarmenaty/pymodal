import numpy as np
import pymodal
from pathlib import Path
from scipy.io import loadmat, whosmat

path = Path(__file__).parent / 'data' / 'FRF' / 'Case0000.mat'
array = loadmat(path)
info = whosmat(path)
info = info[0]
array = array[info[0]]

extensions = ['npy', 'npz', 'mat']
save_path = path.parent.parent / 'save' / 'utils'


def test_save():
    assert isinstance(array, np.ndarray)
    for extension in extensions:
        file_path = save_path / f'test.{extension}'
        pymodal.save_array(array, file_path)
        assert file_path.is_file()


def test_load():
    for extension in extensions:
        file_path = save_path / f'test.{extension}'
        assert np.array_equal(pymodal.load_array(file_path), array)
        file_path.unlink()