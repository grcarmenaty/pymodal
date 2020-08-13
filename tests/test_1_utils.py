import numpy as np
import pymodal
import pathlib
from scipy.io import loadmat, whosmat
import matplotlib.pyplot as plt

path = pathlib.Path(__file__).parent / 'data' / 'FRF' / 'Case0000.mat'
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

# Build a list of all FRF.mat files in the data folder. All of the used FRF
# files have 81 lines of 0Hz to 3200Hz bands with a resolution of 0.5Hz, which
# implies the shape of the arrays is (6401, 81)
path = (pathlib.Path(__file__).parent / 'data' / 'FRF').glob('**/*')
files = [file for file in path if file.is_file()]
length = len(files)
array_3d = [pymodal.load_array(file) for file in files]
array_3d = np.dstack(array_3d)


def test_load_frf():
    frf = pymodal.FRF(frf=array_3d, resolution=0.5)
    array_list_decimals = np.around(array_3d, 4)
    frf_decimals = pymodal.FRF(frf=array_list_decimals, resolution=0.5)
    file_path = pathlib.Path(__file__).parent / 'data' / 'save' / 'FRF' / 'test.zip'
    decimals_file_path = file_path.parent / 'test_decimals.zip'
    assert pymodal.load_FRF(file_path) == frf
    assert pymodal.load_FRF(decimals_file_path) == frf_decimals
    file_path.unlink()
    decimals_file_path.unlink()


def test_value_cfdac():
    ref = array_3d[:, :, 0]
    frf = array_3d[:, :, 1]
    cfdac_function = pymodal.value_CFDAC(ref, frf)
    cfdac = np.nan_to_num(
        ((frf @ ref.conj().transpose()) ** 2) * (1/(np.diag(frf @
        frf.conj().transpose()).reshape(-1,1) @ (np.diag(ref @
        ref.conj().transpose()).reshape(-1,1)).conj().transpose()))
    )
    assert np.array_equal(cfdac_function, cfdac)


def test_SCI():
    ref = array_3d[:, :, 0]
    frf = array_3d[:, :, 1]
    CFDAC_pristine = np.abs(pymodal.value_CFDAC(ref, ref))
    CFDAC_altered = np.abs(pymodal.value_CFDAC(ref, frf))
    PCC = np.corrcoef(CFDAC_pristine.flatten(), CFDAC_altered.flatten())[0,1]
    k = np.sign(np.average(np.tril(CFDAC_altered).flatten()) -
        np.average(np.triu(CFDAC_altered).flatten()))
    SCI = k * (1-np.absolute(PCC))
    SCI_funcion = pymodal.SCI(CFDAC_pristine, CFDAC_altered)
    assert SCI_funcion == SCI


def test_plot_cfdac():
    ref = array_3d[:, :, 0]
    cfdac = pymodal.value_CFDAC(ref, ref)
    pymodal.plot_CFDAC(np.abs(cfdac.real),
                       [0, 3200],
                       [0, 3200],
                       0.5)
    plt.tight_layout()
    save_dir = pathlib.Path(__file__).parent
    save_dir = save_dir / 'data' / 'save' / 'result_images' / 'cfdac_plot'
    file_path = save_dir / 'cfdac.png'
    plt.savefig(file_path)
    plt.close()
    assert file_path.is_file()