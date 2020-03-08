
import numpy as np
import pathlib
from pathlib import Path
from numpy import (
    save as np_save,
    load as np_load,
    savez_compressed,
)
from scipy.io import savemat, loadmat, whosmat


def save_array(array: np.ndarray, path: str):

    try:
        if not isinstance(path, pathlib.PurePath):
            path = path.name
    except Exception as __:  # noqa F841
        pass
    path = Path(path)
    file_type = path.suffix

    if file_type in '.npy':
        with open(path, 'wb+') as fh:
            np_save(fh, array, allow_pickle=False)
    elif file_type in '.npz':
        with open(path, 'wb+') as fh:
            savez_compressed(fh, data=array)
    elif file_type in '.mat':
        with open(path, 'wb+') as fh:
            print(type(fh))
            savemat(fh, {'data': array})
    else:
        raise Exception(f"Extension {file_type} not recognized. This function"
                        f" only recognizes .npy, .npz and .mat")


def load_array(path: str):

    try:
        if not isinstance(path, pathlib.PurePath):
            path = path.name
    except Exception as __:  # noqa F841
        pass

    path = Path(path)
    file_type = path.suffix

    if file_type in '.npy':
        return np_load(path)
    elif file_type in '.npz':
        return np_load(path)['data']
    elif file_type in '.mat':
        with open(path, 'r') as __:  # noqa F841
            array = loadmat(path)
            info = whosmat(path)
            info = info[0]
            return array[info[0]]
    else:
        raise Exception(f"Extension {file_type} not recognized. This function"
                        f" only recognizes .npy, .npz and .mat")
