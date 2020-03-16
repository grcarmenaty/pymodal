import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from zipfile import ZipFile
from scipy import sparse

import papergraph
import pymodal


def load(path: str):

    """

    This function loads an FRF class instance from a compressed
    json file
    """

    path = Path(path)
    with ZipFile(path, 'r') as fh:
        data_path = Path(fh.extract('data.json'))
        with open('data.json', 'r') as z:
            data = json.load(z)
        data_path.unlink()
        ref_path = Path(fh.extract('refference.zip'))
        frf_path = Path(fh.extract('frfs.zip'))
        with open('refference.zip', 'r') as z:
            ref = pymodal.frf.load(z)
        with open('frfs.zip', 'r') as z:
            frf = pymodal.frf.load(z)
        ref_path.unlink()
        frf_path.unlink()
        value = []
        for i in range(len(frf)):
            current_value = []
            for j in range(data['divisions']):
                value_row = []
                for k in range(data['divisions']):
                    value_path = Path(fh.extract(f'{i}-{j}-{k}.npz'))
                    with open(f'{i}-{j}-{k}.npz', 'r') as z:
                        if data['compressed']:
                            value_row.append(sparse.load_npz(z))
                        else:
                            value_row.append(pymodal.load_array(z))
                    value_path.unlink()
                current_value.append(value_row)
            value.append(current_value)
        pristine = []
        for i in range(data['divisions']):
            pristine_row = []
            for j in range(data['divisions']):
                pristine_path = Path(fh.extract('pristine.npz'))
                with open('pristine.npz', 'r') as z:
                    if data['compressed']:
                        pristine_row.append(sparse.load_npz(z))
                    else:
                        pristine_row.append(pymodal.load_array(z))
                pristine_path.unlink()
            pristine.append(pristine_row)

    return pymodal.cfdac.CFDAC(ref=ref,
                               frf=frf,
                               resolution=data['resolution'],
                               bandwidth=data['bandwidth'],
                               max_freq=data['max_freq'],
                               min_freq=data['min_freq'],
                               name=data['name'],
                               part=data['part'],
                               compressed=data['compressed'],
                               diagonal_ratio=data['diagonal_ratio'],
                               threshold=data['threshold'],
                               divisions=data['divisions'],
                               _prstine=pristine,
                               _value=value)

def value(ref: np.ndarray, frf: np.ndarray):

    # The following line is the formula of the CFDAC matrix.
    CFDAC_value = np.nan_to_num(
        ((frf @ ref.conj().transpose()) ** 2) * (1/(np.diag(frf @
        frf.conj().transpose()).reshape(-1,1) @ (np.diag(ref @
        ref.conj().transpose()).reshape(-1,1)).conj().transpose()))
    )
    return CFDAC_value


def compress(CFDAC: np.ndarray, diagonal_ratio: float = None,
    threshold: float = 0.15):

    CFDAC[np.absolute(CFDAC) < threshold] = 0
    # After the following two lines only a strip around the main diagonal,
    # symmetric, remains
    CFDAC = (np.triu(CFDAC, -CFDAC.shape[0] * diagonal_ratio)
        if diagonal_ratio is not None else CFDAC)
    CFDAC = (np.tril(CFDAC, CFDAC.shape[0] * diagonal_ratio)
        if diagonal_ratio is not None else CFDAC)
    CFDAC = sparse.csr_matrix(CFDAC)
    return CFDAC


def SCI(CFDAC_pristine: np.ndarray, CFDAC_altered: np.ndarray):

    PCC = np.corrcoef(CFDAC_pristine.flatten(), CFDAC_altered.flatten())[0,1]
    k = np.sign(np.average(np.tril(CFDAC_altered).flatten()) -
        np.average(np.triu(CFDAC_altered).flatten()))
    # ! POSSIBLE IMPROVEMENT: use sum instead of average?
    SCI_calculation = k * np.absolute(1-PCC)
    return SCI_calculation


# Code developped by Joan Fernández Esmerats
def M2L_func(x, i):

    den = (0.95 + 0.05 * i) ** 2
    return (x / den)


def M2L(CFDAC):

    m = int(np.sqrt(np.size(CFDAC)))
    n = int(0.5 * m)
    M2L_value=np.zeros(int(m))
    for x in range (1, n + 1):
        y = int(m - x - 1)
        M2L_value[x] = CFDAC [x, x]
        M2L_value[y] = CFDAC [y, y]
        for i in range (1, x + 1):
            pos_value_inf = (CFDAC [x - i, x] + CFDAC [x - i, x + i] +
                CFDAC [x, x + i])
            pos_value_sup = (CFDAC [y - i, y] + CFDAC [y - i, y + i] +
                CFDAC [y, y + i])
            neg_value_inf = (CFDAC [x, x - i] + CFDAC [x+i,x-i] +
                CFDAC [x + i, x])
            neg_value_sup = (CFDAC [y, y - i] + CFDAC [y + i, y - i] +
                CFDAC [y + i, y])
            unmod_value_inf = pos_value_inf - neg_value_inf
            unmod_value_sup = pos_value_sup- neg_value_sup
            M2L_value[x] = M2L_value[x] + M2L_func(unmod_value_inf, i)
            M2L_value[y] = M2L_value[y] + M2L_func(unmod_value_sup, i)
    M2L_value[0] = CFDAC[0, 0] + M2L_func(CFDAC[0, 1] - CFDAC[1, 0], 1)
    M2L_value[m - 1] = (CFDAC[m - 1, m - 1] + M2L_func(CFDAC[m - 2, m - 1] -
        CFDAC[m - 1, m - 2], 1))
    return M2L_value
# End of code developped by Joan Fernández Esmerats


def plot(cfdac: np.ndarray,
         xfreq: float,
         yfreq: float,
         resolution: float,
         ax=None,
         fontname: str = 'serif',
         fontsize: float = 12,
         title: str = 'CFDAC Matrix',
         title_size: float = 12,
         major_x_locator: int = 4,
         minor_x_locator: int = 4,
         major_y_locator: int = 4,
         minor_y_locator: int = 4,
         color_map: str = 'jet',
         xlabel: str = 'Frequency/Hz',
         ylabel: str = 'Frequency/Hz',
         decimals: int = 0,
         cbar: bool = True,
         cbar_pad: float = 0.2):

    xfreq = np.arange(xfreq[0], xfreq[1] + resolution/2, resolution)
    yfreq = np.arange(yfreq[0], yfreq[1] + resolution/2, resolution)
    ax = papergraph.imgplot(cfdac,
                            x=xfreq,
                            y=yfreq,
                            ax=ax,
                            fontsize=fontsize,
                            title=title,
                            title_size=title_size,
                            major_x_locator=major_x_locator,
                            minor_x_locator=minor_x_locator,
                            major_y_locator=major_y_locator,
                            minor_y_locator=minor_y_locator,
                            fontname=fontname,
                            color_map=color_map,
                            ylabel=ylabel,
                            xlabel=xlabel,
                            decimals_y=decimals,
                            decimals_x=decimals,
                            cbar=cbar,
                            cbar_pad=cbar_pad)
    return ax
