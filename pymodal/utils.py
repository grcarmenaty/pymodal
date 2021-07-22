import json
import numpy as np
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import math
import matplotlib as mpl
from numpy import (
    save as np_save,
    load as np_load,
    savez_compressed,
)
from scipy.io import savemat, loadmat, whosmat
from scipy import sparse
from zipfile import ZipFile
import pymodal
from pymodal import papergraph


def save_array(array: np.ndarray, path: str):

    try:
        if not isinstance(path, pathlib.PurePath):
            path = path.name
    except Exception as __:  # noqa F841
        pass
    path = Path(path)
    file_type = path.suffix
    file_type = file_type.lower()

    if file_type in ".npy":
        with open(path, "wb+") as fh:
            np_save(fh, array, allow_pickle=False)
    elif file_type in ".npz":
        with open(path, "wb+") as fh:
            savez_compressed(fh, data=array)
    elif file_type in ".mat":
        with open(path, "wb+") as fh:
            print(type(fh))
            savemat(fh, {"data": array})
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

    if file_type in ".npy":
        return np_load(path)
    elif file_type in ".npz":
        return np_load(path)["data"]
    elif file_type in ".mat":
        with open(path, "r") as __:  # noqa F841
            array = loadmat(path)
            info = whosmat(path)
            info = info[0]
            return array[info[0]]
    else:
        raise Exception(f"Extension {file_type} not recognized. This function"
                        f" only recognizes .npy, .npz and .mat")

def load_FRF(path: str):

    """
        Load the FRF object from a zip file.

        Parameters
        ----------
        path : pathlib PurePath object
            Path where the object is stored.

        Returns
        -------
        out: pymodal FRF object
            Object loaded from the compressed zip file.
    """

    path = Path(path)
    with ZipFile(path, "r") as fh:
        data_path = Path(fh.extract("data.json"))
        with open("data.json", "r") as z:
            data = json.load(z)
        data_path.unlink()
        frf = []
        for item in data["name"]:
            value_path = Path(fh.extract(f"{item}.npz"))
            with open(f"{item}.npz", "r") as z:
                frf.append(pymodal.load_array(z))
            value_path.unlink()
        frf = np.dstack(frf)
    try:
        return pymodal.FRF(frf=frf,
                        resolution=data["resolution"],
                        bandwidth=data["bandwidth"],
                        max_freq=data["max_freq"],
                        min_freq=data["min_freq"],
                        name=data["name"],
                        part=data["part"],
                        modal_frequencies=data["modal_frequencies"])
    except Exception as __:
        Warning("This is an old file, it will now be imported as the new"
                " format.")
        return pymodal.FRF(frf=frf,
                        resolution=data["resolution"],
                        bandwidth=data["bandwidth"],
                        max_freq=data["max_freq"],
                        min_freq=data["min_freq"],
                        name=data["name"],
                        part=data["part"])

def plot_FRF(frf: np.ndarray,
             freq: np.ndarray,
             ax=None,
             fontsize: float = 12,
             title: str = "Frequency Response",
             title_size: float = 12,
             major_locator: int = 4,
             minor_locator: int = 4,
             fontname: str = "Times New Roman",
             color: str = "blue",
             ylabel: str = "Normalized amplitude ($m·s^{-2}·N^{-1}$)",
             bottom_ylim: float = None,
             decimals_y: int = 1,
             decimals_x: int = 1,
             part: str = "complex"):

    """

    This function plots an FRF into the specified axis, or unto an
    axis of its own device if None is specified.
    """

    if ax is None:  # If this is not a subplot of a greater figure:
        __, ax = plt.subplots()

    if part == "phase":
        if ylabel is None:  # If no label for y axis was specified
            ylabel = "Phase/rad"
        top_ylim = np.amax(frf) + np.pi/4
        if bottom_ylim is None:
            bottom_ylim = -top_ylim
    elif part == "abs" or part == "complex":
        if bottom_ylim is None:  # If no bottom limit is defined
            # Define the bottom limit as four powers of ten lower than average.
            bottom_ylim = 10 ** int(
                np.around(math.log10(np.average(frf) / 10000)))
        # Define the top limit as the minimum integer exponent of ten necessary
        # to fit 1.25 times the maximum value.
        top_ylim = 10 ** int(np.ceil(math.log10(1.25 * np.amax(frf))))
        if ylabel is None:
            # Amplitude is assumed to be acceleration normalized to force input
            ylabel = (r"Amplitude normalized to input/"
                      r"$\mathrm{m·s^{-2}·N^{-1}}$")
    else:
        top_ylim = None
    xlabel = "Frequency/Hz"
    papergraph.lineplot(x=freq,
                        y=frf,
                        ax=ax,
                        fontsize=fontsize,
                        title=title,
                        title_size=title_size,
                        major_x_locator=major_locator,
                        minor_x_locator=minor_locator,
                        fontname=fontname,
                        color=color,
                        ylabel=ylabel,
                        xlabel=xlabel,
                        decimals_y=decimals_y,
                        decimals_x=decimals_x,
                        bottom_ylim=bottom_ylim,
                        top_ylim=top_ylim)
    ax.plot()
    if part == "phase":
        # For y axis: set as many major and minor divisions as specified
        # (4 major and 4 minor inside each major by default) for each unit of
        # pi the data reaches (so going from pi to -pi means 8 divisions).
        ax.yaxis.set_major_locator(
            plt.MultipleLocator(np.pi / major_locator))
        ax.yaxis.set_minor_locator(
            plt.MultipleLocator(np.pi / (major_locator * minor_locator)))
        # Format tick labels to be expressed in terms of fractions of pi.
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(papergraph.multiple_formatter(
                denominator=major_locator)))
    elif part == "abs" or part == "complex":
        ax.set_yscale("log")
        # Put ticks every log decade
        locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ))
        ax.yaxis.set_major_locator(locmaj)
        # Put 0 to 9 minor ticks in every decade
        locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1)
        ax.yaxis.set_minor_locator(locmin)
    # Minor ticks should have no label
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    return ax


def value_CFDAC(ref: np.ndarray, frf: np.ndarray):

    # The following line is the formula of the CFDAC matrix.
    CFDAC_value = np.nan_to_num(
        ((frf @ ref.conj().transpose()) ** 2) * (1/(np.diag(frf @
        frf.conj().transpose()).reshape(-1,1) @ (np.diag(ref @
        ref.conj().transpose()).reshape(-1,1)).conj().transpose()))
    )
    return CFDAC_value

def value_CFDAC_A(ref: np.ndarray, frf: np.ndarray):

    # The following line is the formula of the CFDAC matrix.
    CFDAC_A_value = np.nan_to_num(
        (2*(frf @ ref.conj().transpose())) * (1/(np.diag(frf @
        frf.conj().transpose()).reshape(-1,1) + (np.diag(ref @
        ref.conj().transpose()).reshape(-1,1)).conj().transpose()))
    )
    return CFDAC_A_value


def value_FDAC(ref: np.ndarray, frf: np.ndarray):

    #The following line is the formula of the FDAC matrix.
    FDAC_value = np.nan_to_num(np.abs(
        ((frf @ ref.conj().transpose()) ** 2)) * (1/(np.diag(frf @
        frf.conj().transpose()).reshape(-1,1) @ (np.diag(ref @
        ref.conj().transpose()).reshape(-1,1)).transpose()))
    )
    return FDAC_value

def value_RVAC(ref: np.ndarray, frf: np.ndarray):

    # The following line is the formula of the RVAC vector.
    RVAC_value = np.nan_to_num((np.abs(np.sum(frf * ref.conj(), axis=1)) ** 2) * 
        (1/(np.sum(frf * frf.conj(), axis=1) * (np.sum(ref * ref.conj(), axis=1)))))
    return RVAC_value

def value_RVAC_2d(ref: np.ndarray, frf: np.ndarray):
    ref = np.diff(ref, n=2)
    frf = np.diff(frf, n=2)
    #The following line is the formula of the RVAC''
    #Implementation of the RVAC to make use of FRF curvature
    RVAC_value_2d = np.nan_to_num((np.abs(np.sum(frf * ref.conj(), axis=1)) ** 2) * 
        (1/(np.sum(frf * frf.conj(), axis=1) * (np.sum(ref * ref.conj(), axis=1)))))
    return RVAC_value_2d

def value_GAC(ref: np.ndarray, frf: np.ndarray):

    #The following line is the formula of the GAC vector
    GAC_value = 2(np.abs(np.sum(frf*ref.conj(),axis=1))*
        (1/(np.sum(frf*frf.conj(),axis=1)+np.sum(ref*ref.conj(),axis=1))))
    return GAC_value

def FRFRMS(ref: np.ndarray, frf: np.ndarray):
    #The following line is the formula of the FRFRMS
    num = (np.log10(np.abs(np.sum(frf, axis=1)))-np.log10(np.abs(np.sum(ref,axis=1))))**2
    den = (np.log10(np.abs(np.sum(ref,axis=1))))**2
    FRFRMS_value = np.sqrt(np.sum(num/den))
    return FRFRMS_value

def FRFSF(ref: np.ndarray, frf: np.ndarray):
    FRFSF_value = np.sum(np.sum(np.abs(ref),axis=1))/(np.sum(np.sum(np.abs(frf),axis=1)))
    return FRFSF_value

def FRFSM(ref: np.ndarray, frf: np.ndarray,std):
    #std is a parameter which is usually set at 6dB
    ref = np.abs(np.sum(ref, axis=1))**2
    frf = np.abs(np.sum(frf, axis=1))**2
    ej = np.abs(10*np.log10(ref)-10*np.log10(frf))
    f = 1/(std*np.sqrt(2*np.pi))*np.exp(-(1/2)*((ej-0)/std)**2)
    f0 = 1/(std*np.sqrt(2*np.pi))
    s = 1/len(ref)*np.sum(f)/f0
    return s

def ODS_diff (ref:np.ndarray, frf:np.ndarray):
    sm_value = np.abs(frf-ref)
    sm_value = np.sum(np.sum(sm_value,axis=1))
    return sm_value

def r2_imag (ref:np.ndarray, frf:np.ndarray):
    ref = np.imag(ref).reshape(1,-1)
    frf = np.imag(frf).reshape(1,-1)
    sstot = np.sum((ref - np.mean(ref))**2)
    ssres = np.sum((ref - frf)**2)
    r2_value = 1-ssres/sstot
    return r2_value

# def compress(CFDAC: np.ndarray, diagonal_ratio: float = None,
#     threshold: float = 0.15):

#     CFDAC[np.absolute(CFDAC) < threshold] = 0
#     # After the following two lines only a strip around the main diagonal,
#     # symmetric, remains
#     CFDAC = (np.triu(CFDAC, -CFDAC.shape[0] * diagonal_ratio)
#         if diagonal_ratio is not None else CFDAC)
#     CFDAC = (np.tril(CFDAC, CFDAC.shape[0] * diagonal_ratio)
#         if diagonal_ratio is not None else CFDAC)
#     CFDAC = sparse.csr_matrix(CFDAC)
#     return CFDAC

def SCI(CFDAC_pristine: np.ndarray, CFDAC_altered: np.ndarray):

    PCC = np.corrcoef(CFDAC_pristine.flatten(), CFDAC_altered.flatten())[0,1]
    k = np.sign(np.average(np.tril(CFDAC_altered).flatten()) -
        np.average(np.triu(CFDAC_altered).flatten()))
    SCI_calculation = k * (1-np.absolute(PCC))
    return SCI_calculation

def DRQ(RVAC: np.ndarray):
    DRQ = np.mean(RVAC)
    return DRQ

def AIGAC(GAC:np.ndarray):
    AIGAC = np.mean(GAC)
    return AIGAC

def unsigned_SCI(CFDAC_pristine: np.ndarray, CFDAC_altered: np.ndarray):

    PCC = np.corrcoef(CFDAC_pristine.flatten(), CFDAC_altered.flatten())[0,1]
    SCI_calculation = 1 - np.absolute(PCC)
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


def plot_CFDAC(cfdac: np.ndarray,
         xfreq: float,
         yfreq: float,
         resolution: float,
         ax=None,
         fontname: str = "serif",
         fontsize: float = 12,
         title: str = "CFDAC Matrix",
         title_size: float = 12,
         major_x_locator: int = 4,
         minor_x_locator: int = 4,
         major_y_locator: int = 4,
         minor_y_locator: int = 4,
         color_map: str = "cubehelix",
         xlabel: str = "Frequency/Hz",
         ylabel: str = "Frequency/Hz",
         decimals: int = 0,
         cbar: bool = True,
         pad=0.05):

    xfreq = np.arange(xfreq[0], xfreq[1] + resolution/2, resolution)
    yfreq = np.arange(yfreq[0], yfreq[1] + resolution/2, resolution)
    ax, img = papergraph.imgplot(cfdac,
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
                                 decimals_x=decimals)
    if cbar:
        pymodal.papergraph.add_cbar(ax=ax,
                                    img=img,
                                    data=cfdac,
                                    lower_lim=0,
                                    upper_lim=1,
                                    locator=5,
                                    fontname=fontname,
                                    fontsize=fontsize,
                                    decimals=1,
                                    pad=pad)
    return ax


def damping_coefficient(omega, mass_multiplier, stiffness_multiplier):
    return mass_multiplier/(2 * omega) + (stiffness_multiplier * omega)/2

    
def synthetic_FRF(min_freq, max_freq, resolution, natural_frequencies,
                  damping):
    freq_vector = np.arange(
        min_freq, max_freq + resolution / 2, resolution
    )
    omega = 2 * np.pi * freq_vector
    omega_n = 2 * np.pi * natural_frequencies
    response = 0
    for i in range(len(omega_n)):
        num = 1
        den = (
            1 - (omega/omega_n[i])**2 + 1j *
            (2*damping[i]*(omega / omega_n[i]))
        )
        response = response + num/den
    return response


def modal_superposition(min_freq, max_freq, resolution, modal_frequencies,
                        damping, mode_shapes, mass_matrix, rovings, drivings):
    freq_vector = np.arange(min_freq, max_freq + resolution / 2, resolution)
    omega = freq_vector * 2 * np.pi
    omega_n = modal_frequencies * 2 * np.pi
    damping = damping_coefficient(omega_n, damping[0], damping[1])
    sigma = omega_n * damping
    nf = omega.shape[0]
    modes = omega_n.shape[0]
    eigvals1 = sigma + 1j*omega_n
    eigvals2 = sigma - 1j*omega_n
    modal_mass = np.diag(mode_shapes.T @ mass_matrix @ mode_shapes)
    modal_participation = 1 / (modal_mass*omega_n)
    frf=[]
    for i in range(rovings):
        for j in range(drivings):
            ResMod = (mode_shapes[i, :]*mode_shapes[j, :])*modal_participation
            Res1 = np.tile(omega**2, (modes, 1)) * np.tile(ResMod, (nf, 1)).T
            Res2 = Res1.conj()
            Den1 = np.tile(1j*omega, (modes, 1))-np.tile(eigvals1.T, (nf, 1)).T
            Den2 = np.tile(1j*omega, (modes, 1))-np.tile(eigvals2.T, (nf, 1)).T
            frf.append(np.sum((Res1/Den1) + (Res2/Den2), axis=0) * 10**-3)
    frf = [np.stack(frf, axis=1)]
    frf = np.dstack(frf)
    return pymodal.FRF(frf, resolution = 0.5)