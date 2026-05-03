import json
import math
import pathlib
from decimal import Decimal
from pathlib import Path
from typing import Optional
from warnings import warn
from zipfile import ZipFile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy import (
    load as np_load,
    save as np_save,
    savez_compressed,
)
from pint import UnitRegistry, Quantity
from scipy import sparse  # noqa F401
from scipy.io import savemat, loadmat, whosmat


# If a function will only be used internally in the utils, use __ before its name, if
# it is intended for use in other modules but not by an end user, use _ before its name
# instead. Don't use preceding _ in any combination except in those cases.


def __check_domain_measurements_pair(
    domain_array: npt.NDArray[np.float64],
    measurements_array: npt.NDArray[np.complex64],
):
    """An auxiliary function that makes sure that a given domain and measurements pair
    are adequately typed, formatted and matched.

    Parameters
    ----------
    domain_array : numpy array of floats
        An array containing the temporal dimension, which measures the rate of physical
        change, be it by using domain, frequency or any other suitable quantity.
    measurements_array : numpy array of complex
        An array with the measurements of the signal recorded along the domain array.

    Returns
    -------
    A numpy array of floats
        An array containing the new temporal dimension, which measures the rate of
        physical change, be it by using domain, frequency or any other suitable
        quantity.
    A numpy array of complexes
        An array with the measurements of the signal recorded along the domain array,
        with the values corresponding to the values of the domain array.
    """
    if domain_array.shape[0] != measurements_array.shape[0]:
        raise ValueError(
            "Both the domain array and the measurements array should be of the same"
            " length along their first axis."
        )
    if np.all(np.diff(domain_array, axis=0) <= 0):
        raise ValueError(
            "Domain array values should be strictly increasing and non-repeating."
        )
    return domain_array, measurements_array


def change_domain_resolution(
    domain_array: npt.NDArray[np.float64],
    measurements_array: npt.NDArray[np.complex64],
    new_resolution: float,
):
    """Change the temporal resolution of an array of signals, assuming the temporal
    dimension of said signal is the first dimension of the array.

    Parameters
    ----------
    domain_array : numpy array of floats
        An array containing the temporal dimension, which measures the rate of physical
        change, be it by using domain, frequency or any other suitable quantity.
    measurements_array: numpy array of complex
        An array with the measurements of the signal recorded along the domain array.
    new_resolution: float
        The desired distance between any two adjacent values of the domain array.

    Returns
    -------
    A numpy array of floats
        An array containing the new temporal dimension, which measures the rate of
        physical change, be it by using domain, frequency or any other suitable
        quantity; with the new temporal resolution.
    A numpy array of complexes
        An array with the new measurements of the signal recorded along the domain
        array, with the values corresponding to the values of the new domain array.

    """

    domain_array, measurements_array = __check_domain_measurements_pair(
        domain_array, measurements_array
    )
    new_resolution = float(new_resolution)
    new_domain_array = np.arange(
        domain_array[0], domain_array[-1] + new_resolution / 2, new_resolution
    )
    # Determine the amount of decimal places the user desires from the amount
    # of decimals in the desired temporal resolution.
    decimal_places = abs(Decimal(str(new_resolution)).as_tuple().exponent)
    np.around(new_domain_array, decimals=decimal_places)
    # Infer the current temporal resolution from the average of differences between
    # elements in the domain array.
    domain_diff = np.diff(domain_array, axis=0)
    resolution = np.average(domain_diff)
    # Make sure the new max domain value is smaller than the previous max
    # domain value.
    domain_max_value = domain_array[-1]
    if not np.allclose(new_domain_array[-1], domain_max_value):
        if new_domain_array[-1] > domain_max_value:
            new_domain_array = new_domain_array[:-1]
        warn(
            (
                "The resulting max domain will be different to accommodate for the"
                " new temporal resolution."
            ),
            UserWarning,
        )
    else:
        new_domain_array[-1] = domain_max_value

    # Check if new resolution is multiple of old resolution or resolution is
    # not constant
    is_multiple = np.allclose(new_resolution % resolution, 0)
    is_constant = np.allclose(domain_diff, np.ones(domain_diff.shape) * resolution)
    if not is_multiple or not is_constant:
        warn(
            (
                "The resulting signal will be interpolated according to the desired new"
                " temporal resolution."
            ),
            UserWarning,
        )
        # Interpolate the values for each signal according to the new domain
        # vector
        measurements_shape = list(measurements_array.shape)
        measurements_shape[0] = new_domain_array.shape[0]
        measurements_shape = tuple(measurements_shape)
        flat_measurements = measurements_array.reshape(
            (measurements_array.shape[0], -1)
        )
        new_measurements_array = np.empty(
            (new_domain_array.shape[0], flat_measurements.shape[-1]),
            dtype=measurements_array.dtype,
        )
        for i in range(flat_measurements.shape[-1]):
            new_measurements_array[:, i] = np.interp(
                new_domain_array, domain_array, flat_measurements[:, i]
            )
        new_measurements_array = new_measurements_array.reshape(measurements_shape)
    else:
        # Keep values corresponding to the new temporal resolution
        step = int(new_resolution / resolution)
        new_measurements_array = measurements_array[0::step, ...]

    return new_domain_array, new_measurements_array


def change_domain_span(
    domain_array: npt.NDArray[np.float64],
    measurements_array: npt.NDArray[np.complex64],
    new_min_domain: Optional[Quantity] = None,
    new_max_domain: Optional[Quantity] = None,
):
    """Change the span of the temporal domain of an array of signals, assuming the
    temporal dimension of said signal is the first dimension of the array.

    Parameters
    ----------
    domain_array : numpy array of floats
        An array containing the temporal dimension, which measures the rate of physical
        change, be it by using domain, frequency or any other suitable quantity.
    measurements_array : numpy array of complex
        An array with the measurements of the signal recorded along the domain array.
    new_min_domain : float, optional
        The desired new minimum value for the domain array, by default None.
    new_max_domain : float, optional
        The desired new maximum value for the domain array, by default None.

    Returns
    -------
    A numpy array of floats
        An array containing the new temporal dimension, which measures the rate of
        physical change, be it by using domain, frequency or any other suitable quantity
        with the new temporal resolution.
    A numpy array of complexes
        An array with the new measurements of the signal recorded along the domain
        array, with the values corresponding to the values of the new domain array.

    """

    domain_array, measurements_array = __check_domain_measurements_pair(
        domain_array, measurements_array
    )
    # Make sure both new max and min domains exist
    if new_min_domain is None:
        new_min_domain = domain_array[0]
    if new_max_domain is None:
        new_max_domain = domain_array[-1]
    # Get current temporal resolution
    domain_diff = np.diff(domain_array, axis=0)
    resolution = np.average(domain_diff)
    # Create copies of inputs to work on them, there are problems if both a new min and
    # max domain values are given otherwise
    new_measurements_array = np.copy(measurements_array)
    new_domain_array = np.copy(domain_array)
    # Add a tail of 0s if max domain is greater than the current max domain
    if new_max_domain > new_domain_array[-1]:
        domain_extension = np.arange(
            new_domain_array[-1],
            new_max_domain + resolution / 2,
            resolution,
        )[1:]
        # Make sure the last domain is coherent with temporal resolution and not
        # greater than the new max domain desired
        if not np.allclose(domain_extension[-1], new_max_domain):
            if domain_extension[-1] > new_max_domain:
                domain_extension = domain_extension[:-1]
            warn("Max domain will be changed to keep sample rate constant", UserWarning)
        new_domain_array = np.hstack((new_domain_array, domain_extension))
        # Add as many measurements points as domain points were created
        measurements_extension_shape = list(new_measurements_array.shape)
        measurements_extension_shape[0] = domain_extension.shape[0]
        measurements_extension = np.zeros(
            tuple(measurements_extension_shape), dtype=measurements_array.dtype
        )
        new_measurements_array = np.concatenate(
            (new_measurements_array, measurements_extension)
        )
    else:
        max_domain_index = (np.abs(new_domain_array - new_max_domain)).argmin()
        new_domain_array = new_domain_array[0 : max_domain_index + 1]
        # Make sure the last domain is coherent with sample rate and not
        # greater than the new max domain desired
        if new_domain_array[-1] != new_max_domain:
            if new_domain_array[-1] > new_max_domain:
                new_domain_array = new_domain_array[:-1]
            warn("Max domain will be changed to keep sample rate constant", UserWarning)
        # Cut the signals to the new max domain
        new_measurements_array = new_measurements_array[0 : max_domain_index + 1, ...]

    # Add a head of 0s to the signals if the new min domain is smaller than
    # the precious min domain
    if new_min_domain < new_domain_array[0]:
        domain_extension = np.arange(
            new_min_domain,
            new_domain_array[0] + resolution / 2,
            resolution,
        )
        # Make sure the domain extension is compatible with the previous domain
        # vector
        if not np.allclose(domain_extension[-1], new_domain_array[0]):
            domain_extension = domain_extension + (
                new_domain_array[0] - domain_extension[-1]
            )
            warn("Min domain will be changed to keep sample rate constant", UserWarning)
        new_domain_array = np.hstack((domain_extension, new_domain_array))
        if new_domain_array[0] < 0:
            new_domain_array = new_domain_array + abs(new_domain_array[0])
        # Add as many measurements points as domain points were created
        measurements_extension_shape = list(new_measurements_array.shape)
        measurements_extension_shape[0] = domain_extension.shape[0]
        measurements_extension = np.zeros(
            tuple(measurements_extension_shape), dtype=measurements_array.dtype
        )
        new_measurements_array = np.concatenate(
            (measurements_extension, new_measurements_array)
        )
    else:
        min_domain_index = (np.abs(new_domain_array - new_min_domain)).argmin()
        new_domain_array = new_domain_array[min_domain_index:]
        # Make sure the new min domain is the closest to the one specified
        # by the user.
        if not np.allclose(new_domain_array[0], new_min_domain):
            warn("Min domain will be changed to keep sample rate constant", UserWarning)
        # Cut the signal from the new min domain
        new_measurements_array = new_measurements_array[min_domain_index:, ...]
    return new_domain_array, new_measurements_array


def lineplot(
    y: np.ndarray,
    x: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    fontname: str = "DejaVu Serif",
    fontsize: float = 12,
    title: str | None = None,
    title_size: float = 12,
    major_y_locator: int = 4,
    minor_y_locator: int = 4,
    major_x_locator: int = 4,
    minor_x_locator: int = 4,
    color: str = "blue",
    linestyle: str = "-",
    ylabel: str | None = None,
    xlabel: str | None = None,
    decimals_y: int = 2,
    decimals_x: int = 2,
    bottom_ylim: float | None = None,
    top_ylim: float | None = None,
    grid: bool = True,
    log: bool = False,
):

    """
    Plot an array of values versus another, preformatted.

    Parameters
    ----------
    y: numpy array of floats
        Vector describing the values along the y axis.
    x: numpy array of floats, optional
        Vector describing the corresponding values along the x axis, default is None.
    ax: axes class, optional
        Axis upon which the line will be plotted. Default is None
    fontsize: float, optional
        Size of the text in the figure, default is 12.
    title: string, optional
        Title for the figure, default is None.
    title_size: float, optional
        Size of the title text, default is 12.
    major_x_locator: int, optional
        How many divisions should there be in the x axis, default is 4.
    minor_x_locator: int, optional
        How many divisions should there be in each major division in the x axis,
        default is 4.
    major_y_locator: int, optional
        How many divisions should there be in the y axis, default is 4.
    minor_y_locator: int, optional
        How many divisions should there be in each major division in the
        y axis, default is 4.
    fontname: string, optional
        Font which will be used across the figure, default is "serif".
    color: string, optional
        What color to use, default is blue.
    xlabel: string, optional
        Label for the x axis, default is None.
    ylabel: string, optional
        Label for the y axis, default is None.
    bottom_ylim: float, optional
        Smallest plotted value, if unspecified will be set to the minimum value of y
        minus 12.5% of the minimum to maximum value of the data, default is None.
    top_ylim: float, optional
        Greatest plotted value, if unspecified will be set to the maximum value of y
        plus 12.5% of the minimum to maximum value of the data, default is None.
    Returns
    -------
    out: AxesSubplot class
        Line plot.
    Notes
    -----
    x and y should have the same length.
    """

    # Define font for mathematical text.
    mpl.rcParams["mathtext.fontset"] = "custom"
    mpl.rcParams["mathtext.rm"] = fontname
    mpl.rcParams["mathtext.it"] = fontname + ":italic"
    mpl.rcParams["mathtext.bf"] = fontname + ":bold"

    ureg = UnitRegistry()
    ureg.setup_matplotlib(True)

    if not isinstance(y, Quantity):
        y = y * ureg("")
    if x is None:
        x = np.arange(y.shape[0])
    if not isinstance(x, Quantity):
        x = x * ureg("")
    if ax is None:  # If this is not a subplot of a greater figure:
        __, ax = plt.subplots()
    # Set limits for x axis between the minimum and maximum domain array values.
    ax.set_xlim(left=x.m[0], right=x.m[-1])
    # Calculate major and minor ticks locators and labels for the x axis
    x_span = x.m[-1] - x.m[0]
    x_step = x_span / major_x_locator
    x_ticks_labels = np.arange(x.m[0], x.m[-1] + x_step / 2, x_step)
    ax.set_xticks([tick for tick in x_ticks_labels])
    ax.set_xticklabels([f"{label:.{decimals_x}f}" for label in x_ticks_labels])
    for label in ax.get_xticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)
    x_minor_step = x_span / (major_x_locator * minor_x_locator)
    x_minor_ticks = np.arange(x.m[0], x.m[-1] + x_minor_step / 2, x_minor_step)
    ax.set_xticks([tick for tick in x_minor_ticks], minor=True)
    if log:  # If y is logarithmic, y ticks and labels are straightforward
        ax.set_yscale("log")
        top = np.nanmax(y.m)
        bottom = np.nanmin(y.m)
        top_ylim = 10 ** np.ceil(np.log10(top))
        bottom_ylim = 10 ** np.floor(np.log10(bottom))
        ax.set_ylim(top=top_ylim, bottom=bottom_ylim)
    else:
        # Set limits for y axis between the minimum and maximum data values with 12.5%
        # margin unless otherwise specified.
        if bottom_ylim is None or top_ylim is None:
            top = np.nanmax(y.m)
            bottom = np.nanmin(y.m)
            span = np.abs(top - bottom)
            bottom_ylim = bottom - 0.125 * span if bottom_ylim is None else bottom_ylim
            top_ylim = top + 0.125 * span if top_ylim is None else top_ylim
        ax.set_ylim(top=top_ylim, bottom=bottom_ylim)
        # Calculate major and minor ticks locators and labels for the y axis
        y_span = top_ylim - bottom_ylim
        y_step = y_span / major_y_locator
        y_ticks_labels = np.arange(bottom_ylim, top_ylim + y_step / 2, y_step)
        ax.set_yticks([tick for tick in y_ticks_labels])
        ax.set_yticklabels([f"{label:.{decimals_y}f}" for label in y_ticks_labels])
        y_minor_step = y_span / (major_y_locator * minor_y_locator)
        y_minor_ticks = np.arange(
            bottom_ylim, top_ylim + y_minor_step / 2, y_minor_step
        )
        ax.set_yticks([tick for tick in y_minor_ticks], minor=True)
    # Set labels and titles
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontname=fontname, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontname=fontname, fontsize=fontsize)
    if title is not None:  # If there is a title text (by default there is)
        ax.set_title(title, pad=15, fontname=fontname, fontsize=title_size)
    # Add ticks with labels on both sides of both axes but only on the down and
    # left parts, looking in.
    ax.tick_params(
        axis="both",
        pad=10,
        direction="in",
        which="both",
        bottom=True,
        top=True,
        left=True,
        right=True,
        labelbottom=True,
        labeltop=False,
        labelleft=True,
        labelright=False,
    )
    if grid:
        ax.grid(color="grey", linestyle=":", linewidth=1)
    img = ax.plot(x.m, y.m, color=color, linewidth=0.5, linestyle=linestyle)
    for label in ax.get_yticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)
    plt.tight_layout()
    return ax, img


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

    import pymodal

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

    from pymodal import papergraph

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
    GAC_value = 2*(np.abs(np.sum(frf*ref.conj(),axis=1))*
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


def FRFSM(ref: np.ndarray, frf: np.ndarray, std):
    #std is a parameter which is usually set at 6dB
    ref = np.abs(np.sum(ref, axis=1))**2
    frf = np.abs(np.sum(frf, axis=1))**2
    ej = np.abs(10*np.log10(ref)-10*np.log10(frf))
    f = 1/(std*np.sqrt(2*np.pi))*np.exp(-(1/2)*((ej-0)/std)**2)
    f0 = 1/(std*np.sqrt(2*np.pi))
    s = 1/len(ref)*np.sum(f)/f0
    return s


def ODS_diff(ref: np.ndarray, frf: np.ndarray):
    sm_value = np.abs(frf-ref)
    sm_value = np.sum(np.sum(sm_value,axis=1))
    return sm_value


def r2_imag(ref: np.ndarray, frf: np.ndarray):
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
    DRQ = np.real(np.mean(RVAC))
    return DRQ


def AIGAC(GAC: np.ndarray):
    AIGAC = np.real(np.mean(GAC))
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

    import pymodal

    xfreq = np.arange(xfreq[0], xfreq[1] + resolution/2, resolution)
    yfreq = np.arange(yfreq[0], yfreq[1] + resolution/2, resolution)
    ax, img = pymodal.papergraph.imgplot(cfdac,
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
    import pymodal

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
    return pymodal.FRF(frf, resolution=0.5)


def plot_control_chart(di_array: np.ndarray,
                       di_train: np.ndarray,
                       threshold: int,
                       colors: list,
                       method_name: str,
                       n_pcs: int):

    """
    This function plots a Control Chart of the measurements.
    Dividing between training and test, and classifying among:
    Undamaged, False Alarm, Unnoticed Damage and Damaged conditions.
    """

    plt.figure("Control Chart: "+str(method_name)+" Number of PCs: "+str(n_pcs))
    ax = plt.gca()
    ax.scatter(range(len(di_array)),di_array,c=colors,s=5)
    ax.set_yscale("log")
    ax.set_xlabel("Measurements")
    ax.set_ylabel("Mahalanobis Distance")
    ax.set_ylim(0, np.max(di_array)*1.2)

    #Plot Threshold Line
    ax.plot(range(len(di_array)),np.ones((len(di_array),))*threshold,linewidth=1,color='black')

    #Plot Division Line between Training and Test data
    ax.plot(np.ones((2,))*len(di_train),[0,np.max(di_array)*1000],
                linewidth=1,
                linestyle='--',
                color='black')

    #Custom legend
    legend_elements = [mpl.lines.Line2D([0], [0], color='black', lw=1, label='Threshold'),
                                    mpl.lines.Line2D([0],[0], color='black',linestyle='--',  lw=1, label='Training-Test Split'),
                                    mpl.lines.Line2D([0],[0],color='w',markerfacecolor='green',marker='o', label='Undamaged'),
                                    mpl.lines.Line2D([0],[0],color='w', markerfacecolor='orange',marker='o', label='False Alarm'),
                                    mpl.lines.Line2D([0],[0],color='w', markerfacecolor='blue', marker='o', label='Unnoticed Damage'),
                                    mpl.lines.Line2D([0],[0],color='w', markerfacecolor='red',marker='o', label='Damaged')]

    ax.legend(handles=legend_elements,bbox_to_anchor=(0.5,1.2), loc='upper center',ncol=6,fancybox=True)

    plt.figure("Data Visualization")
    plt.show()
