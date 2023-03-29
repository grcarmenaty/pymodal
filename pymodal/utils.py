import numpy as np
import numpy.typing as npt
from warnings import warn
from decimal import Decimal
from typing import Optional
from pint import UnitRegistry, Quantity
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    new_min_domain: Optional[float] = None,
    new_max_domain: Optional[float] = None,
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
    x: np.ndarray = None,
    ax: plt.Axes = None,
    fontname: str = "serif",
    fontsize: float = 12,
    title: str = None,
    title_size: float = 12,
    major_y_locator: int = 4,
    minor_y_locator: int = 4,
    major_x_locator: int = 4,
    minor_x_locator: int = 4,
    color: str = "blue",
    linestyle: str = "-",
    ylabel: str = None,
    xlabel: str = None,
    decimals_y: int = 0,
    decimals_x: int = 0,
    bottom_ylim: float = None,
    top_ylim: float = None,
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
        minus 25% of its value, default is None.
    top_ylim: float, optional
        Greatest plotted value, if unspecified will be set to the maximum value of y
        plus 25% of its value, default is None.
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
        fig, ax = plt.subplots()
    # Set limits for x axis between the minimum and maximum domain array values.
    ax.set_xlim(left=x.m[0], right=x.m[-1])
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
    if log:
        ax.set_yscale('log')
    else:
        if bottom_ylim is None or top_ylim is None:
            top = np.nanmax(y.m)
            bottom = np.nanmin(y.m)
            span = np.abs(top - bottom)
            bottom_ylim = bottom - 0.125 * span if bottom_ylim is None else bottom_ylim
            top_ylim = top + 0.125 * span if top_ylim is None else top_ylim
        ax.set_ylim(top=top_ylim, bottom=bottom_ylim)
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
    return img, ax


if __name__ == "__main__":

    # Create a sinusoidal signal from a domain array
    domain_array = np.arange(0, 2 * np.pi + np.pi / 200, np.pi / 100)
    measurements_array = np.sin(domain_array)
    # Change the resolution to half the original resolution to test
    # change_domain_resolution
    new_domain_array, new_measurements_array = change_domain_resolution(
        domain_array=domain_array,
        measurements_array=measurements_array,
        new_resolution=np.pi / 10,
    )
    # You should be able to see the difference in the peaks.
    img, ax = lineplot(x=domain_array, y=measurements_array)
    lineplot(x=new_domain_array, y=new_measurements_array, color="red", ax=ax)
    plt.show()

    # Use the previously created sinusoidal signal and domain and change_domain_span
    # first to extend it by a period after and before
    extended_domain_array, extended_measurements_array = change_domain_span(
        domain_array=domain_array,
        measurements_array=measurements_array,
        new_min_domain=-2 * np.pi,
        new_max_domain=4 * np.pi,
    )
    # then to cut it to the middle half of the period
    cut_domain_array, cut_measurements_array = change_domain_span(
        domain_array=domain_array,
        measurements_array=measurements_array,
        new_min_domain=np.pi / 2,
        new_max_domain=3 * np.pi / 2,
    )
    # You should be able to see two overlapping signals and a displaced one
    img, ax = lineplot(x=domain_array, y=measurements_array)
    img, ax = lineplot(
        x=cut_domain_array, y=cut_measurements_array, color="green", ax=ax
    )
    lineplot(x=extended_domain_array, y=extended_measurements_array, color="red", ax=ax)
    plt.show()
