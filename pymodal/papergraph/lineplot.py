import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def lineplot(y: np.ndarray,
             x: np.ndarray = None,
             ax=None,
             fontname: str = 'serif',
             fontsize: float = 12,
             title: str = None,
             title_size: float = 12,
             major_y_locator: int = 4,
             minor_y_locator: int = 4,
             major_x_locator: int = 4,
             minor_x_locator: int = 4,
             color: str = 'blue',
             linestyle: str = '-',
             ylabel: str = None,
             xlabel: str = None,
             decimals_y: int = 0,
             decimals_x: int = 0,
             bottom_ylim: float = None,
             top_ylim: float = None,
             grid: bool = True):

    """
    Plot an array of values versus another. [UNDER CONSTRUCTION]

    Parameters
    ----------
    img: array
        Matrix to be plotted as image.
    y: array
        Vector describing the y axis. Default is None.
    x: array, optional
        Vector describing the x axis. Default is None.
    ax: axes class, optional
        Axis upon which the image will be plotted. Default is None
    fontsize: float, optional
        Size of the text in the figure. Default is 12.
    title: string, optional
        Title for the figure. Default is None.
    title_size: float, optional
        Size of the title text. Default is 12.
    major_x_locator: int, optional
        How many divisions should there be in the x axis. Default is 4.
    minor_x_locator: int, optional
        How many divisions should there be in each major division in the
        x axis. Default is 4.
    major_y_locator: int, optional
        How many divisions should there be in the y axis. Default is 4.
    minor_y_locator: int, optional
        How many divisions should there be in each major division in the
        y axis. Default is 4.
    fontname: string, optional
        Font which will be used across the figure. Default is 'Times
        New Roman'.
    color: string, optional
        What color to use. Default is blue.
    xlabel: string, optional
        Label for the x axis. Default is None.
    ylabel: string, optional
        Label for the y axis. Default is None.
    bottom_ylim: float, optional
        Smallest plotted value.
    top_ylim: float, optional
        Greatest plotted value.

    Returns
    -------
    out: AxesSubplot class
        Line plot.

    Notes
    -----
    x and y should have the same length.
    """

    # Define font for mathematical text.
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = fontname
    mpl.rcParams['mathtext.it'] = fontname + ':italic'
    mpl.rcParams['mathtext.bf'] = fontname + ':bold'
    if x is None:
        x = np.arange(y.shape[0])
    if ax is None:  # If this is not a subplot of a greater figure:
        fig, ax = plt.subplots()
    # Set limits for x axis between the minimum and maximum frequency.
    ax.set_xlim(left=x[0], right=x[-1])
    if bottom_ylim is None:  # If no bottom limit is defined
        # Define the bottom limit as four powers of ten lower than average.
        bottom_ylim = np.amin(y) - 0.25*np.abs(np.amin(y))
    if top_ylim is None:  # If no bottom limit is defined
        # Define the bottom limit as four powers of ten lower than average.
        top_ylim = np.amax(y) + 0.25*np.abs(np.amax(y))
    # Set axis limits as previously defined
    ax.set_ylim(top=top_ylim, bottom=bottom_ylim)
    x_span = x[-1] - x[0]
    y_span = top_ylim - bottom_ylim
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontname=fontname, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontname=fontname, fontsize=fontsize)
    x_step = x_span / major_x_locator
    x_ticks_labels = np.arange(x[0], x[-1] + x_step/2, x_step)
    ax.set_xticks([tick for tick in x_ticks_labels])
    ax.set_xticklabels([f'{label:.{decimals_x}f}' for label in x_ticks_labels])
    for label in ax.get_xticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)
    x_minor_step = x_span / (major_x_locator*minor_x_locator)
    x_minor_ticks = np.arange(x[0], x[-1] + x_minor_step/2, x_minor_step)
    ax.set_xticks([tick for tick in x_minor_ticks], minor=True)
    y_step = y_span / major_y_locator
    y_ticks_labels = np.arange(bottom_ylim, top_ylim + y_step/2, y_step)
    ax.set_yticks([tick for tick in y_ticks_labels])
    ax.set_yticklabels([f'{label:.{decimals_y}f}' for label in y_ticks_labels])
    for label in ax.get_yticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)
    y_minor_step = y_span / (major_y_locator*minor_y_locator)
    y_minor_ticks = np.arange(
        bottom_ylim,
        top_ylim + y_minor_step/2,
        y_minor_step
    )
    ax.set_yticks([tick for tick in y_minor_ticks], minor=True)
    if title is not None:  # If there is a title text (by default there is)
        ax.set_title(title, pad=15, fontname=fontname, fontsize=title_size)
    # Add ticks with labels on both sides of both axes but only on the down and
    # left parts, looking in.
    ax.tick_params(
        axis="both", pad=10, direction='in', which='both', bottom=True,
        top=True, left=True, right=True, labelbottom=True, labeltop=False,
        labelleft=True, labelright=False)
    if grid:
        ax.grid(color='grey', linestyle=':', linewidth=1)
    img = ax.plot(x, y, color=color, linewidth=0.5, linestyle=linestyle)
    plt.tight_layout()
    return img, ax
