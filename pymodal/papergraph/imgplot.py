import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pymodal import papergraph


def imgplot(data: np.ndarray,
            y: np.ndarray = None,
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
            color_map: str = 'cubehelix',
            ylabel: str = None,
            xlabel: str = None,
            decimals_y: int = 0,
            decimals_x: int = 0,
            grid: bool = True):

    """
    [UNDER CONSTRUCTION]

    Parameters
    ----------
    data: array
        Matrix to be plotted as image.
    y: array, optional
        Vector describing the y axis. Default is None.
    x: array, optional
        Vector describing the x axis. Default is None.
    ax: axes class, optional
        Axis upon which the image will be plotted. Default is None
    fontname: string, optional
        Font which will be used across the figure. Default is 'serif'.
    fontsize: float, optional
        Size of the text in the figure. Default is 12.
    title: string, optional
        Title for the figure. Default is None.
    title_size: float, optional
        Size of the title text. Default is 12.
    major_y_locator: int, optional
        How many divisions should there be in the y axis. Default is 4.
    minor_y_locator: int, optional
        How many divisions should there be in each major division in the
        y axis. Default is 4.
    major_x_locator: int, optional
        How many divisions should there be in the x axis. Default is 4.
    minor_x_locator: int, optional
        How many divisions should there be in each major division in the
        x axis. Default is 4.
    color_map: str, optional
        What color map to use. Default is viridis.
    ylabel: string, optional
        Label for the y axis. Default is None.
    xlabel: string, optional
        Label for the x axis. Default is None.
    decimals_y: int, optional
        Amount if decimals to show in the tick labels for the y axis.
    decimals_x: int, optional
        Amount if decimals to show in the tick labels for the x axis.
    grid: bool, optional
        Whether to add a grid on top of the plotted image or not.
        Default is True.

    Returns
    -------
    out: AxesSubplot class
        Image plot.

    Notes
    -----
    x and y should correspond in length to the shape of data.
    """

    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = fontname
    mpl.rcParams['mathtext.it'] = fontname + ':italic'
    mpl.rcParams['mathtext.bf'] = fontname + ':bold'
    if x is None:
        x = np.arange(data.shape[1])
    if y is None:
        y = np.arange(data.shape[0])
    x_span = x[-1] - x[0]
    y_span = y[-1] - y[0]
    if ax is None:
        fig, ax = plt.subplots()
    # Set both axes' labels with their corresponding font and size
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontname=fontname, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontname=fontname, fontsize=fontsize)
    ax.set_aspect(aspect='equal')
    # Set font and size for all tick labels in both axes
    x_step = x_span / major_x_locator
    x_ticks_labels = np.arange(x[0] - x_step, x[-1] + 3*x_step/2, x_step)
    ax.set_xticks([tick for tick in x_ticks_labels])
    ax.set_xticklabels([f'{label:.{decimals_x}f}' for label in x_ticks_labels])
    for label in ax.get_xticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)
    x_minor_step = x_span / (major_x_locator*minor_x_locator)
    x_minor_ticks = np.arange(x[0], x[-1] + x_minor_step/2, x_minor_step)
    ax.set_xticks([tick for tick in x_minor_ticks], minor=True)
    y_step = y_span / major_y_locator
    y_ticks_labels = np.arange(y[0] - y_step, y[-1] + 3*y_step/2, y_step)
    ax.set_yticks([tick for tick in y_ticks_labels])
    ax.set_yticklabels([f'{label:.{decimals_y}f}' for label in y_ticks_labels])
    for label in ax.get_yticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)
    y_minor_step = y_span / (major_y_locator*minor_y_locator)
    y_minor_ticks = np.arange(y[0], y[-1] + y_minor_step/2, y_minor_step)
    ax.set_yticks([tick for tick in y_minor_ticks], minor=True)
    if title is not None:
        ax.set_title(title, pad=15, fontname=fontname, fontsize=title_size)
    # Add grey ticks with labels on both sides of both axes, looking in
    ax.tick_params(axis="both", color='dimgray', pad=10, direction='in',
                   which='both', bottom=True, top=True, left=True, right=True,
                   labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False)
    x, y = np.meshgrid(x, y)
    # Plot the CFDAC
    img = ax.pcolormesh(x, y, data, cmap=color_map, shading='auto')
    if grid:
        ax.grid(color='grey', linestyle=':', linewidth=1)
    plt.tight_layout()
    return ax, img
