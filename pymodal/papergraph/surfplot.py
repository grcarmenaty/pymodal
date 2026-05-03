import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pymodal import papergraph
from mpl_toolkits.mplot3d import Axes3D


def surfplot(data: np.ndarray,
             colordata: np.ndarray = None,
             y: np.ndarray = None,
             x: np.ndarray = None,
             ax=None,
             fontname: str = 'serif',
             fontsize: float = 12,
             title: str = None,
             title_size: float = 12,
             major_z_locator: int = 4,
             major_y_locator: int = 4,
             major_x_locator: int = 4,
             color_map: str = 'viridis',
             zlabel: str = None,
             ylabel: str = None,
             xlabel: str = None,
             decimals_z: int = 0,
             decimals_y: int = 0,
             decimals_x: int = 0,
             bottom_zlim: float = None,
             top_zlim: float = None,
             cbar: bool = True,
             cbar_locator: int = 5,
             cbar_pad: float = 0.2,
             cbar_label: str = None,
             decimals_cbar: int = 1,
             bottom_cbar_lim: float = None,
             top_cbar_lim: float = None):

    """
    Plot surface graph. [UNDER CONSTRUCTION]

    Parameters
    ----------
    data: array
        Matrix to be plotted.
    colordata: array, optional
        Matrix with color values.
    x: array, optional
        Vector describing the x axis. Default is None.
    y: array, optional
        Vector describing the y axis. Default is None.
    ax: axes class, optional
        Axis upon which the image will be plotted. Default is None
    fontname: string, optional
        Font which will be used across the figure. Default is 'Times
        New Roman'.
    fontsize: float, optional
        Size of the text in the figure. Default is 12.
    title: string, optional
        Title for the figure. Default is None.
    title_size: float, optional
        Size of the title text. Default is 12.
    major_x_locator: int, optional
        How many divisions should there be in the x axis. Default is 4.
    major_y_locator: int, optional
        How many divisions should there be in the y axis. Default is 4.
    major_z_locator: int, optional
        How many divisions should there be in the z axis. Default is 4.
    aspect_ratio: float, optional
        Enforce aspect ratio for the image. Default is None.
    cbar: bool, optional
        Whether to add a color bar to the figure.
    color_map: str, optional
        What color map to use. Default is viridis.
    cbar_pad: float, optional
        Distance between the color bar and the image. Default is 0.2.
    xlabel: string, optional
        Label for the x axis. Default is None.
    ylabel: string, optional
        Label for the y axis. Default is None.
    zlabel: string, optional
        Label for the z axis. Default is None.        

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
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
    if bottom_zlim is None:  # If no bottom limit is defined
        # Define the bottom limit as four powers of ten lower than average.
        bottom_zlim = np.amin(data) - 0.25*np.abs(np.amin(data))
        if bottom_zlim == np.amin(data):
            bottom_zlim = bottom_zlim - 0.25*np.abs(np.amax(data))
    if top_zlim is None:  # If no bottom limit is defined
        # Define the bottom limit as four powers of ten lower than average.
        top_zlim = np.amax(data) + 0.25*np.abs(np.amax(data))
        if top_zlim == np.amax(data):
            top_zlim = top_zlim - 0.25*np.abs(np.amin(data))
    # Set axis limits as previously defined
    x_span = x[-1] - x[0]
    y_span = y[-1] - y[0]
    z_span = top_zlim - bottom_zlim
    # Set both axes' labels with their corresponding font and size
    labelpad = fontsize * 1.2
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontname=fontname, fontsize=fontsize,
                      labelpad=labelpad + decimals_x)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontname=fontname, fontsize=fontsize,
                      labelpad=labelpad + decimals_y)
    if zlabel is not None:
        ax.set_zlabel(zlabel, fontname=fontname, fontsize=fontsize,
                      labelpad=labelpad + decimals_z)
    # Set font and size for all tick labels in both axes
    x_step = x_span / major_x_locator
    x_ticks_labels = np.arange(x[0] - x_step, x[-1] + 3*x_step/2, x_step)
    ax.set_xticks([tick for tick in x_ticks_labels])
    ax.set_xticklabels([f'{label:.{decimals_x}f}' for label in x_ticks_labels])
    for label in ax.get_xticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)
    ax.set_xlim(left=x[0], right=x[-1])

    y_step = y_span / major_y_locator
    y_ticks_labels = np.arange(y[0] - y_step, y[-1] + 3*y_step/2, y_step)
    ax.set_yticks([tick for tick in y_ticks_labels])
    ax.set_yticklabels([f'{label:.{decimals_y}f}' for label in y_ticks_labels])
    for label in ax.get_yticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)
    ax.set_ylim(bottom=y[0], top=y[-1])
    
    z_step = z_span / major_z_locator
    z_ticks_labels = np.arange(bottom_zlim, top_zlim + z_step/2, z_step)
    ax.set_zticks([tick for tick in z_ticks_labels])
    ax.set_zticklabels([f'{label:.{decimals_z}f}' for label in z_ticks_labels])
    for label in ax.get_zticklabels():
        label.set_fontname(fontname)
        label.set_fontsize(fontsize)
    plotdata = data
    plotdata[plotdata > top_zlim] = np.nan
    plotdata[plotdata < bottom_zlim] = np.nan
    ax.set_zlim(top=top_zlim, bottom=bottom_zlim)

    if title is not None:
        ax.set_title(title, pad=15, fontname=fontname, fontsize=title_size)
    # Add grey ticks with labels on both sides of both axes, looking in
    ax.tick_params(axis="both", color='dimgray', pad=10, direction='in',
                   which='both', bottom=True, top=True, left=True, right=True,
                   labelbottom=True, labeltop=False, labelleft=True,
                   labelright=False)
    x, y = np.meshgrid(x, y)
    cmap = plt.get_cmap(color_map)
    if colordata is None:
        colordata = data
    # else:
        # norm = mpl.colors.Normalize(vmin=np.amin(colordata),
        #                             vmax=np.amax(colordata))
        # facecolors = cmap(norm(colordata))
        # img = ax.plot_surface(x, y, data, cmap=cmap, facecolors=facecolors)
    norm = mpl.colors.Normalize(vmin=np.amin(colordata),
                                vmax=np.amax(colordata))
    facecolors = cmap(norm(colordata))
    img = ax.plot_surface(x, y, plotdata, cmap=cmap, facecolors=facecolors)
    if cbar:
        ax, img, cb, cax = papergraph.add_cbar(
            ax,
            img,
            colordata,
            fontname=fontname,
            fontsize=fontsize,
            pad=cbar_pad,
            locator=cbar_locator,
            label=cbar_label,
            decimals=decimals_cbar,
            lower_lim=bottom_cbar_lim,
            upper_lim=top_cbar_lim
        )
    plt.tight_layout()
    return ax
