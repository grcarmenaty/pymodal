from math import gcd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# Code found at https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib  # noqa: E501
# on 19/02/2020 by user Scott Centoni
def multiple_formatter(denominator=4, number=np.pi, latex=r'\pi'):

    def _multiple_formatter(x, pos):
        den = denominator
        num = int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return fr'${latex}$'
            elif num == -1:
                return fr'$-{latex}$'
            else:
                return fr'${num}{latex}$'
        else:
            if num == 1:
                return fr'$\dfrac{{{latex}}}{{{den}}}$'
            elif num == -1:
                return fr'$-\dfrac{{{latex}}}{{{den}}}$'
            else:
                return fr'$\dfrac{{{num}{latex}}}{{{den}}}$'

    return _multiple_formatter


def add_cbar(ax,
             img,
             data: np.ndarray,
             lower_lim: float = None,
             upper_lim: float = None,
             locator: int = None,
             fontname: str = 'Times New Roman',
             fontsize: float = 12,
             pad: float = 0.2,
             label: str = None,
             decimals: int = 1):

    lower_lim = (np.around(np.amin(data), decimals) if lower_lim is None
        else lower_lim)
    upper_lim = (np.around(np.amax(data), decimals) if upper_lim is None
        else upper_lim)
    locator = 5 if locator is None else locator
    step = (upper_lim-lower_lim) / locator
    cmap = img.get_cmap()
    norm = mpl.colors.Normalize(vmin=lower_lim,
                                vmax=upper_lim)
    values = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    if lower_lim > np.around(np.amin(data), decimals):
        if upper_lim < np.around(np.amax(data), decimals):
            cbar = plt.colorbar(values, extend='both', pad=pad, aspect=12)
            img.set_clim(lower_lim - 3*step/4, upper_lim + 3*step/4)
            cbar.ax.set_ylim(bottom=lower_lim - 3*step/4,
                             top=upper_lim + 3*step/4)
        else:
            cbar = plt.colorbar(values, extend='min', pad=pad, aspect=12)
            img.set_clim(lower_lim - 3*step/4, upper_lim)
            cbar.ax.set_ylim(bottom=lower_lim - 3*step/4, top=upper_lim)
    else:
        if upper_lim < np.around(np.amax(data), decimals):
            cbar = plt.colorbar(values, extend='max', pad=pad, aspect=12)
            img.set_clim(lower_lim, upper_lim + 3*step/4)
            cbar.ax.set_ylim(bottom=lower_lim, top=upper_lim + 3*step/4)
        else:
            cbar = plt.colorbar(values, pad=pad, aspect=12)
            img.set_clim(lower_lim, upper_lim)
            cbar.ax.set_ylim(bottom=lower_lim, top=upper_lim)
    y_ticks_labels = np.arange(lower_lim - step,
                               upper_lim + 3*step/2,
                               step)
    cbar.set_ticks([tick for tick in y_ticks_labels])
    cbar.set_ticklabels([f'{ticklabel:.{decimals}f}' 
                         for ticklabel in y_ticks_labels])
    for ticklabel in cbar.ax.get_yticklabels():
        ticklabel.set_fontname(fontname)
        ticklabel.set_fontsize(fontsize)
    if label is not None:
        cbar.ax.set_ylabel(label, fontsize=fontsize, fontname=fontname,
                           labelpad=10)
    return ax, img, cbar, cbar.ax


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
        ax, img, cb, cax = add_cbar(
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


