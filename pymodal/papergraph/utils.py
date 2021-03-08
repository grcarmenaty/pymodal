from math import gcd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


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
