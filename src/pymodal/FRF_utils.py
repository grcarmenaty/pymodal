import compress_json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.core import defchararray
import os
import papergraph
import scipy.io as sio

def unpack_FRF_mat(path:str):

    """
    
    Unpack a Frequency Response Function (FRF) array of complexes 
    saved to a .mat file on path.
    """

    frf = sio.loadmat(os.path.normpath(path)) # Load the .mat file
    frf_info = sio.whosmat(os.path.normpath(path)) # Read .mat information
    frf_info = frf_info[0] # Read .mat variable name
    frf = frf[frf_info[0]] # Save refference FRF to file
    return frf

def load(path:str):

    """
    
    This function loads an FRF class instance from a compressed 
    json file
    """

    data = compress_json.load(path)
    # data["frf"]
    frf = data['frf']
    # This loop takes the frf matrixes one by one, adds a j to the string 
    # complex numbers, and adds the array of complexes to a list, which is the 
    # input for the instance of the FRF class.
    for i in range(len(frf)):
        frf[i] = defchararray.add(np.asarray(frf[i]), 'j').astype(complex)
    
    return FRF(frf = frf,
               resolution = data['resolution'], 
               bandwidth = data['bandwidth'], 
               max_freq = data['max_freq'], 
               min_freq = data['min_freq'], 
               name = data['name'], 
               part = data['part']
               )

def plot(*,
             FRF:np.ndarray,
             max_freq:float,
             min_freq:float,
             resolution:float,
             ax=None,
             fontsize:float=12,
             title:str='Frequency Response',
             title_size:float=None,
             major_locator:int=4,
             minor_locator:int=4,
             fontname:str='Times New Roman',
             color:str='blue',
             ylabel:str="Normalized amplitude ($m·s^{-2}·N^{-1}$)",
             bottom_ylim:float=None,
             part:str='complex'):

    """
    
    This function plots an FRF into the specified axis, or unto an 
    axis of its own device if None is specified.
    """

    if ax is None: # If this is not a subplot of a greater figure:
        fig, ax = plt.subplots()

    if part == 'phase':
        if ylabel is None: # If no label for y axis was specified
            ylabel = 'Phase/rad'
    else:
        if bottom_ylim is None: # If no bottom limit is defined
            # Define the bottom limit as four powers of ten lower than average.
            bottom_ylim = 10 ** int(
                np.around(math.log10(np.average(FRF) / 10000)))
        # Define the top limit as the minimum integer exponent of ten necessary 
        # to fit 1.25 times the maximum value.
        top_ylim = 10 ** int(np.ceil(math.log10(1.25 * np.amax(FRF))))
        if ylabel is None:
            # Amplitude is assumed to be acceleration normalized to force input
            ylabel = "Amplitude normalized to input/$\mathrm{m·s^{-2}·N^{-1}}$"
    xlabel = 'Frequency/Hz'
    freq = np.arange(min_freq, max_freq + resolution / 2, resolution)
    ax = lineplot(x=freq,
                  y=FRF,
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
                  bottom_ylim=bottom_ylim,
                  top_ylim=top_ylim)
    if part == 'phase':
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
                denominator = major_locator)))
    else:
        ax.set_yscale('log')
        # Put ticks every log decade
        locmaj = mpl.ticker.LogLocator(base=10.0, subs=(1.0, ))
        ax.yaxis.set_major_locator(locmaj)
        # Put 0 to 9 minor ticks in every decade
        locmin = mpl.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1)
        ax.yaxis.set_minor_locator(locmin)
    # Minor ticks should have no label
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    if ax is None: # If this is not a subplot of a greater figure
        return fig # Return a figure
    else:
        return img # Else return the ax object upon which this was plotted