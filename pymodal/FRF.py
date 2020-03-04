import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.core import defchararray
import ntpath
import warnings

import pymodal

class FRF():

    """
    
    This class takes either a path to a Frequency Response Function 
    (FRF) file, a list of paths to FRF files (as ['path/to/file_1', ...,
    'path/to/file_n']), a single 2D array or a 3D array where the third
    dimension designates diferent FRFs, and turns it into a list of FRFs
    where each item is one of the 2D arrays that consitute a FRF.
    
    Resolution, bandwidth, max_freq (maximum frequency) and min_freq 
    (minimum frequency) are assumed to be in Hertz (Hz) and HAVE TO be 
    coherent with each other and the amount of points the function has. 
    All inputted FRFs HAVE TO have the same resolution, bandwidth, 
    min_freq and max_freq, and the one against which both conditions are
    checked is the first one.
    
    If no list of names is provided then an 'Unknown name' list of 
    length equal to the amount of FRFs will be created. If some names 
    are provided, but not as many as FRFs, those names will be assumed 
    to be for the first FRFs, and the name list will be extended with 
    'Unknown name' entries to have as many names as there are FRFs.
    """

    def __init__(self,
                 frf:list,
                 resolution:float=None,
                 bandwidth:float=None,
                 max_freq:float=None,
                 min_freq:float=0,
                 name:list=None,
                 part:str='complex'):
        
        # The following structure makes sure either of the possible inputs is 
        # correctly processed so that a list of FRFs is composed, and name info
        # is assigned in case no name info was provided.

        if isinstance(frf, list):
            if isinstance(frf[0], np.ndarray):
                self.value = list(frf)
                if name is None:
                    self.name = []
                    for i in range(len(frf)):
                        self.name.append(f'Unknown name {i + 1}')
            # If it is not a list of arrays, then it is assumed to be a list of
            # file locations.
            else:
                self.value = []
                if name is None:
                    self.name = []
                for item in frf:
                    self.value.append(pymodal.unpack_FRF_mat(item))
                    if name is None:
                        self.name.append(ntpath.split(item)[-1])
        elif isinstance(frf, np.ndarray):
            # Try to use frf as a 3D array, if it is not, the except code will 
            # execute, assuming frf is a 2D array.
            try:
                self.value = []
                if name is None:
                    self.name = []
                # This should fail if frf is not a 3D array
                for i in range(frf.shape[2]):
                    if name is None:
                        self.name.append(f'Unknown name {i + 1}')
                    # Append every FRF along the third dimension 
                    self.value.append(frf[:,:,i])
            except: # Assuming frf is a 2D array
                if name is None:
                    self.name = ['Unknown name 1']
                self.value = [frf]
        # The last assumption the function makes is that frf is a file path
        else:
            self.value = [pymodal.unpack_FRF_mat(frf)]
            self.name = [ntpath.split(frf)[-1]]
        
        # The following structure makes sure name info is properly assigned if 
        # it is provided
        if name is not None:
            if isinstance(name, str):
                self.name = [name]
            elif isinstance(name, np.ndarray):
                self.name = name.tolist()
            # If else it is considered to be a list
            else:
                self.name = list(name)
        # For every additional item value has versus the length of name, 
        # append an 'Unknown name' entry.
        for i in range(len(self.name), len(self)):
            self.name.append(f'Unknown name {i + 1}')

        if len(self.name) != len(self.value):
            raise Exception((f"There were {len(self.name)} names for "
                             f"{len(self.value)} values."))

        # The following structure makes sure at least one of max_freq, 
        # resolution or bandwidth is defined (min_freq is assumed to be zero) 
        # and calculates the non-specified variables.
        self.min_freq = min_freq # Minimum frequency is assumed to be 0
        if bandwidth is None:
            if max_freq is None:
                # If neither bandwidth nor maximum frequency are defined
                self.resolution = resolution # Then resolution MUST be defined
                # So bandwidth and maximum frequency can be calculated
                self.bandwidth = (self.value[0].shape[0] - 1) * self.resolution
                self.max_freq = self.min_freq + self.bandwidth
            else: # If bandwidth is not defined but maximum frequency is
                self.max_freq = max_freq
                self.bandwidth = self.max_freq - self.min_freq
                if resolution is None:
                    self.resolution = (self.bandwidth / 
                                      (self.value[0].shape[0] - 1))
                else:
                    self.resolution = resolution
        else:
            self.bandwidth = bandwidth
            if max_freq is None:
                self.max_freq = self.min_freq + self.bandwidth
                if resolution is None:
                    self.resolution = (self.bandwidth /
                                      (self.value[0].shape[0] - 1))
                else:
                    self.resolution = resolution
            else:
                self.max_freq = max_freq
                if resolution is None:
                    self.resolution = (self.bandwidth / 
                                      (self.value[0].shape[0] - 1))
                else:
                    self.resolution = resolution

        # In case the user inputs more values than is necessary and those 
        # values don't make sense together, raise an exception.
        calculated_resolution = self.bandwidth / (self.value[0].shape[0] - 1)
        resolution_error = not(self.resolution == calculated_resolution)
        calculated_bandwidth = self.max_freq - self.min_freq
        bandwidth_error = not(self.bandwidth == calculated_bandwidth)
        if resolution_error or bandwidth_error:
            raise Exception((f"The resolution ({resolution} Hz), bandwidth "
                f"({bandwidth} Hz), min_freq ({min_freq} Hz) and/or max_freq "
                f"({max_freq} Hz) values introduced were not coherent with "
                f"each other."))
        for i in range(len(self)):
            if not(self.value[i].shape == self.value[0].shape):
                raise Exception((f"One of the FRFs in the provided list has a "
                    f"different resolution, bandwidth, min_freq and/or "
                    f"max_freq. The offending entry is: {i}"))

        self.part = part 

    def __repr__(self):

        """
        
        This is what the instance of the class FRF returns when printed.
        It states how many FRFs are stored within this instance, how 
        many lines each FRF has, as well as resolution, bandwidth 
        (specifying minimum and maximum frequency) and how many data 
        points there are inside each line.
        """
        
        dict_to_print = dict(self.__dict__)
        
        array_plural = "arrays"  if len(self) > 1 else "array"
        shape = self.value[0].shape
        dict_to_print['value'] = f"{len(self)} {array_plural} of shape {shape}"
        return print(dict_to_print)
        # frf_amount = len(self)
        # entry = "entries of"  if len(self) > 1 else "entry of"
        # lines_amount = self.value[0].shape[1]
        # lines = "lines" if self.value[0].shape[1] > 1 else "line"
        # each = " each" if len(self) > 1 else ""
        # data_points = self.value[0].shape[0]

        # return (f"FRF object with {frf_amount} {entry} {lines_amount} {lines}"
        #         f"{each},\nResolution: {self.resolution} Hz ({data_points} "
        #         f"data points),\nBandwidth: {self.bandwidth} ({self.min_freq} "
        #         f"to {self.max_freq}) Hz)")

    def __eq__(self, other):
        if isinstance(other, pymodal.FRF):
            own_dict = dict(self.__dict__)
            other_dict = dict(other.__dict__)
            own_dict['value'] = [item.tolist() for item in own_dict['value']]
            other_dict['value'] = [item.tolist() 
                                   for item in other_dict['value']]
            return own_dict == other_dict
        else:
            return False

    def __getitem__(self, index:slice):

        """
        
        This makes the FRF class sliceable. Each slice returns an 
        instance of the FRF class with only the FRFs specified by the 
        slice.
        """

        return FRF(frf=self.value[index],
                   resolution = self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name = self.name[index],
                   part=self.part)

    def __len__(self):

        """
        
        This function returns how many FRFs are stored inside the 
        instance of the class
        """ 

        return len(self.value)

    def extend(self, frf:list, name:list=None):

        """
        
        This takes FRFs in the same possible formats as the FRF class 
        and adds them to the current instance's value.
        """

        # New class instance with the FRFs that should be appended. This is 
        # necessary so that FRFs and names are propperly formatted lists, in 
        # this way, using the list.extend method becomes an option.
        
        extension = FRF(frf=frf,
                        resolution=self.resolution,
                        bandwidth=self.bandwidth,
                        max_freq=self.max_freq,
                        min_freq=self.min_freq,
                        name=name,
                        part=self.part)
        
        if name is None:
            name = []
            for i in range(len(extension)):
                name.append(f'Unknown name {len(self) + i + 1}')
            extension.name = name

        self.value.extend(extension.value)
        self.name.extend(extension.name)
    
    def change_resolution(self, new_resolution:float):

        """
        
        This function returns a new instance of the FRF class with all 
        of the FRFs from the instance from where the method was called 
        but with a new resolution. The new resolution will be rounded up
        to the closest multiple of the original resolution, since 
        interpolation is undesirable.
        """

        step = int(np.around(new_resolution / self.resolution))
        if new_resolution % self.resolution != 0:
            warnings.warn((f"The specified new resolution is not divisible by "
                f"the old resolution. The new reolution will be "
                f"{step * self.resolution} Hz instead."))
        new_resolution = step * self.resolution
        new_value = list(self.value)
        for index, item in enumerate(new_value):
            new_value[index] = item[0::step, :]

        return FRF(frf=new_value,
                   resolution=new_resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part=self.part)

    def change_lines(self, line_selection:list):

        """
        
        This function takes a list of desired lines, where line refers 
        to each of the points of the experimental mesh of points where 
        the different spectra were measured, and returns a new instance 
        of the class with the previous instance's FRFs composed only by 
        the selected lines.
        """

        line_selection = list(line_selection) # Line selection must be a list
        new_value = self.value
        for index, item in enumerate(new_value):
            new_value[index] = [item[:, i] for i in line_selection]
            new_value[index] = np.asarray(new_value[index]).conj().T

        return FRF(frf=new_value,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part=self.part)

    def change_frequencies(self, frequencies:list):
        
        """
        
        This function takes a list, tuple or array with two items, 
        the first referring to the new minimum frequency, and the second
        referring to the new maximum frequency, and returns a new 
        instance of the class with the previous instance's FRFs between 
        the newly specified minimum and maximum frequency.
        """

        frequencies = list(frequencies) # Make sure frequencies is a list
        if not(len(frequencies) == 2): # frequencies should have 2 items
            raise Exception((f"frequencies should be a list, tuple or array "
                f"with two items, the first referring to the minimum "
                f"frequency, and the second referring to the maximum "
                f"frequency."))
        new_value = self.value
        frequency_start = int(np.around(frequencies[0] / self.resolution))
        frequency_end = int(np.around(frequencies[-1] / self.resolution) + 1)
        for index, item in enumerate(new_value):
            new_value[index] = item[frequency_start:frequency_end, :]

        return FRF(frf=new_value,
                   resolution=self.resolution,
                   bandwidth=frequencies[-1] - frequencies[0],
                   max_freq=frequencies[-1],
                   min_freq=frequencies[0],
                   name=self.name,
                   part=self.part)

    def real(self):

        """
        
        This function returns a new instance of the FRF class with only 
        the real part of the original instance's FRF.
        """

        new_value = list(self.value)
        for index, item in enumerate(new_value):
            new_value[index] = np.absolute(item.real)
        
        return FRF(frf=new_value,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part='real')

    def imag(self):

        """
        
        This function returns a new instance of the FRF class with only 
        the imaginary part of the original instance's FRF.
        """

        new_value = list(self.value)
        for index, item in enumerate(new_value):
            new_value[index] = np.absolute(item.imag)
        
        return FRF(frf=new_value,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part='imag')

    def abs(self):

        """
        
        This function returns a new instance of the FRF class with only 
        the magnitude of the original instance's FRF.
        """

        new_value = list(self.value)
        for index, item in enumerate(new_value):
            new_value[index] = np.absolute(item)
        
        return FRF(frf=new_value,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part='abs')

    def phase(self):

        """

        This function returns a new instance of the FRF class with only 
        the phase of the original instance's FRF.
        """

        new_value = list(self.value)
        for index, item in enumerate(new_value):
            new_value[index] = np.angle(item)
        
        return FRF(frf=new_value,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part='phase')

    def plot(self,
             ax=None,
             fontsize:float=12,
             title:str='Frequency Response',
             title_size:float=None,
             major_locator:int=4,
             minor_locator:int=4,
             fontname:str='Times New Roman',
             color:str=None,
             ylabel:str=None,
             bottom_ylim:float=None):

        """
        
        This function plots as many FRFs as there are in the instance
        of the class. If no axes are specified, then it plots all of the
        FRFs in the same figure (this makes the most sense since if you
        plotted them into different figures, you could only save the 
        last one). If axes are specified, then there should be as many
        axes as there are FRFs, otherwise behaviour turns unpredictable.
        
        Font size and family can be specified, as well as a special size
        for the title. A list of colors can be specified, it is assumed
        that these colors are assigned to the axes in order, and if
        there are less colors than there are axes then color for the
        remaining axes will be assumed to be blue. The amount of major
        and minor divisions can be manually changed, as well as label
        of the y axis (which is assumed to represent the response
        acceleration normalized to the input force) and the bottom limit
        of the y axis. The top limit will be a decade higher than the
        highest represented value.
        
        If phase is being represented, then a graph with fractions of pi
        in the y axis will be plotted, usually going from -pi to pi and
        a bit more.
        """

        # The following lines of code create a list of colors. Every FRF in the
        # class instance will be pltted unto the same figure cycling through
        # the list of colors.
        color_list = list(mpl.colors.BASE_COLORS.keys())[0:-1]
        color_list.extend(list(mpl.colors.TABLEAU_COLORS.keys()))
        css_colors = list(mpl.colors.CSS4_COLORS.keys())
        # Randomize the css colors, which are arranged in a color scale, so 
        # that graphs are more readable if it ever comes to representing this 
        # many FRFs.
        random.shuffle(css_colors)
        # Get as many matplotlib named color lists as possible, it is not 
        # reasonable to expect more than these many colors.
        color_list.extend(css_colors)
        # Set red as second color instead of green, and green as third color 
        # instead of red.
        color_list[1] = 'r'
        color_list[2] = 'g'
        # If no color vector is specified, then create a vector of 'b' 
        # (signifying blue color) as long as the amount of FRFs that should 
        # be represented.
        if color is None:
            color = list(color_list[0:len(self)])
        else:
            color = list(color) # Make sure color is a list
            for i in range(len(color), len(self)):
                color.append(color_list[i-len(color)])
        
        # Make sure ax is a list of axes as long as there are FRFs stored.
        if ax is None:
            ax = [plt.gca()]
        elif isinstance(ax, np.ndarray):
            ax = ax.flatten()
        ax = list(ax)
        for _ in range(len(ax), len(self)):
            ax.append(plt.gca())

        img = self.real().value[i] if self.part == 'complex' else self.value[i]
        for index, item in enumerate(img):
            img[index] = pymodal.plot_FRF(frf=item,
                                          max_freq=self.max_freq,
                                          min_freq=self.min_freq,
                                          resolution=self.resolution,
                                          ax=ax[i],
                                          fontsize=fontsize,
                                          title=title,
                                          title_size=title_size,
                                          major_locator=major_locator,
                                          minor_locator=minor_locator,
                                          fontname=fontname,
                                          color=color[i],
                                          ylabel=ylabel,
                                          bottom_ylim=bottom_ylim,
                                          part=self.part)
        return img

    def save(self, path:str, decimal_places:int=None):

        """
        
        This function dumps all data necessary to recreate this instance
        of the class into a json file.\n"""

        decimal_places = 4 if decimal_places is None else decimal_places
        frf = self.value
        for index, item in enumerate(frf):
            real_part = np.char.mod(f'%.{decimal_places}E', item.real)
            imag_part = np.char.mod(f'+%.{decimal_places}E', item.imag)
            frf[index] = defchararray.add(real_part, imag_part).tolist()
        
        data = {'frf': frf,
                'resolution': self.resolution, 
                'bandwidth': self.bandwidth, 
                'max_freq': self.max_freq, 
                'min_freq': self.min_freq, 
                'name': self.name, 
                'part': self.part}

        compress_json.dump(data, path)