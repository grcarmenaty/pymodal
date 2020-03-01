import matplotlib.pyplot as plt
import numpy as np
import ntpath
import warnings

from .FRF_utils import (
    unpack_FRF_mat,
    load,
    plot,
)

class FRF():

    """
    
    This class takes either a path to a Frequency Response Function 
    (FRF) file, a list of paths to FRF files (as ['path/to/file_1', ...,
    'path/to/file_n']), a single 2D array or a 3D array where the third
    dimension designates diferent FRFs, and turns it into a list of FRFs
    where each element is one of the 2D arrays that consitute a FRF.
    
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

        # If frf is a string, it is assumed it is a path to file
        if isinstance(frf, str):
            # Value will be a one-length list with the unpacked frf 2D array
            self.value = [unpack_FRF_mat(frf)]
            # Name will have the name of the file
            if name is None:
                self.name = [ntpath.split(frf)[-1]]
        elif isinstance(frf, list):
            #If the first element of the frf list is a string, it is assumed to
            # be a list of file locations
            if isinstance(frf[0], str):
                self.value = []
                if name is None:
                    self.name = []
                for element in frf:
                    self.value.append(unpack_FRF_mat(element))
                    if name is None:
                        self.name.append(ntpath.split(element)[-1])
            # If it is not a list of file locations, then it is assumed to be a
            # list of arrays.
            else:
                self.value = frf
                if name is None:
                    self.name = []
                    for i in range(len(frf)):
                        self.name.append('Unknown name ' + str(i + 1))
        # If frf is not a string nor it is a list, it is assumed to be an array
        else:
            # Try to use frf as a 3D array, if it is not, the except code will 
            # execute, assuming frf is a 2D array.
            try:
                self.value = []
                if name is None:
                    self.name = []
                # This should fail if frf is not a 3D array
                for i in range(frf.shape[2]):
                    if name is None:
                        self.name.append('Unknown name ' + str(i + 1))
                    # Append every FRF along the third dimension 
                    self.value.append(frf[:,:,i])
            except: # Assuming frf is a 2D array
                if name is None:
                    self.name = ['Unknown name 1']
                self.value = [frf]
        
        # The following structure makes sure name info is properly assigned if 
        # it is provided
        if name is not None:
            if isinstance(name, str):
                self.name = [name]
            elif isinstance(name, np.ndarray):
                self.name = name.tolist()
            # If else it is considered to be a list
            else:
                self.name = name
        # For every additional element value has versus the length of name, 
        # append an 'Unknown name' entry.
        for i in range(len(self.name), len(self)):
            self.name.append('Unknown name' + str(i + 1))

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
            raise Exception((f"""The resolution ({resolution} Hz), bandwidth
                ({bandwidth} Hz), min_freq ({min_freq} Hz) and/or max_freq
                ({max_freq} Hz) values introduced were not coherent with each 
                other."""))
        for i in range(len(self)):
            if not(self.value[i].shape == self.value[0].shape):
                raise Exception((f"""One of the FRFs in the provided list has a 
                    different resolution, bandwidth, min_freq and/or max_freq. 
                    The offending entry is: {i}"""))

        self.part = part 

    def __repr__(self):

        """
        
        This is what the instance of the class FRF returns when printed.
        It states how many FRFs are stored within this instance, how 
        many lines each FRF has, as well as resolution, bandwidth 
        (specifying minimum and maximum frequency) and how many data 
        points there are inside each line.
        """
        
        frf_amount = len(self)
        entry = "entries of"  if len(self) > 1 else "entry of"
        lines_amount = self.value[0].shape[1]
        lines = "lines" if self.value[0].shape[1] > 1 else "line"
        each = "each" if len(self) > 1 else "."
        data_points = self.value[0].shape[0]

        return f"""
            FRF object with {frf_amount} {entry} {lines_amount} {lines} {each},
            Resolution: {self.resolution} Hz ({data_points} data points),
            Bandwidth: {self.bandwidth} ({self.min_freq} to {self.max_freq}) Hz
            """

    def __getitem__(self, index:slice):

        """
        
        This makes the FRF class sliceable. Each slice returns an 
        instance of the FRF class with only the FRFs specified by the 
        slice.
        """

        return FRF(self.value[index],
                   resolution = self.resolution,
                   name = self.name[index])

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
        
        new_resolution = int(np.around(new_resolution / self.resolution))
        if new_resolution % self.resolution != 0:
            warnings.warn(f"""The specified new resolution is not divisible by 
                the old resolution.
                The new reolution will be {new_resolution} Hz instead.""")
        new_value = self.value
        for element in new_value:
            element = element[0::new_resolution, :]

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
        for element in new_value:
            for i in line_selection:
                element = element[:, i]
        new_value = np.asarray(new_value).conj().T

        return FRF(frf=new_value,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part=self.part)

    def change_frequencies(self, frequencies:tuple):
        
        """
        
        This function takes a list, tuple or array with two elements, 
        the first referring to the new minimum frequency, and the second
        referring to the new maximum frequency, and returns a new 
        instance of the class with the previous instance's FRFs between 
        the newly specified minimum and maximum frequency.
        """

        frequencies = list(frequencies) # Make sure frequencies is a list
        if not(len(frequencies) == 2): # frequencies should have 2 elements
            raise Exception("""frequencies should be a list, tuple or array 
                with two elements, the first referring to the minimum 
                frequency, and the second referring to the maximum frequency.
                """)
        new_value = self.value
        frequency_start = int(np.around(frequencies[0] / self.resolution))
        frequency_end = int(np.around(frequencies[-1] / self.resolution) + 1)
        for element in new_value:
            element = element[frequency_start:frequency_end, :]

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

        new_value = self.value
        for element in new_value:
            element = np.absolute(element.real)
        
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

        new_value = self.value
        for element in new_value:
            element = np.absolute(element.imag)
        
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

        new_value = self.value
        for element in new_value:
            element = np.absolute(element)
        
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

        new_value = self.value
        for element in new_value:
            element = np.angle(element)
        
        return FRF(frf=new_value,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part='phase')

    # def plot(self,
    #          ax = None,
    #          fontsize:float=12,
    #          title:str='Frequency Response',
    #          title_size:float=None,
    #          major_locator:int=4,
    #          minor_locator:int=4,
    #          fontname:str='Times New Roman',
    #          color:str=None,
    #          ylabel:str=None,
    #          bottom_ylim:float=None):

    #     """
        
    #     This function plots as many FRFs as there are in the instance
    #     of the class. If no axes are specified, then it plots all of the
    #     FRFs in the same figure (this makes the most sense since if you
    #     plotted them into different figures, you could only save the 
    #     last one). If axes are specified, then there should be as many
    #     axes as there are FRFs, otherwise behaviour turns unpredictable.
        
    #     Font size and family can be specified, as well as a special size
    #     for the title. A list of colors can be specified, it is assumed
    #     that these colors are assigned to the axes in order, and if
    #     there are less colors than there are axes then color for the
    #     remaining axes will be assumed to be blue. The amount of major
    #     and minor divisions can be manually changed, as well as label
    #     of the y axis (which is assumed to represent the response
    #     acceleration normalized to the input force) and the bottom limit
    #     of the y axis. The top limit will be a decade higher than the
    #     highest represented value.
        
    #     If phase is being represented, then a graph with fractions of pi
    #     in the y axis will be plotted, usually going from -pi to pi and
    #     a bit more.
    #     """

    #     # The following lines of code create a list of colors. Every FRF in the
    #     # class instance will be pltted unto the same figure cycling through
    #     # the list of colors.
    #     color_list = list(mpl.colors.BASE_COLORS.keys())[0:-1]
    #     color_list.extend(list(mpl.colors.TABLEAU_COLORS.keys()))
    #     css_colors = list(mpl.colors.CSS4_COLORS.keys())
    #     # Randomize the css colors, which are arranged in a color scale, so 
    #     # that graphs are more readable if it ever comes to representing this 
    #     # many FRFs.
    #     random.shuffle(css_colors)
    #     # Get as many matplotlib named color lists as possible, it is not 
    #     # reasonable to expect more than these many colors.
    #     color_list.extend(css_colors)
    #     # Set red as second color instead of green, and green as third color 
    #     # instead of red.
    #     color_list[1] = 'r'
    #     color_list[2] = 'g'
    #     # If no color vector is specified, then create a vector of 'b' 
    #     # (signifying blue color) as long as the amount of FRFs that should 
    #     # be represented.
    #     if color is None:
    #         color = ['b'] * len(self)
    #     else:
    #         color = list(color) # Make sure color is a list
    #         for i in range(len(color), len(self)):
    #             color.append('b') # Start appending 'b' until the color list is as long as there are FRFs
    #     # Preallocate lists
    #     img = []
    #     ax_list = []
    #     if isinstance(ax, np.ndarray): # If axis is an array, flatten it to make sure no list of arrays is created in ax
    #         ax = ax.flatten()
    #     # If axis is iterable, add its elements to ax_list, else (if it's not iterable), apend ax to ax_list
    #     try:
    #         ax_list.extend(ax)
    #     except:
    #         ax_list.append(ax)
    #     ax = ax_list # Once ax_list is a propperly formatted list of axes, assign it to ax
    #     len_ax = len(ax) # Save ax's original length, it can change while operating with it and the code doesn't work with a list of changing length
    #     if len_ax == 1 and ax[0] is None: # If no axis was designated
    #         ax[0] = plt.gca() # Create an axis
    #     for i in range(len_ax, len(self)):
    #         if len_ax == 1: # If there is only one axis, either created inside this method or specified outside of it
    #             ax.append(ax[0]) # Make sure every FRF will be plotted over that same axis
    #             color[i] = color_list[i] # Change the color of every plot beyond the first according to the matplotlib color list
    #         else:
    #             raise Exception('Either specify no axis or specify as many axes as FRFs this instance has in its value.') # If some axes have been specified and some haven't, behaviour can be unexpected, and therefore such a situation is undesirable and of little use. If there is more than one axis but not as many as there are FRFs, then raise an Exception
    #     for i in range(len(self)):
    #         img.append(plot_FRF(self.real().value[i] if self.part == 'complex' else self.value[i], self.max_freq, self.min_freq, self.resolution, ax[i], fontsize, title, title_size, major_locator, minor_locator, fontname, color[i], ylabel, bottom_ylim, self.part)) # Add the axis or figure to the output list
    #     return img

    # def save(self, path: str, decimal_places: int = None):

    #     """\n\nThis function dumps all data necessary to recreate this instance of the class into a json file, which allows comfortable and safe retrieval of the FRFs.\n"""

    #     if decimal_places is None:
    #         decimal_places = '.4' # If no amount of decimal places are designated, then 4 are stored
    #     else:
    #         decimal_places = '.' + str(int(decimal_places)) # If an amount of decimal places is specified, then take the number and format it correctly

    #     data = {
    #         'frf': [np.core.defchararray.add(np.char.mod('%' + decimal_places + 'E', self.value[i].real),np.char.mod('%+' + decimal_places + 'E', self.value[i].imag)).tolist() for i in range(len(self))], # Turn every complex element of the frf array into 'realEx+-imagEy' string format so that it can be stored in a json file. This is done transforming every row in a list, then storing these row lists into frf lists (frf arrays can't be turned dumped to json). finally, these frf lists are stored into a list, which is what is stored into the json file.
    #         'resolution': self.resolution, 
    #         'bandwidth': self.bandwidth, 
    #         'max_freq': self.max_freq, 
    #         'min_freq': self.min_freq, 
    #         'name': self.name, 
    #         'part': self.part,
    #         } # Prepare a dictionary to dump into a json file. Every entry has the FRF class input it corresponds to as key, and their values as values.

    #     compress_json.dump(data, path) # And dump the dictionary