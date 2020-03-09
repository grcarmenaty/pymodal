import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ntpath
import random
import warnings
from zipfile import ZipFile

import pymodal


class FRF():

    """
    Store and manipulate Frequency Response Functions (FRFs).

    Parameters
    ----------
    frf: list of 2D arrays or 3D array or list of file paths or 
            file path
        Collection of FRF arrays.
    resolution: float, optional
        How many Hz there are between two adjacent points of the FRFs,
        default is None.
    bandwidth: float, optional
        How many Hz there are between the minimum and maximum
        frequencies of the FRFs, default is None.
    max_freq: float, optional
        What is the maximum frequency of the FRFs, in Hz, default is
        None.
    min_freq: float, optional
        What is the minimum frequency of the FRFs, in Hz, default is
        None.
    name: list of strings, optional
        A descriptive name for each FRF, default is None.
    part: string, optional
        Whether the values of the FRF is the real, imaginary, phase,
        module or complex part of the values, default is complex.

    Returns
    -------
    out: FRF class
        FRF class object with all introduced FRFs and their information.

    Notes
    -----
    At least one of resolution, bandwidth or max_freq must be specified,
    and their values must be coherent.
    """


    def __init__(self,
                 frf: list,
                 resolution: float = None,
                 bandwidth: float = None,
                 max_freq: float = None,
                 min_freq: float = 0,
                 name: list = None,
                 part: str = 'complex'):

        """
        Constructor for FRF class.

        Parameters
        ----------
        frf: list of 2D arrays or 3D array or list of file paths or 
                file path
            Collection of FRF arrays.
        resolution: float, optional
            How many Hz there are between two adjacent points of the
            FRFs, default is None.
        bandwidth: float, optional
            How many Hz there are between the minimum and maximum
            frequencies of the FRFs, default is None.
        max_freq: float, optional
            What is the maximum frequency of the FRFs, in Hz, default is
            None.
        min_freq: float, optional
            What is the minimum frequency of the FRFs, in Hz, default is
            None.
        name: list of strings, optional
            A descriptive name for each FRF, default is None.
        part: string, optional
            Whether the values of the FRF is the real, imaginary, phase,
            module or complex part of the values, default is complex.

        Returns
        -------
        out: FRF class
            FRF class object with all introduced FRFs and their
            information.

        Notes
        -----
        At least one of resolution, bandwidth or max_freq must be 
        specified, and their values must be coherent.
        """

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
                    self.value.append(pymodal.load_array(item))
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
                    self.value.append(frf[:, :, i])
            except Exception as __:  # Assuming frf is a 2D array  # noqa F841
                if name is None:
                    self.name = ['Unknown name 1']
                self.value = [frf]
        # The last assumption the function makes is that frf is a file path
        else:
            self.value = [pymodal.load_array(frf)]
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
        self.min_freq = min_freq  # Minimum frequency is assumed to be 0
        if bandwidth is None:
            if max_freq is None:
                # If neither bandwidth nor maximum frequency are defined
                self.resolution = resolution  # Then resolution MUST be defined
                # So bandwidth and maximum frequency can be calculated
                self.bandwidth = (self.value[0].shape[0] - 1) * self.resolution
                self.max_freq = self.min_freq + self.bandwidth
            else:  # If bandwidth is not defined but maximum frequency is
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
            raise Exception((
                f"The resolution ({resolution} Hz), bandwidth ({bandwidth} Hz)"
                f", min_freq ({min_freq} Hz) and/or max_freq ({max_freq} Hz) "
                f"values introduced were not coherent with each other."))
        for i in range(len(self)):
            if not(self.value[i].shape == self.value[0].shape):
                raise Exception((
                    f"One of the FRFs in the provided list has a different "
                    f"number of lines, resolution, bandwidth, min_freq and/or "
                    f"max_freq. The offending entry is: {i} with shape "
                    f"{self.value[i].shape}"))

        self.part = part


    def __repr__(self):
        return f"{self.__class__} ({self.__dict__})"


    def __eq__(self, other):
        if isinstance(other, pymodal.frf.FRF):
            own_dict = dict(self.__dict__)
            del own_dict['value']
            other_dict = dict(other.__dict__)
            del other_dict['value']
            try:
                equal_value = all([np.array_equal(
                    self.value[i],
                    other.value[i]
                    ) for i in range(len(self.value))])
            except Exception as __:  # noqa F841
                equal_value = False
            return own_dict == other_dict and equal_value
        else:
            return False


    def __getitem__(self, index: slice):
        return FRF(frf=self.value[index],
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name[index],
                   part=self.part)

    def __len__(self):
        return len(self.value)

    def extend(self, frf: list, name: list = None):

        """
        Adds an FRF to the current instance and returns None.

        Parameters
        ----------
        frf : list of 2D arrays or 3D array or list of file paths or 
                file path
            Collection of FRF arrays.
        name: list of strings, optional
            A descriptive name for each FRF, default is None.

        Returns
        -------
        out: None

        Notes
        -----
        The rest of parameters for the new FRFs are assumed to be the
        same as in the instance that is being extended.
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

    def change_resolution(self, new_resolution: float):

        """
        Create a new instance with the same FRFs, but with a different
        frequential resolution.

        Parameters
        ----------
        new_resolution : float
            Desired new frequential resolution.

        Returns
        -------
        out: FRF class
            New instance with a different frequential resolution.

        Notes
        -----
        In order to avoid interpolation, the new frequency can only be
        a multiple of the original frequency.
        """

        if new_resolution < self.resolution:
            raise Exception("The new resolution must be greater than the old"
                            "one.")
        step = int(np.around(new_resolution / self.resolution))
        if new_resolution % self.resolution != 0:
            warnings.warn((
                f"The specified new resolution is not divisible by "
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

    def change_lines(self, line_selection: list):

        """
        Create a new instance with the same FRFs, but with a different
        set of lines.

        Parameters
        ----------
        line_selection : list of integers
            A list of which lines are to be used for the new instance.

        Returns
        -------
        out: FRF class
            New instance with a different set of lines.
        """

        line_selection = list(line_selection)  # Line selection must be a list
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

    def change_frequencies(self, frequencies: list):

        """
        Create a new instance with the same FRFs, but with a different
        frequency band.

        Parameters
        ----------
        frequencies : list of float
            The new minimum and maximum frequencies.

        Returns
        -------
        out: FRF class
            New instance with a different frequency band.

        Notes
        -----
        frequencies must be a two-element iterable object.
        """

        frequencies = list(frequencies)  # Make sure frequencies is a list
        if not(len(frequencies) == 2):  # frequencies should have 2 items
            raise Exception((
                f"frequencies should be a list, tuple or array "
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
        Create a new instance with the real part of the FRFs.

        Returns
        -------
        out: FRF class
            New instance with only the real part of the FRFs.
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
        Create a new instance with the imaginary part of the FRFs.

        Returns
        -------
        out: FRF class
            New instance with only the imaginary part of the FRFs.
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
        Create a new instance with the absolute value of the FRFs.

        Returns
        -------
        out: FRF class
            New instance with only the absolute value of the FRFs.
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
        Create a new instance with the phase of the FRFs.

        Returns
        -------
        out: FRF class
            New instance with only the phase of the FRFs.
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
             fontsize: float = 12,
             title: str = 'Frequency Response',
             title_size: float = None,
             major_locator: int = 4,
             minor_locator: int = 4,
             fontname: str = 'Times New Roman',
             color: str = None,
             ylabel: str = None,
             bottom_ylim: float = None):

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
            color = list(color)  # Make sure color is a list
            for i in range(len(color), len(self)):
                color.append(color_list[i-len(color)])

        # Make sure ax is a list of axes as long as there are FRFs stored.
        if ax is None:
            ax = [plt.gca()]
        elif isinstance(ax, np.ndarray):
            ax = ax.flatten()
        try:
            ax = list(ax)
        except Exception as __:  # noqa F841
            ax = [ax]
        for _ in range(len(ax), len(self)):
            ax.append(plt.gca())

        img = self.real().value if self.part == 'complex' else self.value
        for index, item in enumerate(img):
            img[index] = pymodal.frf.plot(frf=item,
                                          max_freq=self.max_freq,
                                          min_freq=self.min_freq,
                                          resolution=self.resolution,
                                          ax=ax[index],
                                          fontsize=fontsize,
                                          title=title,
                                          title_size=title_size,
                                          major_locator=major_locator,
                                          minor_locator=minor_locator,
                                          fontname=fontname,
                                          color=color[index],
                                          ylabel=ylabel,
                                          bottom_ylim=bottom_ylim,
                                          part=self.part)
        return img

    # def save(self, path: str, decimal_places: int = None):
    def save(self, path: str, decimals: int = None):

        """

        This function dumps all data necessary to recreate this instance
        of the class into a json file.
        """

        frf_value = (list(self.value) if decimals is None
                     else [np.around(item, decimals) for item in self.value])
        file_list = []
        for index, item in enumerate(frf_value):
            file_list.append(path.parents[0] / f'{self.name[index]}.npz')
            pymodal.save_array(item, file_list[index])

        data = {'resolution': self.resolution,
                'bandwidth': self.bandwidth,
                'max_freq': self.max_freq,
                'min_freq': self.min_freq,
                'name': self.name,
                'part': self.part}
        file_list.append(path.parents[0] / 'data.json')
        with open(path.parents[0] / 'data.json', 'w') as fh:
            json.dump(data, fh)

        with ZipFile(path, 'w') as fh:
            for item in file_list:
                fh.write(item, item.name)
                item.unlink()
