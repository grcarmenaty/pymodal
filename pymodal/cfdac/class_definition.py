import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ntpath
import random
import warnings
from zipfile import ZipFile
from scipy import sparse

import pymodal


class CFDAC():

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
                 ref: pymodal.frf.FRF,
                 frf: pymodal.frf.FRF,
                 resolution: float = None,
                 bandwidth: float = None,
                 max_freq: float = None,
                 min_freq: float = 0,
                 name: list = None,
                 part: str = 'complex',
                 compress = True,
                 diagonal_ratio = None,
                 threshold = 0.15,
                 divisions = 1,
                 _pristine: list = None,
                 _value: list = None):

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

        if type(frf) is pymodal.frf.FRF:
            self.frf = frf
        else:
            self.frf = pymodal.frf.FRF(frf, resolution, bandwidth, max_freq,
                                       min_freq, name)

        if type(ref) is pymodal.frf.FRF:
            self.ref = ref
        elif type(ref) is int:
            self.ref = self.frf[ref]
        else:
            self.ref = pymodal.frf.FRF(ref, resolution, bandwidth, max_freq,
                                       min_freq, name)

        if not(len(self.ref) == 1):
            raise Exception((f"There can only be one refference FRF. There are"
                             f"{len(self.ref)} refferences at the moment."))

        if not(self.ref.value[0].shape == self.frf.value[0].shape
               and self.ref.resolution == self.frf.resolution
               and self.ref.bandwidth == self.frf.bandwidth
               and self.ref.min_freq == self.frf.min_freq
               and self.ref.max_freq == self.frf.max_freq):
            raise Exception(("Refference and altered values must have the same"
                             "number of lines, resolution, bandwidth, minimum"
                             "frequency and maximum frequency."))
        
        self.resolution = self.ref.resolution
        self.divisions = divisions
        self.compressed = compress
        self.bandwidth = self.ref.bandwidth
        self.part = part
        self.min_freq = self.ref.min_freq
        self.max_freq = self.ref.max_freq
        self._xfreq = []
        self._yfreq = []
        for i in range(divisions):
            x_freq_row = []
            y_freq_row = []
            for j in range(divisions):
                x_freq_row.append([self.ref.min_freq +
                                    (j/divisions)*self.bandwidth,
                                    self.ref.min_freq +
                                    ((j+1)/divisions)*self.bandwidth])
                y_freq_row.append([self.ref.min_freq +
                                    ((i)/divisions)*self.bandwidth,
                                    self.ref.min_freq +
                                    ((i+1)/divisions)*self.bandwidth])
            self._xfreq.append(x_freq_row)
            self._yfreq.append(y_freq_row)
        self.diagonal_ratio = diagonal_ratio
        self.threshold = threshold
        if name is None:
            self.name = list(self.frf.name)
            for index, name in enumerate(self.name):
                naming_list = []
                for i in range(self.divisions):
                    naming_list.append([f'{name[i]} ({i}, {j})'
                                        for j in range(self.divisions)])
                self.name[index] = list(naming_list)
        else:
            self.name = name
            try:
                self.name[0][0][0]
            except Exception as __:
                raise Exception('Name not properly formatted')

        if _value is None:
            self.value = []
            for frf in self.frf:
                current_value = pymodal.cfdac.value(self.ref.value[0],
                                                    frf.value[0])
                shape = current_value.shape[0]
                divided_value = []
                for i in range(divisions):
                    start_i = int(((i / self.divisions) * shape))
                    stop_i = int((((i + 1) / self.divisions) * shape))
                    current_value_row = []
                    for j in range(divisions):
                        start_j = int(((j / self.divisions) * shape))
                        stop_j = int((((j + 1) / self.divisions) * shape))
                        current_division = current_value[start_i:stop_i, 
                                                        start_j:stop_j]
                        if self.compressed:
                            current_division = pymodal.cfdac.compress(
                                current_division,
                                self.diagonal_ratio,
                                self.threshold
                            )
                        current_value_row.append(current_division)
                    divided_value.append(current_value_row)
                self.value.append(divided_value)
        else:
            self.value = _value

        if _pristine is None:
            pristine_value = pymodal.cfdac.value(self.ref.value[0],
                                                 frf.value[0])
            shape = pristine_value.shape[0]
            self.pristine = []
            for i in range(divisions):
                start_i = int(((i / self.divisions) * shape))
                stop_i = int((((i + 1) / self.divisions) * shape))
                pristine_row = []
                for j in range(divisions):
                    start_j = int(((j / self.divisions) * shape))
                    stop_j = int((((j + 1) / self.divisions) * shape))
                    pristine_division = pristine_value[start_i:stop_i, 
                                                       start_j:stop_j]
                    if self.compressed:
                        pristine_division = pymodal.cfdac.compress(
                            pristine_division,
                            self.diagonal_ratio,
                            self.threshold
                        )
                    pristine_row.append(pristine_division)
                self.pristine.append(pristine_row)
        else:
            self.pristine = _pristine
            

    def __repr__(self):
        return f"{self.__class__} ({self.__dict__})"


    def __eq__(self, other):
        if isinstance(other, pymodal.cfdac.CFDAC):
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
        return CFDAC(ref=self.ref,
                     frf=self.frf[index],
                     resolution=self.resolution,
                     bandwidth=self.bandwidth,
                     max_freq=self.max_freq,
                     min_freq=self.min_freq,
                     name=self.name[index],
                     part=self.part,
                     compress=self.compressed,
                     diagonal_ratio=self.diagonal_ratio,
                     threshold=self.threshold,
                     divisions=self.divisions,
                     _pristine=self.pristine,
                     _value=self.value[index])


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

        extension = CFDAC(ref=self.ref,
                          frf=frf,
                          resolution=self.resolution,
                          bandwidth=self.bandwidth,
                          max_freq=self.max_freq,
                          min_freq=self.min_freq,
                          name=name,
                          part=self.part,
                          compress=self.compressed,
                          diagonal_ratio=self.diagonal_ratio,
                          threshold=self.threshold,
                          divisions=self.divisions,
                          _pristine=self.pristine)


        self.frf.extend(extension.frf)
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

        return CFDAC(ref=self.ref.change_resolution(new_resolution),
                     frf=self.frf.change_resolution(new_resolution),
                     resolution=self.resolution,
                     bandwidth=self.bandwidth,
                     max_freq=self.max_freq,
                     min_freq=self.min_freq,
                     name=self.name,
                     part=self.part,
                     compress=self.compressed,
                     diagonal_ratio=self.diagonal_ratio,
                     threshold=self.threshold,
                     divisions=self.divisions)


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

        return CFDAC(ref=self.ref.change_lines(line_selection),
                     frf=self.frf.change_lines(line_selection),
                     resolution=self.resolution,
                     bandwidth=self.bandwidth,
                     max_freq=self.max_freq,
                     min_freq=self.min_freq,
                     name=self.name,
                     part=self.part,
                     compress=self.compressed,
                     diagonal_ratio=self.diagonal_ratio,
                     threshold=self.threshold,
                     divisions=self.divisions)


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

        return CFDAC(ref=self.ref.change_frequencies(frequencies),
                     frf=self.frf.change_frequencies(frequencies),
                     resolution=self.resolution,
                     bandwidth=self.bandwidth,
                     max_freq=self.max_freq,
                     min_freq=self.min_freq,
                     name=self.name,
                     part=self.part,
                     compress=self.compressed,
                     diagonal_ratio=self.diagonal_ratio,
                     threshold=self.threshold,
                     divisions=self.divisions)


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
            for i in range(self.divisions):
                for j in range(self.divisions):
                    new_value[index][i][j] = np.absolute(item[i][j].real)
        new_pristine = list(self.pristine)
        for i in range(self.divisions):
            for j in range(self.divisions):
                new_pristine[i][j] = np.absolute(new_pristine[i][j].real)

        return CFDAC(ref=self.ref.real(),
                     frf=self.frf.real(),
                     resolution=self.resolution,
                     bandwidth=self.bandwidth,
                     max_freq=self.max_freq,
                     min_freq=self.min_freq,
                     name=self.name,
                     part='real',
                     compress=self.compressed,
                     diagonal_ratio=self.diagonal_ratio,
                     threshold=self.threshold,
                     divisions=self.divisions,
                     _pristine=new_pristine,
                     _value=new_value)


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
            for i in range(self.divisions):
                for j in range(self.divisions):
                    new_value[index][i][j] = np.absolute(item[i][j].imag)
        new_pristine = list(self.pristine)
        for i in range(self.divisions):
            for j in range(self.divisions):
                new_pristine[i][j] = np.absolute(new_pristine[i][j].imag)

        return CFDAC(ref=self.ref.imag(),
                     frf=self.frf.imag(),
                     resolution=self.resolution,
                     bandwidth=self.bandwidth,
                     max_freq=self.max_freq,
                     min_freq=self.min_freq,
                     name=self.name,
                     part='imag',
                     compress=self.compressed,
                     diagonal_ratio=self.diagonal_ratio,
                     threshold=self.threshold,
                     divisions=self.divisions,
                     _pristine=new_pristine,
                     _value=new_value)


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
            for i in range(self.divisions):
                for j in range(self.divisions):
                    new_value[index][i][j] = np.absolute(item[i][j])
        new_pristine = list(self.pristine)
        for i in range(self.divisions):
            for j in range(self.divisions):
                new_pristine[i][j] = np.absolute(new_pristine[i][j])

        return CFDAC(ref=self.ref.abs(),
                     frf=self.frf.abs(),
                     resolution=self.resolution,
                     bandwidth=self.bandwidth,
                     max_freq=self.max_freq,
                     min_freq=self.min_freq,
                     name=self.name,
                     part='abs',
                     compress=self.compressed,
                     diagonal_ratio=self.diagonal_ratio,
                     threshold=self.threshold,
                     divisions=self.divisions,
                     _pristine=new_pristine,
                     _value=new_value)


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
            for i in range(self.divisions):
                for j in range(self.divisions):
                    new_value[index][i][j] = np.absolute(np.angle(item[i][j]))
        new_pristine = list(self.pristine)
        for i in range(self.divisions):
            for j in range(self.divisions):
                new_pristine[i][j] = np.absolute(np.angle(new_pristine[i][j]))

        return CFDAC(ref=self.ref.phase(),
                     frf=self.frf.phase(),
                     resolution=self.resolution,
                     bandwidth=self.bandwidth,
                     max_freq=self.max_freq,
                     min_freq=self.min_freq,
                     name=self.name,
                     part='phase',
                     compress=self.compressed,
                     diagonal_ratio=self.diagonal_ratio,
                     threshold=self.threshold,
                     divisions=self.divisions,
                     _pristine=new_pristine,
                     _value=new_value)


    def M2L(self):

        if self.part == 'complex':
            M2L_value = self.real().value
        else:
            M2L_value = self.value
        for i in range(len(self)):
            for j in range(self.divisions):
                for k in range(self.divisions):
                    M2L_value[i][j][k] = pymodal.cfdac.M2L(M2L_value[i][j][k])
        return M2L_value


    def SCI(self):

        if self.part == 'complex':
            SCI_value = self.real().value
            pristine = self.real().pristine
        else: # Else use the pristine and altered values for this instance
            SCI_value = self.value
            pristine = self.pristine
        for i in range(len(self)): # For every FRF
            for j in range(self.divisions):
                for k in range(self.divisions):
                    SCI_value[i][j][k] = pymodal.cfdac.SCI(
                        pristine[j][k],
                        SCI_value[i][j][k]
                    ) # Calculate the SCI of that altered CFDAC
        return SCI_value


    def plot(self,
             ax=None,
             fontname: str = 'serif',
             fontsize: float = 12,
             title: list = None,
             title_size: float = None,
             major_x_locator: int = 4,
             minor_x_locator: int = 4,
             major_y_locator: int = 4,
             minor_y_locator: int = 4,
             color_map: str = 'jet',
             xlabel: str = 'Frequency/Hz',
             ylabel: str = 'Frequency/Hz',
             decimals: int = 0,
             cbar: bool = True,
             cbar_pad: float = 0.2):
        ax_list = []
        for i in range(len(self)):
            for j in range(self.divisions):
                for k in range(self.divisions):
                    plot_title = (self.name[i][j][k] if title is None
                                  else title[i][j][k])
                    value = (self.real().value if self.part == 'complex'
                             else self.value)
                    value = (value[i][j][k].toarray() if self.compressed 
                             else value[i][j][k])
                    plot_ax = None if ax is None else ax[i][j][k]
                    ax_list.append(pymodal.cfdac.plot(
                        cfdac=value,
                        xfreq=self._xfreq[j][k],
                        yfreq=self._yfreq[j][k],
                        resolution=self.resolution,
                        ax=plot_ax,
                        fontname=fontname,
                        fontsize=fontsize,
                        title=plot_title,
                        major_x_locator=major_x_locator,
                        minor_x_locator=minor_x_locator,
                        major_y_locator=major_y_locator,
                        minor_y_locator=minor_y_locator,
                        color_map=color_map,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        decimals=decimals,
                        cbar=cbar,
                        cbar_pad=cbar_pad
                    ))
        return ax_list


    # def save(self, path: str, decimal_places: int = None):
    def save(self, path: str, decimals: int = None):

        """

        This function dumps all data necessary to recreate this instance
        of the class into a json file.
        """

        file_list = []
        file_list.append(path.parents[0] / 'refference.zip')
        self.ref.save(file_list[0])
        file_list.append(path.parents[0] / 'frfs.zip')
        self.ref.save(file_list[1])
        for i in range(len(self)):
            for j in range(self.divisions):
                for k in range(self.divisions):
                    file_list.append(path.parents[0] / f'{i}-{j}-{k}.npz')
                    value = (self.value[i][j][k] if decimals is None else
                             np.around(self.value[i][j][k], decimals))
                    if self.compressed:
                        sparse.save_npz(file_list[k + 2], value)
                    else:
                        pymodal.save_array(value, file_list[k + 2])
        for i in range(self.divisions):
                for j in range(self.divisions):
                    file_list.append(path.parents[0] / 'pristine.npz')
                    value = (self.pristine[i][j][k] if decimals is None else
                             np.around(self.pristine[i][j][k], decimals))
                    if self.compressed:
                        sparse.save_npz(file_list[-1], value)
                    else:
                        pymodal.save_array(value, file_list[-1])

        data = {'resolution': self.resolution,
                'bandwidth': self.bandwidth,
                'max_freq': self.max_freq,
                'min_freq': self.min_freq,
                'name': self.name,
                'part': self.part,
                'compresseded': self.compressed,
                'diagonal_ratio': self.diagonal_ratio,
                'threshold': self.threshold,
                'divisions': self.divisions}
        file_list.append(path.parents[0] / 'data.json')
        with open(path.parents[0] / 'data.json', 'w') as fh:
            json.dump(data, fh)

        with ZipFile(path, 'w') as fh:
            for item in file_list:
                fh.write(item, item.name)
                item.unlink()
