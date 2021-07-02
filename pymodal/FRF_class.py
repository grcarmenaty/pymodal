import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import optimize
from scipy import interpolate
import random
import warnings
import pathlib
from zipfile import ZipFile

import pymodal


class FRF():
    """
    Store and manipulate Frequency Response Functions (FRFs).
    """


    def __init__(self,
                 frf: np.ndarray,
                 resolution: float = None,
                 bandwidth: float = None,
                 max_freq: float = None,
                 min_freq: float = 0,
                 name: list = None,
                 part: str = 'complex',
                 modal_frequencies: list = None):
        """
        Constructor for FRF class.

        Parameters
        ----------
        frf : 3D array
            Collection of FRF arrays.

        resolution : float, optional
            How many Hz there are between two adjacent points of the
            FRFs, default is None.

        bandwidth : float, optional
            How many Hz there are between the minimum and maximum
            frequencies of the FRFs, default is None.

        max_freq : float, optional
            What is the maximum frequency of the FRFs, in Hz, default is
            None.

        min_freq : float, optional
            What is the minimum frequency of the FRFs, in Hz, default is
            None.

        name : list of strings, optional
            A descriptive name for each FRF, default is None.

        part : string, optional
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

        self.value = frf

        # The following structure makes sure name info is properly assigned if
        # it is provided
        if name is None:
            name = []
        # For every additional item value has versus the length of name,
        # append an 'Unknown name' entry.
        self.data_points = self.value.shape[0]
        self.lines = self.value.shape[1]
        self.value = self.value.reshape((self.data_points, self.lines, -1))
        name = list(name)
        if len(name) > len(self):
            raise Exception("Too many names were provided.")
        for i in range(len(name), len(self)):
            name.append(f'Unknown name {i + 1}')
        self.name = name

        # The following structure makes sure at least one of max_freq,
        # resolution or bandwidth is defined (min_freq is assumed to be zero)
        # and calculates the non-specified variables.
        self.min_freq = min_freq  # Minimum frequency is assumed to be 0
        if bandwidth is None:
            if max_freq is None:
                # If neither bandwidth nor maximum frequency are defined
                self.resolution = resolution  # Then resolution MUST be defined
                # So bandwidth and maximum frequency can be calculated
                self.bandwidth = (self.data_points - 1) * self.resolution
                self.max_freq = self.min_freq + self.bandwidth
            else:  # If bandwidth is not defined but maximum frequency is
                self.max_freq = max_freq
                self.bandwidth = self.max_freq - self.min_freq
                if resolution is None:
                    self.resolution = (self.bandwidth /
                                       (self.data_points - 1))
                else:
                    self.resolution = resolution
        else:
            self.bandwidth = bandwidth
            if max_freq is None:
                self.max_freq = self.min_freq + self.bandwidth
                if resolution is None:
                    self.resolution = (self.bandwidth /
                                       (self.data_points - 1))
                else:
                    self.resolution = resolution
            else:
                self.max_freq = max_freq
                if resolution is None:
                    self.resolution = (self.bandwidth /
                                       (self.data_points - 1))
                else:
                    self.resolution = resolution

        # In case the user inputs more values than is necessary and those
        # values don't make sense together, raise an exception.
        calculated_resolution = self.bandwidth / (self.data_points - 1)
        resolution_error = not(self.resolution == calculated_resolution)
        calculated_bandwidth = self.max_freq - self.min_freq
        bandwidth_error = not(self.bandwidth == calculated_bandwidth)
        if resolution_error or bandwidth_error:
            raise Exception((
                f"The resolution ({resolution} Hz), bandwidth ({bandwidth} Hz)"
                f", min_freq ({min_freq} Hz) and/or max_freq ({max_freq} Hz)"
                f" values introduced were not coherent with each other.\n\n"
                f"bandwidth = (data_points - 1) * resolution\n"
                f"max_freq = min_freq + bandwidth"
            ))

        self.freq_vector = np.arange(
            self.min_freq, self.max_freq + self.resolution / 2, self.resolution
        )
        self.part = part
        if modal_frequencies is None:
            warnings.warn("The modal frequencies will now be approximated from"
                    " the observed peaks in the signal. Take this with a grain"
                    " of salt.")
            self.modal_frequencies = list(self._modal_frequencies(distance=5))
        else:
            self.modal_frequencies = list(modal_frequencies)


    def __repr__(self):
        return f"{self.__class__} ({self.__dict__})"


    def __eq__(self, other):
        if isinstance(other, pymodal.FRF):
            own_dict = dict(self.__dict__)
            own_vectors = []
            key_list = list(own_dict.keys())
            for key in key_list:
                if isinstance(own_dict[key], np.ndarray):
                    own_vectors.append(own_dict[key])
                    del own_dict[key]
            other_dict = dict(other.__dict__)
            other_vectors = []
            key_list = list(other_dict.keys())
            for key in key_list:
                if isinstance(other_dict[key], np.ndarray):
                    other_vectors.append(other_dict[key])
                    del other_dict[key]
            equal_vector = len(own_vectors) == len(other_vectors)
            for i, own_vector in enumerate(own_vectors):
                if not(equal_vector):
                    break
                equal_vector = (np.array_equal(own_vector, other_vectors[i])
                                and equal_vector)

            return own_dict == other_dict and equal_vector
        else:
            return False


    def __getitem__(self, index: slice):
        if not isinstance(index, slice):
            index = slice(index, index + 1, None)
        return FRF(frf=self.value[:, :, index],
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name[index.start:index.stop:index.step],
                   part=self.part,
                   modal_frequencies=self.modal_frequencies)


    def __len__(self):
        return self.value.shape[2]


    def extend(self, frf: list, name: list = None):
        """
        Adds an FRF to the current instance and returns None.

        Parameters
        ----------
        frf : 3D array
            Collection of FRF arrays.
        name : list of strings, optional
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

        if name is None:
            name = []
        name = list(name)
        frf = frf.reshape((frf.shape[0], frf.shape[1], -1))
        for i in range(len(name), frf.shape[2]):
            name.append(f'Unknown name {len(self) + i + 1}')
        if len(name) > frf.shape[2]:
            raise Exception('Too many names were provided')
        new_name = list(self.name)
        new_name.extend(name)
        return FRF(frf=np.dstack((self.value, frf)),
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=new_name,
                   part=self.part,
                   modal_frequencies=self.modal_frequencies)


    def normalize(self):  #untested
        """
        Create a new instance with the same FRFs normalized to their
        maximum.

        Returns
        -------
        out : FRF class
            New instance with all FRFs normalized to their maximum.

        Notes
        -----
        frequencies must be a two-element iterable object.
        """

        assert self.part == 'complex'
        frf = np.array(self.value)
        for i in range(len(self)):
            frf[:, :, i] = frf[:, :, i] / np.amax(np.abs(frf[:, :, i]))
        return FRF(frf=frf,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part=self.part,
                   modal_frequencies=self.modal_frequencies)


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
        out : FRF class
            New instance with a different frequential resolution.

        Notes
        -----
        In order to avoid interpolation, the new frequency can only be
        a multiple of the original frequency.
        """
        
        if new_resolution % self.resolution != 0:
            warnings.warn("The resulting FRF will be interpolated according to"
                          " the desired new resolution.")
            print(self.freq_vector.shape)
            new_freq_vector = np.arange(
                self.min_freq, self.max_freq+new_resolution, new_resolution
            )
            new_value = []
            for i in range(self.value.shape[-1]):
                frf = self.value[..., i]
                new_frf = []
                for j in range(frf.shape[-1]):
                    line_real = frf[..., j].real
                    line_imag = frf[..., j].imag
                    new_line_real = interpolate.interp1d(new_freq_vector,
                                                         line_real)
                    new_line_imag = interpolate.interp1d(new_freq_vector,
                                                         line_imag)
                    new_line = (new_line_real(new_freq_vector) +
                                new_line_imag(new_freq_vector)*1j)
                    new_frf.append(new_line)
                new_value.append(new_frf)
            new_value = np.array(new_value)
        else:
            step = int(new_resolution / self.resolution)
            new_value = self.value[0::step, :, :]

        return FRF(frf=new_value,
                   resolution=new_resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part=self.part,
                   modal_frequencies=self.modal_frequencies)


    def change_lines(self, line_selection: np.ndarray):
        """
        Create a new instance with the same FRFs, but with a different
        set of lines.

        Parameters
        ----------
        line_selection : list of integers
            A list of which lines are to be used for the new instance.

        Returns
        -------
        out : FRF class
            New instance with a different set of lines.
        """

        return FRF(frf=self.value[:, line_selection, :],
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part=self.part,
                   modal_frequencies=self.modal_frequencies)


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
        out : FRF class
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
        frequency_start = int(np.around(frequencies[0] / self.resolution))
        frequency_end = int(np.around(frequencies[-1] / self.resolution) + 1)
        return FRF(frf=self.value[frequency_start:frequency_end, :, :],
                   resolution=self.resolution,
                   bandwidth=frequencies[-1] - frequencies[0],
                   max_freq=frequencies[-1],
                   min_freq=frequencies[0],
                   name=self.name,
                   part=self.part,
                   modal_frequencies=self.modal_frequencies)


    def real(self):
        """
        Create a new instance with the real part of the FRFs.

        Returns
        -------
        out : FRF class
            New instance with only the real part of the FRFs.
        """

        return FRF(frf=self.value.real,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part='real',
                   modal_frequencies=self.modal_frequencies)


    def imag(self):
        """
        Create a new instance with the imaginary part of the FRFs.

        Returns
        -------
        out : FRF class
            New instance with only the imaginary part of the FRFs.
        """

        return FRF(frf=self.value.imag,
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part='imag',
                   modal_frequencies=self.modal_frequencies)


    def abs(self):
        """
        Create a new instance with the absolute value of the FRFs.

        Returns
        -------
        out : FRF class
            New instance with only the absolute value of the FRFs.
        """

        return FRF(frf=np.absolute(self.value),
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part='abs',
                   modal_frequencies=self.modal_frequencies)


    def phase(self):
        """
        Create a new instance with the phase of the FRFs.

        Returns
        -------
        out : FRF class
            New instance with only the phase of the FRFs.
        """

        return FRF(frf=np.angle(self.value),
                   resolution=self.resolution,
                   bandwidth=self.bandwidth,
                   max_freq=self.max_freq,
                   min_freq=self.min_freq,
                   name=self.name,
                   part='phase',
                   modal_frequencies=self.modal_frequencies)


    def silhouette(self):
        silhouette = []
        for i in range(len(self)):
            silhouette.append(self.value[:, :, i].max(axis=1))
        silhouette = np.dstack(silhouette)
        return silhouette
    

    def _modal_frequencies(self, prominence=0.02, distance=50):
        modal_frequencies = []
        distance = distance / self.resolution
        for i in range(len(self)):
            peaks = signal.find_peaks(
                np.abs(self.silhouette()[:, 0, i].imag),
                prominence=prominence,
                distance=distance
            )
            modal_frequencies.append(self.freq_vector[peaks[0]].tolist())
        return modal_frequencies


    def mode_shapes(self, prominence=0.02, distance=50):
        mode_shapes = []
        for i in range(len(self)):
            mode_shape = []
            for j in range(self.lines):
                modal_frequencies = self[i].modal_frequencies(prominence,
                                                              distance)
                mode_shape.append(self[i].change_lines(j).imag().value[
                    (modal_frequencies[0] / self.resolution).astype(int), 0, 0
                ])
            mode_shape = np.stack(mode_shape, axis=1)
            mode_shapes.append(mode_shape.transpose())
        return mode_shapes


    def synthesize(self, prominence, distance, alpha, beta):
        # This should be in utils really
        frf_class = []
        for i in range(len(self)):
            omega = self[i].freq_vector * 2 * np.pi
            omega_n = self[i].modal_frequencies(prominence, distance)[0] * 2 * np.pi
            mode_shapes = self[i].mode_shapes(prominence, distance)[0]
            xi = pymodal.damping_coefficient(omega_n, alpha, beta)
            sigma = omega_n * xi
            nf = omega.shape[0]
            nmodes = omega_n.shape[0]
            eigvals1 = sigma + 1j*omega_n
            eigvals2 = sigma - 1j*omega_n

            modal_participation = np.ones(nmodes)
            modal_participation[13] = 10000
            frf=[]
            for j in range(self.lines):
                ResMod = (mode_shapes[j, :]*mode_shapes[0, :]) * modal_participation
                Res1 = np.tile(omega ** 2, (nmodes, 1)) * np.tile(ResMod, (nf, 1)).T
                Res2 = Res1.conj()
                Den1 = np.tile(1j*omega, (nmodes, 1)) - np.tile(eigvals1.T, (nf, 1)).T
                Den2 = np.tile(1j*omega, (nmodes, 1)) - np.tile(eigvals2.T, (nf, 1)).T
                frf.append(np.sum((Res1/Den1) + (Res2/Den2), axis=0) * 10**-3)
            frf = np.stack(frf, axis=1)
            frf_class.append(frf)
        frf = np.dstack(frf_class)
        return pymodal.FRF(frf, resolution = 0.5)
    
    # The intention for the following code was to estimate the modal properties
    # by using the normal identification estimation method. I'm halting it
    # because I believe this can only be done with a full transfer function
    # matrix, or else extensive theorization and testing is required.
    # Equations 10-13 of http://sci-hub.st/10.1016/j.ymssp.2008.02.001 are to
    # be used, but re-formulated in vector form, not in matrix form
    # def get_transformation_matrix(self):
    #     G = []
    #     for i in range(len(self)):
    #         G.append(-np.dot(self[i].imag().value[:, :, 0], 
    #                          np.linalg.inv(self[i].real().value[:, :, 0])))
    #     G = np.dstack(G)
    #     return G


    # def get_normal_FRF(self):
    #     H_N = []
    #     for i in range(len(self)):
    #         G = self[i].get_transformation_matrix()[:, :, 0]
    #         real = self[i].real().value[:, :, 0]
    #         imag = self[i].imag().value[:, :, 0]
    #         H_N.append(real + np.dot(-G, imag))
    #     H_N = np.dstack(H_N)
    #     return H_N


    # def get_matrices(self):
    #     # Code by user Ben at https://stackoverflow.com/questions/31292374/how-do-i-put-2-matrix-into-scipy-optimize-minimize
    #     def toVector(w, z):
    #         assert w.shape == (self.lines, self.lines)
    #         assert z.shape == (self.lines, self.lines)
    #         return np.hstack([w.flatten(), z.flatten()])

    #     def toWZ(vec):
    #         assert vec.shape == (2 * self.lines * self.lines,)
    #         return vec[:2*4].reshape(2,4), vec[2*4:].reshape(2,4)

    #     def doOptimization(f_of_w_z, w0, z0):
    #         def f(x): 
    #             w, z = toWZ(x)
    #             return f_of_w_z(w, z)

    #         result = optimize.minimize(f, toVector(w0, z0))
    #         # Different optimize functions return their
    #         # vector result differently. In this case it's result.x:
    #         result.x = toWZ(result.x) 
    #         return result
    #     # End of code by user Ben
        
    #     def stiffness_damping(C, D):
    #         np.dot(omega * H_N) 
    #         mse = (np.square(A - B)).mean()

    #     C = []
    #     D = []
    #     M = []
    #     K = []
    #     omega = self.freq_vector * 2 * np.pi
    #     for i in range(len(self)):
    #         H_N = self[i].get_normal_FRF()[:, :, 0]
    #         G = self[i].get_transformation_matrix()[:, :, 0]



    def get_CFDAC(self, ref: int, frf: list = None):
        ref_FRF = self.value[:, :, ref]
        if frf is None:
            CFDAC = pymodal.value_CFDAC(ref_FRF,
                                        self.value[:, :, 0])
            CFDAC.reshape((CFDAC.shape[0], CFDAC.shape[1], -1))
            for i in range(1, len(self)):
                CFDAC = np.dstack(
                    (CFDAC, pymodal.value_CFDAC(ref_FRF, self.value[:, :, i]))
                )
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            CFDAC = pymodal.value_CFDAC(ref_FRF,
                                        self.value[:, :, frf[0]])
            CFDAC.reshape((CFDAC.shape[0], CFDAC.shape[1], -1))
            for i in frf[1:]:
                CFDAC = np.dstack(
                    (CFDAC, pymodal.value_CFDAC(ref_FRF, self.value[:, :, i]))
                )
        return CFDAC

    def get_FDAC(self, ref: int, frf: list = None):
        ref_FRF = self.value[:, :, ref]
        if frf is None:
            FDAC = pymodal.value_FDAC(ref_FRF,
                                        self.value[:, :, 0])
            FDAC.reshape((FDAC.shape[0], FDAC.shape[1], -1))
            for i in range(1, len(self)):
                FDAC = np.dstack(
                    (FDAC, pymodal.value_FDAC(ref_FRF, self.value[:, :, i]))
                )
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            FDAC = pymodal.value_FDAC(ref_FRF,
                                        self.value[:, :, frf[0]])
            FDAC.reshape((FDAC.shape[0], FDAC.shape[1], -1))
            for i in frf[1:]:
                FDAC = np.dstack(
                    (FDAC, pymodal.value_FDAC(ref_FRF, self.value[:, :, i]))
                )
        return FDAC


    def get_SCI(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        SCI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            SCI = np.append(SCI, pymodal.SCI(ref_CFDAC, CFDAC[:, :]))
        return SCI
    

    def get_M2L(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            CFDAC = np.abs(self.get_CFDAC(ref))
        elif part == 'real':
            CFDAC = np.real(self.get_CFDAC(ref))
        elif part == 'imag':
            CFDAC = np.imag(self.get_CFDAC(ref))
        M2L = np.reshape(pymodal.M2L(CFDAC[:, :, 0]), (1, -1))
        for i in range(1, len(self)):
            M2L = np.append(M2L, 
                            np.reshape(pymodal.M2L(CFDAC[:, :, i]), (1, -1)),
                            axis=0)
        return M2L


    def plot(self,
             ax: list = None,
             fontsize: float = 12,
             title: str = 'Frequency Response',
             title_size: float = None,
             major_locator: int = 4,
             minor_locator: int = 4,
             fontname: str = 'serif',
             color: list = None,
             ylabel: str = None,
             bottom_ylim: float = None,
             decimals_y: int = 1,
             decimals_x: int = 1):
        """
        Plot the FRFs in the instance one over the other with
        varying colors unless otherwise specified.

        Parameters
        ----------
        ax : list of axes, optional
            The axes upon which the FRFs are to be plotted.

        fontsize : float, optional, default: 12
            Size of all text in the figure.

        title : str, optional, default: 'Frequency Response'
            The axes upon which the FRFs are to be plotted.

        titlesize : float, optional
            Size of the title of the figure, if different from the rest.

        major_locator : int, optional, default: 4
            How many major divisions should the axes present.

        minor_locator : int, optional, default: 4
            How many minor divisions should the axes present.

        fontname : str, optional, default: serif
            The font for all text in the figures.

        color : list of strings, optional
            A list of desired colors for each FRF. Defaults to blue or random
            different colors if there is more than one FRF in a figure.

        ylabel : str, optional
            Label for the y axis, defaults to accelerance.

        bottom_ylim : float, optional
            Inferior limit for the plot. Defaults to four powers of ten
            lower than the average value of the FRF.

        Returns
        -------
        out : list
            List of matplotlib axes.
        """
        
        # DEV NOTE: In the futuree, it would be nice to generate as many color
        # samples as needed from the chosen color map, default cubehelix.
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

        # Make sure ax is a list of axes as long as there are FRFs stored.
        if ax is None:
            ax = [plt.gca()]
        elif isinstance(ax, np.ndarray):
            ax = ax.flatten()
        try:
            ax = list(ax)
        except Exception as __:  # noqa F841
            ax = [ax]
        if color is None:
            color = []
            for i in range(len(ax)):
                color.append('b')
            #! DEV NOTE: Substitute for extend
            for i in range(len(color), len(self)):  # Add enough colors
                color.append(color_list[i - len(color) + 1])
            color = list(color_list[0:len(self) + 1])
        else:
            if isinstance(color, str):
                color = [color]
            color = list(color)  # Make sure color is a list
            #! DEV NOTE: Substitute for extend
            for i in range(len(color), len(self)):  # Add enough colors
                color.append(color_list[i - len(color) + 1])
        for _ in range(len(ax), len(self)):
            ax.append(plt.gca())

        plot_values = self.abs() if self.part == 'complex' else self
        img = []
        for i in range(len(self)):
            img.append(pymodal.plot_FRF(frf=plot_values[i].value[:, :, 0],
                                        freq=self.freq_vector,
                                        ax=ax[i],
                                        fontsize=fontsize,
                                        title=title,
                                        title_size=title_size,
                                        major_locator=major_locator,
                                        minor_locator=minor_locator,
                                        fontname=fontname,
                                        color=color[i],
                                        ylabel=ylabel,
                                        decimals_y=decimals_y,
                                        decimals_x=decimals_x,
                                        bottom_ylim=bottom_ylim,
                                        part=self.part))
        return img


    def time_domain(self):
        value_time_domain = []
        for i in range(len(self)):
            frf_time_domain = []
            for j in range(self.lines):
                frf_time_domain.append(np.fft.irfft(self.value[:, j, i]))
            value_time_domain.append(np.array(frf_time_domain))
        return np.array(value_time_domain).T


    # def save(self, path: str, decimal_places: int = None):
    def save(self, path: pathlib.PurePath, decimals: int = None):
        """
        Save the FRF object to a zip file.

        Parameters
        ----------
        path : pathlib PurePath object
            Path where the object is to be saved.

        decimals : int, optional
            Amount of decimals to be written to the file at most.

        Returns
        -------
        out : pathlib PurePath object
            Path where the object was saved.
        """
        
        path = pathlib.Path(str(path))

        frf_value = (self.value if decimals is None
                     else np.around(self.value, decimals))
        file_list = []
        for i in range(len(self)):
            file_list.append(path.parent / f'{self.name[i]}.npz')
            pymodal.save_array(frf_value[:, :, i], file_list[i])
        data = {'resolution': self.resolution,
                'bandwidth': self.bandwidth,
                'max_freq': self.max_freq,
                'min_freq': self.min_freq,
                'name': self.name,
                'part': self.part,
                'modal_frequencies': self.modal_frequencies}
        file_list.append(path.parent / 'data.json')
        with open(path.parent / 'data.json', 'w') as fh:
            json.dump(data, fh)

        with ZipFile(path, 'w') as fh:
            for item in file_list:
                fh.write(item, item.name)
                item.unlink()

        return path
