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
from astropy import stats
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
            Warning("The modal frequencies will now be approximated from the"
                    " observed peaks in the signal. Take this with a grain of"
                    " salt.")
            self.modal_frequencies = list(self._modal_frequencies())
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
    
    def generate_transmissibility(self):
        """
        Create a new instance of the FRF converting to transmissibility matrix

        Returns
        -------
        out: FRF class (Transmissibility matrix)

        Notes
        -----
        Frequencies must be a two-element iterable object.
        """

        count = 0
        frf = np.asarray(self.value[:,:,:])

        for output_h in range(frf.shape[1]-1):
            input_h = output_h + 1
            while input_h <= frf.shape[1]-1:
                if count == 0:
                    T = frf[:,output_h,:]/frf[:,input_h,:]
                    T = np.reshape(T, (np.shape(frf)[0],1,-1))
                    count = 1
                    input_h = input_h + 1

                else:
                    t = frf[:,output_h,:]/frf[:,input_h,:]
                    t = np.reshape(t, (np.shape(frf)[0],1,-1))
                    T = np.hstack((T,t))
                    input_h = input_h + 1

        T = pymodal.FRF(T, min_freq = self.min_freq, max_freq = self.max_freq)
        return T


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
            modal_frequencies.append(self.freq_vector[peaks[0]])
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
        """
        Create a covariance matrix of two different sets of FRF's.
        Marco A. Pérez and Roger Serra-López. A frequency domain-based correlation approach for structural assessment and damage identification.
        Mech. Syst. Signal Process
        DOI: https://doi.org/10.1016/j.ymssp.2018.09.042

        Returns
        -------
        out : CFDAC matrix
            Correlation matrix in the complex domain. The magnitude of the CFDAC corresponds to the FDAC.
        """
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

    def get_FIAC(self, ref: int, frf: list = None):
        ref_FRF = self.value[:, :, ref]
        if frf is None:
            FIAC = pymodal.value_FIAC(ref_FRF,self.value[:, :, 0])
            FIAC.reshape((FIAC.shape[0], FIAC.shape[1], -1))
            for i in range(1, len(self)):FIAC = np.dstack((FIAC, pymodal.value_FIAC(ref_FRF, self.value[:, :, i])))
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            FIAC = pymodal.value_FIAC(ref_FRF,self.value[:, :, frf[0]])
            FIAC.reshape((FIAC.shape[0], FIAC.shape[1], -1))
            for i in frf[1:]:
                FIAC = np.dstack((FIAC, pymodal.value_FIAC(ref_FRF, self.value[:, :, i])))
        return FIAC

    def get_CFDAC_1d(self, ref: int, frf: list = None):
        ref_FRF = self.value[:, :, ref]
        if frf is None:
            CFDAC = pymodal.value_CFDAC_1d(ref_FRF, self.value[:, :, 0], self.resolution)
            CFDAC.reshape((CFDAC.shape[0], CFDAC.shape[1], -1))
            for i in range(1, len(self)):
                CFDAC = np.dstack((CFDAC, pymodal.value_CFDAC_1d(ref_FRF, self.value[:, :, i],self.resolution))
                )
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            CFDAC = pymodal.value_CFDAC_1d(ref_FRF,self.value[:, :, frf[0]],self.resolution)
            CFDAC.reshape((CFDAC.shape[0], CFDAC.shape[1], -1))
            for i in frf[1:]:
                CFDAC = np.dstack((CFDAC, pymodal.value_CFDAC_1d(ref_FRF, self.value[:, :, i],self.resolution))
                )
        return CFDAC

    def get_CFDAC_2d(self, ref: int, frf: list = None):
        ref_FRF = self.value[:, :, ref]
        if frf is None:
            CFDAC = pymodal.value_CFDAC_2d(ref_FRF, self.value[:, :, 0], self.resolution)
            CFDAC.reshape((CFDAC.shape[0], CFDAC.shape[1], -1))
            for i in range(1, len(self)):
                CFDAC = np.dstack((CFDAC, pymodal.value_CFDAC_2d(ref_FRF, self.value[:, :, i],self.resolution))
                )
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            CFDAC = pymodal.value_CFDAC_2d(ref_FRF,self.value[:, :, frf[0]],self.resolution)
            CFDAC.reshape((CFDAC.shape[0], CFDAC.shape[1], -1))
            for i in frf[1:]:
                CFDAC = np.dstack((CFDAC, pymodal.value_CFDAC_2d(ref_FRF, self.value[:, :, i],self.resolution))
                )
        return CFDAC

    def get_full_CFDAC(self, ref: int,frf: list = None ):
        ref_FRF = self.value[:, :, ref]
        if frf is None:
            full_CFDAC = pymodal.value_full_CFDAC(ref_FRF,
                                        self.value[:, :, 0])
            full_CFDAC.reshape((full_CFDAC.shape[0], full_CFDAC.shape[1], -1))
            for i in range(1, len(self)):
                full_CFDAC = np.dstack(
                    (full_CFDAC, pymodal.value_full_CFDAC(ref_FRF, self.value[:, :, i]))
                )
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            full_CFDAC = pymodal.value_full_CFDAC(ref_FRF,
                                        self.value[:, :, frf[0]])
            full_CFDAC.reshape((full_CFDAC.shape[0], full_CFDAC.shape[1], -1))
            for i in frf[1:]:
                full_CFDAC = np.dstack(
                    (full_CFDAC, pymodal.value_full_CFDAC(ref_FRF, self.value[:, :, i]))
                )
        return full_CFDAC

    def get_CFDAC_A(self, ref: int, frf: list = None):
        ref_FRF = self.value[:, :, ref]
        if frf is None:
            CFDAC_A = pymodal.value_CFDAC_A(ref_FRF,
                                        self.value[:, :, 0])
            CFDAC_A.reshape((CFDAC_A.shape[0], CFDAC_A.shape[1], -1))
            for i in range(1, len(self)):
                CFDAC_A = np.dstack(
                    (CFDAC_A, pymodal.value_CFDAC_A(ref_FRF, self.value[:, :, i]))
                )
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            CFDAC_A = pymodal.value_CFDAC_A(ref_FRF,
                                        self.value[:, :, frf[0]])
            CFDAC_A.reshape((CFDAC_A.shape[0], CFDAC_A.shape[1], -1))
            for i in frf[1:]:
                CFDAC_A = np.dstack(
                    (CFDAC_A, pymodal.value_CFDAC_A(ref_FRF, self.value[:, :, i]))
                )
        return CFDAC_A

    def get_FDAC(self, ref: int, frf: list = None):
        """
        Create a covariance matrix of two different sets of FRF's.
        Rodrigo Pascual, J.-C Golinval, and Mario Razeto. A Frequency Domain Correlation Technique for Model Correlation and Updating. 
        Proc. Int. Modal Anal. Conf. - IMAC, 1, 1997
        
        Returns
        -------
        out : FDAC matrix
            Correlation matrix in absolute terms.
        """
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

    def get_RVAC(self, ref: int, frf: list = None):
        """
        Extracts correlation vector at each frequency pair, equivalent to the main diagonal of the FDAC.
        Ward Heylen, S Lammens, and Paul Sas. Modal Analysis Theory and Testing. 1997.
        
        Notes:
        RVAC function is equivalent to the Global Shape Criterion (GSC) and the Cross Signature Assurance Criterion (CSAC).
        
        Returns
        -------
        out : RVAC vector
            Correlation vector in absolute terms.
        """

        ref_FRF = self.value[:, :, ref]
        if frf is None:
            RVAC = pymodal.value_RVAC(ref_FRF,self.value[:, :, 0])
            RVAC.reshape((RVAC.shape[0], RVAC.shape[1], -1))
            for i in range(1, len(self)):
                RVAC = np.dstack((RVAC, pymodal.value_RVAC(ref_FRF, self.value[:, :, i])))
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            RVAC = pymodal.value_RVAC(ref_FRF,self.value[:, :, frf[0]])
            for i in frf[1:]:
                RVAC = np.dstack((RVAC, pymodal.value_RVAC(ref_FRF, self.value[:, :, i])))
        return RVAC

    def get_RVAC_2d(self, ref: int, frf: list = None):
        """
        Extracts correlation vector at each frequency pair, using the FRF curvature (Second derivative of the FRF).
        R.P.C. Sampaio and N. M.M. Maia. Strategies for an effcient indicator of structural damage. 
        Mech. Syst. Signal Process
        DOI: https://doi.org/10.1016/j.ymssp.2008.07.015
        
        Returns
        -------
        out : RVAC'' vector
            Correlation vector in absolute terms.
        """
        ref_FRF = self.value[:, :, ref]
        if frf is None:
            RVAC_2d = pymodal.value_RVAC_2d(ref_FRF,self.value[:, :, 0])
            for i in range(1, len(self)):
                RVAC_2d = np.dstack((RVAC_2d, pymodal.value_RVAC_2d(ref_FRF, self.value[:, :, i])))
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            RVAC_2d = pymodal.value_RVAC_2d(ref_FRF,self.value[:, :, frf[0]])
            for i in frf[1:]:
                RVAC_2d = np.dstack((RVAC_2d, pymodal.value_RVAC_2d(ref_FRF, self.value[:, :, i])))
        return RVAC_2d

    def get_GAC(self, ref: int, frf: list = None):
        """
        Extracts correlation vector at each frequency pair, this method is sensitive to amplitude changes.
        C. Zang, H. Grafe, and M. Imregun. Frequency-domain criteria for correlating and updating dynamic finite element models. 
        Mech Syst Signal Process
        DOI: https://doi.org/10.1006/mssp.2000.1357
        
        Returns
        -------
        out : GAC vector
            Correlation vector in absolute terms, sensitive to amplitude changes.
        """
        ref_FRF = self.value[:, :, ref]
        if frf is None:
            GAC = pymodal.value_GAC(ref_FRF,self.value[:, :, 0])
            GAC.reshape((GAC.shape[0], GAC.shape[1], -1))
            for i in range(1, len(self)):
                GAC = np.dstack((GAC, pymodal.value_GAC(ref_FRF, self.value[:, :, i])))
        else:
            if isinstance(frf, slice):
                frf = list(range(frf.start, frf.stop, frf.step))
            else:
                try:
                    frf = list(frf)
                except Exception as __:
                    frf = [frf]
            GAC = pymodal.value_GAC(ref_FRF,self.value[:, :, frf[0]])
            #RVAC.reshape((RVAC.shape[0], RVAC.shape[1], -1))
            for i in frf[1:]:
                GAC = np.dstack((GAC, pymodal.value_GAC(ref_FRF, self.value[:, :, i])))
        return GAC

    def get_SCI(self, ref: int, part: str = 'abs'):
        """
        Scalar damage indicator of two different CFDAC's (Pristine-Pristine) and (Pristine-Damage).
        This damage indicator is based on the Pearson Correlation.
        Marco A. Pérez and Roger Serra-López. A frequency domain-based correlation approach for structural assessment and damage identification.
        Mech. Syst. Signal Process
        DOI: https://doi.org/10.1016/j.ymssp.2018.09.042

        Returns
        -------
        out : SCI value
            Scalar value between 0 and 1. 0 indicating perfect correlation and 1 completely uncorrelated.
        """
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

    def get_CCC(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        CCC = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            CCC = np.append(CCC, pymodal.CCC(ref_CFDAC, CFDAC[:, :]))
        return CCC

    def get_wSCI(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        wSCI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            wSCI = np.append(wSCI, pymodal.wSCI(ref_CFDAC, CFDAC[:, :]))
        return wSCI

    def get_BMC(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_full_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_full_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_full_CFDAC(0))
        BMC = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_full_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_full_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_full_CFDAC(ref, i))
            BMC = np.append(BMC, stats.biweight_midcorrelation(ref_CFDAC.flatten(), CFDAC.flatten(), 9))
        return BMC

    def get_DICE(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        DICE = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            DICE = np.append(DICE, pymodal.DICE(ref_CFDAC, CFDAC[:, :]))
        return DICE

    def get_JACCARD(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        JACCARD = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            JACCARD = np.append(JACCARD, pymodal.JACCARD(ref_CFDAC, CFDAC[:, :]))
        return JACCARD

    def get_OVERLAP(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        OVERLAP = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            OVERLAP = np.append(OVERLAP, pymodal.OVERLAP(ref_CFDAC, CFDAC[:, :]))
        return OVERLAP

    def get_robust_SCI(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_full_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_full_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_full_CFDAC(0))
        SCI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_full_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_full_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_full_CFDAC(ref, i))
            SCI = np.append(SCI, pymodal.robust_SCI(ref_CFDAC, CFDAC[:, :]))
        return SCI

    def get_SSI(self, ref: int, part: str = 'abs'):
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
            SCI = np.append(SCI, pymodal.SSI(ref_CFDAC, CFDAC[:, :]))
        return SCI

    def get_FBC(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        FBC = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            FBC = np.append(FBC, pymodal.FBC(ref_CFDAC, CFDAC[:, :]))
        return FBC

    def get_FDC(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        FDC = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            FDC = np.append(FDC, pymodal.FDC(ref_CFDAC, CFDAC[:, :]))
        return FDC

    def get_SSIM(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        SSIM = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            SSIM = np.append(SSIM, pymodal.SSIM(ref_CFDAC, CFDAC[:, :]))
        return SSIM

    def get_SSIM_FCFDAC(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_full_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_full_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_full_CFDAC(0))
        SSIM = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_full_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_full_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_full_CFDAC(ref, i))
            SSIM = np.append(SSIM, pymodal.SSIM(ref_CFDAC, CFDAC[:, :]))
        return SSIM

    def get_FICI(self, ref: int):
        ref_FIAC = np.abs(self[ref].get_FIAC(0))
        FICI = np.array([])
        for i in range(len(self)):
            FIAC = np.abs(self.get_FIAC(ref, i))
            FICI = np.append(FICI, pymodal.FSI(ref_FIAC, FIAC[:, :]))
        return FICI

    def get_SCI_1d(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC_1d(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC_1d(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC_1d(0))
        SCI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC_1d(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC_1d(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC_1d(ref, i))
            SCI = np.append(SCI, pymodal.SCI(ref_CFDAC, CFDAC[:, :]))
        return SCI

    def get_SCI_2d(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC_2d(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC_2d(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC_2d(0))
        SCI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC_2d(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC_2d(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC_2d(ref, i))
            SCI = np.append(SCI, pymodal.SCI(ref_CFDAC, CFDAC[:, :]))
        return SCI

    def get_similarity(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_full_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_full_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_full_CFDAC(0))
        sim_index = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_full_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_full_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_full_CFDAC(ref, i))
            sim_index = np.append(sim_index, pymodal.similarity(ref_CFDAC, CFDAC[:, :]))
        return sim_index

    def get_full_SCI(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_full_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_full_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_full_CFDAC(0))
        SCI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_full_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_full_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_full_CFDAC(ref, i))
            SCI = np.append(SCI, pymodal.SCI(ref_CFDAC, CFDAC[:, :]))
        return SCI

    def get_FSI(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_full_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_full_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_full_CFDAC(0))
        FSI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_full_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_full_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_full_CFDAC(ref, i))
            FSI = np.append(FSI, pymodal.FSI(ref_CFDAC, CFDAC[:, :]))
        return FSI

    def get_FQI(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_full_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_full_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_full_CFDAC(0))
        FQI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_full_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_full_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_full_CFDAC(ref, i))
            FQI = np.append(FQI, pymodal.FQI(ref_CFDAC, CFDAC[:, :]))
        return FQI

    def get_FDC(self, ref:int, part: str='abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_full_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_full_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_full_CFDAC(0))
        FDC = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_full_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_full_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_full_CFDAC(ref, i))
            FDC = np.append(FDC, np.mean(ref_CFDAC - CFDAC))
        return FDC  

    def get_SI(self, ref: int, part: str = 'abs'):
        SI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_full_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_full_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_full_CFDAC(ref, i))
            SI = np.append(SI, pymodal.SI(CFDAC[:, :]))
        return SI     

    def get_FSI_CFDAC(self, ref: int, part: str = 'abs'):
        if part == 'abs':
            ref_CFDAC = np.abs(self[ref].get_CFDAC(0))
        elif part == 'real':
            ref_CFDAC = np.real(self[ref].get_CFDAC(0))
        elif part == 'imag':
            ref_CFDAC = np.imag(self[ref].get_CFDAC(0))
        FSI = np.array([])
        for i in range(len(self)):
            if part == 'abs':
                CFDAC = np.abs(self.get_CFDAC(ref, i))
            elif part == 'real':
                CFDAC = np.real(self.get_CFDAC(ref, i))
            elif part == 'imag':
                CFDAC = np.imag(self.get_CFDAC(ref, i))
            FSI = np.append(FSI, pymodal.FSI(ref_CFDAC, CFDAC[:, :]))
        return FSI

    def get_CSM(self, ref:int, part: str='abs'):
        CSM = np.array([])
        print(range(len(self)))
        for i in range(len(self)):
            if part == 'abs':
                pri = np.abs(self.value[:,:,ref])
                frf = np.abs(self.value[:,:,i])
            if part == 'real':
                pri = np.real(self.value[:,:,ref])
                frf = np.real(self.value[:,:,i])                
            if part == 'imag':
                pri = np.imag(self.value[:,:,ref])
                frf = np.imag(self.value[:,:,i])    
 
            num = np.sum(pri*frf,axis=0)
            print('Numerador')
            print(num)
            den = np.sqrt(np.sum(pri**2,axis=0))*np.sqrt(np.sum(frf**2,axis=0))
            print('Denominador')
            print(den)
            CSM_value = np.mean(num/den)  
            print(CSM_value)
            CSM = np.append(CSM, CSM_value)

        return CSM

    def get_DRQ(self, ref:int):
        """
        Scalar damage indicator, arithmetic average of the RVAC vector.
        R.P.C. Sampaio and N. M.M. Maia. Strategies for an effcient indicator of structural damage. 
        Mech. Syst. Signal Process
        DOI: https://doi.org/10.1016/j.ymssp.2008.07.015
        
        Notes:
        The DRQ damage indicator is equivalent to the AIGSC based on the Global Shape Criterion (GSC)

        Returns
        -------
        out : DRQ value
            Scalar value between 0 and 1. 1 meaning perfect correlation.
        """
        DRQ = np.array([])
        for i in range(len(self)):
            RVAC = self.get_RVAC(ref,i)
            DRQ = np.append(DRQ, pymodal.DRQ(RVAC))
        return DRQ

    def get_AIGAC(self, ref:int):
        """
        Scalar damage indicator, arithmetic average of the GAC vector.
        C. Zang, H. Grafe, and M. Imregun. Frequency-domain criteria for correlating and updating dynamic finite element models. 
        Mech Syst Signal Process
        DOI: https://doi.org/10.1006/mssp.2000.1357
        
        Returns
        -------
        out : AIGAC value
            Scalar value between 0 and 1. 1 meaning perfect correlation.
        """
        AIGAC = np.array([])
        for i in range(len(self)):
            GAC = self.get_GAC(ref,i)
            AIGAC = np.append(AIGAC, pymodal.AIGAC(GAC))
        return AIGAC
    
    def get_DRQ_2d(self, ref:int):
        """
        Scalar damage indicator, arithmetic average of the RVAC'' vector.
        R.P.C. Sampaio and N. M.M. Maia. Strategies for an effcient indicator of structural damage. 
        Mech. Syst. Signal Process
        DOI: https://doi.org/10.1016/j.ymssp.2008.07.015

        Returns
        -------
        out : DRQ'' value
            Scalar value between 0 and 1. 1 meaning perfect correlation.
        """
        DRQ_2d = np.array([])
        for i in range(len(self)):
            RVAC_2d = self.get_RVAC_2d(ref,i)
            DRQ_2d = np.append(DRQ_2d, pymodal.DRQ(RVAC_2d))
        return DRQ_2d

    def get_FRFSF(self, ref:int):
        """
        Ratio between pristine and damaged spectrums.
        Timothy Marinone and Adam Moya. Comparison of frf correlation techniques. 
        Conf. Proc. Soc. Exp. Mech. Ser

        Returns
        -------
        out : FRFSF value
            Unbounded damage indicator.
        """
        FRFSF = np.array([])        
        for i in range(len(self)):
            FRFSF_value = np.sum(np.abs(self.value[:,:,ref]))/np.sum(np.abs(self.value[:,:,i]))
            FRFSF = np.append(FRFSF, FRFSF_value)
        return FRFSF

    def get_FRFRMS(self, ref:int):
        """
        Dennis Göge and Michael Link. Assessment of computational model updating procedures with regard to model validation. 
        Aero. Sci. Tech.
        DOI: https://doi.org/10.1016/S1270-9638(02)01193-8

        Returns
        -------
        out : FRFRMS value
            Unbounded damage indicator.
        """
        FRFRMS = np.array([])
        for i in range(len(self)):
            num = np.nan_to_num((np.log10(np.abs(self.value[:,:,i]))-np.log10(np.abs(self.value[:,:,ref])))**2)
            den = np.nan_to_num((np.log10(np.abs(self.value[:,:,ref])))**2)
            FRFRMS_value = np.nan_to_num(np.sqrt(np.sum(num/den)))
            FRFRMS = np.append(FRFRMS, FRFRMS_value)
        return FRFRMS



    def get_FRFSM(self, ref:int, std:int):
        """
        Damage indicator based on the Probability Density Function (PDF)
        Dooho Lee, Tae Soo Ahn, and Hyeon Seok Kim. A metric on the similarity between two frequency response functions. 
        J. Sound Vib.
        DOI: https://doi.org/10.1016/j.jsv.2018.08.051
        
        Notes:
        The argument std is selected by the operator based on his/her expertise.
        As a rule of thumb, this parameter can be set to 6 dB.

        Returns
        -------
        out : FRFSM value
            Bounded between 0 and 1. 1 meaning perfect correlation.
        """
        FRFSM = np.array([])
        for i in range(len(self)):
            pristine = np.sum(np.abs(self.value[:,:,ref]), axis=1)**2
            frf = np.sum(np.abs(self.value[:,:,i]), axis=1)**2
            
            ej = np.nan_to_num(np.abs(10*np.log10(pristine)-10*np.log10(frf)))
            f = 1/(std*np.sqrt(2*np.pi))*np.exp(-(1/2)*((ej-0)/std)**2)
            f0 = 1/(std*np.sqrt(2*np.pi))
            s = 1/len(pristine)*np.sum(f/f0)
            FRFSM = np.append(FRFSM, s)
        return FRFSM

    def get_ODS_diff(self, ref:int):
        """
        Damage indicator based on Operating Deflection Shapes (delta_ODS)
        R.P.C. Sampaio, N. M.M. Maia, R. A.B. Almeida, and A. P.V. Urgueira. A simple damage detection indicator using operational deflection shapes. 
        Mech Syst Signal Process
        DOI: https://doi.org/10.1016/j.ymssp.2015.10.023

        Returns
        -------
        out : delta_ODS value
            Unbounded damage indicator
        """
        sm = np.array([])
        for i in range(len(self)):
            sm_value = np.sum(np.abs(self.value[:,:,i] - self.value[:,:,ref]))
            sm = np.append(sm, sm_value)
            ODS = 1 - sm/max(sm)
        
        return sm

    def ODS_diff (ref:np.ndarray, frf:np.ndarray):
        sm_value = np.abs(frf-ref)
        sm_value = np.sum(sm_value)
        return sm_value

    def get_r2_imag(self, ref:int):
        """
        Coefficient of Determination (R2) based on the imaginary part of the FRF.
        Timothy Marinone and Adam Moya. Comparison of frf correlation techniques. Conf. Proc. Soc. Exp. Mech. Ser

        Returns
        -------
        out : R2 value
            Unbounded damage indicator
        """
        r2_imag = np.array([])
        for i in range(len(self)):
            pristine = np.reshape(np.imag(self.value[:,:,ref]),-1)
            frf = np.reshape(np.imag(self.value[:,:,i]),-1)
            sstot = np.sum((pristine - np.mean(pristine))**2)
            ssres = np.sum((pristine - frf)**2)
            r2_value = 1-(ssres/sstot)
            r2_imag = np.append(r2_imag, r2_value)
        return r2_imag

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

    def time_domain(self):
        value_time_domain = []
        for i in range(len(self)):
            frf_time_domain = []
            for j in range(self.lines):
                frf_time_domain.append(np.fft.irfft(self.value[:, j, i]))
            value_time_domain.append(np.array(frf_time_domain))
        return np.array(value_time_domain).T

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
            print(data)
            print('fh')
            print(fh)
            json.dump(data, fh)
        

        with ZipFile(path, 'w') as fh:
            for item in file_list:
                fh.write(item, item.name)
                item.unlink()

        return path
    
    def implicit_pca(self,
                     damage_existence: np.ndarray,
                     eovs: np.ndarray,
                     train_proportion: float,
                     n_pcs: int,
                     discarded_pcs: np.ndarray,
                     print_results: bool = False,
                     save_results: bool = True,
                     plot_results: bool = False):

        """
            EOV Mitigation using Implicit PCA.
            This method discards a subset of Principal Components from the overall. 
            The main reason is that EOPs are more significantly present in the first PCs.
            Parameters
            ----------
            psd : 3D array
                Collection of Power Spectral Density arrays.
            damage_existence : 1D array (rows should correspond to different observations)
                0 equals undamaged conditions and 1 equals damaged conditions.
            eovs : 1D or 2D array (rows should correspond to different observations)
                Information regarding the environmental and/or operational conditions of a given observation.
            train_proportion : float
                Value between 0 and 1, which splits data between training and test.
            n_pcs : int
                Number of Principal Components extracted from the Power Spectral Density array.
            discarded_pcs : list
                List of PCs discarded
            print_results : bool
                Option to print results from EOV Mitigation Procedure on terminal.
            save_results : bool
                Option to save results from EOV Mitigation Procedure in a .csv file
            plot_results : bool
                Plots helpful to explain the results

            Returns
            ----------
            damage_index: list
                List of damage indexes for each PSD.
        """

        data = self.value
        labels = np.hstack((eovs,damage_existence))

        #Flatten Power Spectral Density matrices if more than one accelerometer is used.
        data_flatten = data.reshape(-1,np.shape(data)[2])
        data_flatten = data_flatten.T

        data_u = []
        data_d = []
        labels_u = []
        labels_d = []

        #Split dataset between undamaged and damaged observations
        print('--------------FOR------------')
        for i in range(np.shape(labels)[0]):
            if damage_existence[i] == 1: #Undamaged
                if data_u == []:
                    data_u = data_flatten[i,:]
                    labels_u = labels[i]
                else:
                    data_u = np.vstack((data_u, data_flatten[i,:]))
                    labels_u = np.vstack((labels_u, labels[i,:]))
            elif damage_existence[i] == 0: #Damaged
                if data_d == []:
                    data_d = data_flatten[i,:]
                    labels_d = labels[i]
                else:
                    data_d = np.vstack((data_d, data_flatten[i,:]))
                    labels_d = np.vstack((labels_d, labels[i,:]))


        data_train, data_u_test, labels_train, labels_u_test = train_test_split(data_u,
                                                                                labels_u, 
                                                                                train_size = train_proportion, 
                                                                                random_state = 42,
                                                                                stratify = labels_u[:,:-1])

        data_test = np.vstack((data_u_test, data_d))
        labels_test = np.vstack((labels_u_test, labels_d))

        dataset = np.vstack((data_train, data_test))
        labels_dataset = np.vstack((labels_train, labels_test))

        #Use training data to establish normalization
        scaler = MinMaxScaler()
        scaler.fit(data_train)

        #Normalize data
        data_train_scaled = scaler.transform(data_train)
        data_test_scaled = scaler.transform(data_test)

        #Apply Principal Components Analysis (PCA)
        pca = PCA(n_components=n_pcs)
        #Use training data to fit PCA, then apply to transform both datasets
        #Training and Testing
        pca.fit(data_train_scaled)

        train_pca = pca.transform(data_train_scaled)
        test_pca = pca.transform(data_test_scaled)
        dataset_pca = np.vstack((train_pca, test_pca))

        #Discard PCs which have high correlation with EOVs.
        train_pca = np.delete(train_pca, discarded_pcs, 1)
        test_pca = np.delete(test_pca, discarded_pcs, 1)
        dataset_pca = np.delete(dataset_pca, discarded_pcs, 1)

        #Mahalanobis Distance
        sigma = np.linalg.inv(np.cov(train_pca.T))

        di_array = []

        for observation in range(np.shape(dataset_pca)[0]):
            d = dataset_pca[observation].reshape(1,-1)@sigma@dataset_pca[observation].reshape(-1,1)

            if di_array == []:
                di_array = d
            else:
                di_array = np.append(di_array,d)

        if save_results == True:
            np.savetxt("di_implicit_pca_npcs_"+str(n_pcs)+".csv",di_array,delimiter=",")

        #Calculate F1 Score
        di_train = di_array[0:len(train_pca)]

        threshold = np.percentile(di_train,q=95)

        y_pred = di_array < threshold
        y_pred = y_pred.astype(int)

        u = 0
        fa = 0
        ud = 0
        d = 0

        damage_existence = labels_dataset[:,-1]
        colors = []

        for i in range(np.shape(dataset_pca)[0]):
            if y_pred[i] == 1 and damage_existence[i] == 1:
                u = u + 1
                colors.append('green')
            elif y_pred[i] == 1 and damage_existence[i] == 0:
                ud = ud + 1
                colors.append('blue')
            elif y_pred[i] == 0 and damage_existence[i] == 1:
                fa = fa + 1
                colors.append('orange')
            elif y_pred[i] == 0 and damage_existence[i] == 0:
                d = d + 1
                colors.append('red')

        np.savetxt("y_pred.csv",y_pred,delimiter=",")
        np.savetxt("di_array.csv",di_array,delimiter=",")
        np.savetxt("damage_existence.csv",damage_existence,delimiter=",")

        f1 = f1_score(damage_existence, y_pred)

        if print_results == True:
            print('IMPLICIT PCA')
            print('---DATA---')
            print('How many PCs are used? ' + str(n_pcs))
            print('How many PCs have been discarded? ' + str(len(discarded_pcs)))

            print('---PREDICTIONS---')
            print('Undamaged:' + str(u))
            print('False Alarm:' + str(fa))
            print('Unnoticed Damagae:' + str(ud))
            print('Damage:' + str(d))

            print('---PERFORMANCE---')
            print('F1 Score: ' + str(f1))

        if plot_results ==True:
            pymodal.plot_control_chart(di_array,di_train,threshold,colors,"Implicit PCA",n_pcs)

        return di_array,f1,threshold

    def explicit_pca_reg(self,
                         damage_existence: np.ndarray,
                         eovs: np.ndarray,
                         train_proportion: float,
                         n_pcs: int,
                         discarded_pcs: np.ndarray,
                         max_order: int,
                         print_results: bool = False,
                         save_results: bool = True,
                         plot_results: bool = False):

        """
            EOV Mitigation using Explicit PCA.
            This method uses EOVs measured to generate polynomial regression models.
            Parameters
            ----------
            psd : 3D array
                Collection of Power Spectral Density arrays.
            damage_existence : 1D array (rows should correspond to different observations)
                0 equals undamaged conditions and 1 equals damaged conditions.
            eovs : 1D or 2D array (rows should correspond to different observations)
                Information regarding the environmental and/or operational conditions of a given observation.
            train_proportion : float
                Value between 0 and 1, which splits data between training and test.
            n_pcs : int
                Number of Principal Components extracted from the Power Spectral Density array.
            discarded_pcs : list
                List of PCs discarded
            max_order : int
                Highest order for the Polynomial Regression model to try.
            print_results : bool
                Option to print results from EOV Mitigation Procedure on terminal.
            save_results : bool
                Option to save results from EOV Mitigation Procedure in a .csv file
            plot_results : bool
                Plots helpful to explain the results

            Returns
            ----------
            damage_index: list
                List of damage indexes for each PSD.
        """

        data = self.value
        labels = np.hstack((eovs,damage_existence))

        #Flatten Power Spectral Density matrices if more than one accelerometer is used.
        data_flatten = data.reshape(-1,np.shape(data)[2])
        data_flatten = data_flatten.T

        data_u = []
        data_d = []
        labels_u = []
        labels_d = []

        #Split dataset between undamaged and damaged observations
        print('--------------FOR------------')
        for i in range(np.shape(labels)[0]):
            if damage_existence[i] == 1: #Undamaged
                if data_u == []:
                    data_u = data_flatten[i,:]
                    labels_u = labels[i]
                else:
                    data_u = np.vstack((data_u, data_flatten[i,:]))
                    labels_u = np.vstack((labels_u, labels[i,:]))
            elif damage_existence[i] == 0: #Damaged
                if data_d == []:
                    data_d = data_flatten[i,:]
                    labels_d = labels[i]
                else:
                    data_d = np.vstack((data_d, data_flatten[i,:]))
                    labels_d = np.vstack((labels_d, labels[i,:]))

        data_train, data_u_test, labels_train, labels_u_test = train_test_split(data_u,
                                                                                labels_u, 
                                                                                train_size = train_proportion, 
                                                                                random_state = 42,
                                                                                stratify = labels_u[:,:-1])

        data_test = np.vstack((data_u_test, data_d))
        labels_test = np.vstack((labels_u_test, labels_d))

        dataset = np.vstack((data_train, data_test))
        labels_dataset = np.vstack((labels_train, labels_test))

        #Use training data to establish normalization
        scaler = MinMaxScaler()
        scaler.fit(data_train)

        #Normalize data
        data_train_scaled = scaler.transform(data_train)
        data_test_scaled = scaler.transform(data_test)
        data_validation_scaled = scaler.transform(data_u_test)

        #Apply Principal Components Analysis (PCA)
        pca = PCA(n_components=n_pcs)
        #Use training data to fit PCA, then apply to transform both datasets
        #Training and Testing
        pca.fit(data_train_scaled)

        train_pca = pca.transform(data_train_scaled)
        test_pca = pca.transform(data_test_scaled)
        validation_pca = pca.transform(data_validation_scaled) #Used only to validate results from regression model
        dataset_pca = np.vstack((train_pca, test_pca))

        #Discard PCs which have high correlation with EOVs.
        train_pca = np.delete(train_pca, discarded_pcs, 1)
        test_pca = np.delete(test_pca, discarded_pcs, 1)
        validation_pca = np.delete(validation_pca, discarded_pcs, 1)
        dataset_pca = np.delete(dataset_pca, discarded_pcs, 1)

        data_train  = []
        dataset = []

        for eov in range(np.shape(eovs)[1]): #Do not iterate through discarded PCs
            for pca in range(np.shape(train_pca)[1]):

                mse_results = []
                order_regression = []

                X_train = labels_train[:,eov].reshape(-1,1)
                Y_train = train_pca[:,pca].reshape(-1,1)

                X_validation = labels_u_test[:,eov].reshape(-1,1)
                Y_validation = validation_pca[:,pca].reshape(-1,1)

                X_dataset = labels_dataset[:,eov].reshape(-1,1)
                Y_dataset = dataset_pca[:,pca].reshape(-1,1)

                for order in range(max_order):
                    order = order + 1

                    poly_features = PolynomialFeatures(degree=order, include_bias=False)
                    X_train_poly = poly_features.fit_transform(X_train)
                    X_validation_poly = poly_features.transform(X_validation)

                    reg = LinearRegression()

                    reg = reg.fit(X_train_poly, Y_train)

                    X_vals = np.linspace(np.min(X_train), np.max(X_train),100).reshape(-1,1)
                    X_vals_poly = poly_features.transform(X_vals)
                    Y_vals = reg.predict(X_vals_poly)

                    Y_train_pred = reg.predict(X_train_poly)
                    Y_validation_pred = reg.predict(X_validation_poly)

                    mse = mean_squared_error(Y_train, Y_train_pred)

                    mse_results.append(mse)
                    order_regression.append(order)

                best_order = min(mse_results)
                index = mse_results.index(best_order)

                poly_features = PolynomialFeatures(degree=order_regression[index], include_bias = False)
                X_train_poly = poly_features.fit_transform(X_train)

                reg = LinearRegression()
                reg = reg.fit(X_train_poly, Y_train)

                X_dataset_poly = poly_features.transform(X_dataset)
                Y_dataset_pred = reg.predict(X_dataset_poly)

                corrected_train = Y_train - Y_train_pred
                corrected_dataset = Y_dataset - Y_dataset_pred

                if data_train == []:
                    data_train = corrected_train
                    dataset = corrected_dataset
                else:
                    data_train = np.hstack((data_train, corrected_train))
                    dataset = np.hstack((dataset, corrected_dataset))

        #Mahalanobis Distance
        sigma = np.linalg.inv(np.cov(data_train.T))
        di_array = []

        for observation in range(np.shape(dataset_pca)[0]):
            d = dataset[observation].reshape(1,-1) @ sigma @ dataset[observation].reshape(-1,1)

            if di_array == []:
                di_array = d
            else:
                di_array = np.append(di_array,d)

        if save_results == True:
            np.savetxt("di_explicit_pca_reg_npcs_" + str(n_pcs) + ".csv",di_array,delimiter=",")

        #Calculate F1 Score
        di_train = di_array[0:len(train_pca)]

        threshold = np.percentile(di_train,q=95)

        y_pred = di_array < threshold
        y_pred = y_pred.astype(int)

        u = 0
        fa = 0
        ud = 0
        d = 0

        damage_existence = labels_dataset[:,-1]

        for i in range(np.shape(dataset_pca)[0]):
            if y_pred[i] == 1 and damage_existence[i] == 1:
                u = u + 1
            elif y_pred[i] == 1 and damage_existence[i] == 0:
                ud = ud + 1
            elif y_pred[i] == 0 and damage_existence[i] == 1:
                fa = fa + 1
            elif y_pred[i] == 0 and damage_existence[i] == 0:
                d = d + 1

        f1 = f1_score(damage_existence, y_pred)

        if print_results == True:
            print('EXPLICIT PCA REGRESSION')
            print('---DATA---')
            print('How many PCs are used? ' + str(n_pcs))
            print('How many PCs have been discarded? ' + str(len(discarded_pcs)))
            print('Maximum order used in Polynomial Regression Models? ' + str(max_order))

            print('---PREDICTIONS---')
            print('Undamaged:' + str(u))
            print('False Alarm:' + str(fa))
            print('Unnoticed Damagae:' + str(ud))
            print('Damage:' + str(d))

            print('---PERFORMANCE---')
            print('F1 Score: ' + str(f1))

        if plot_results ==True:
            pymodal.plot_control_chart(di_array,di_train,threshold,colors,"Explicit PCA Regression",n_pcs)

        return di_array,f1,threshold,y_pred
    
    def pc_informed_reg(self,
                        damage_existence: np.ndarray,
                        eovs: np.ndarray,
                        train_proportion: float,
                        n_pcs: int,
                        eov_sensitive_pcs: np.ndarray,
                        max_order: int,
                        print_results: bool = False,
                        save_results: bool = True,
                        plot_results: bool = False):

        """
            EOV Mitigation using Principal Component Informed Regression.
            Method proposed in ECCOMAS SMART 2023.
            J.Font-Moré, L.D. Avendano-Valencia, D. Garcia-Cava, M.A. Pérez, "Interpreting
            environmental variability from damage sensitive features"
            X ECCOMAS Thematic Conference on Smart Structures and Materials (SMART 2023)

            In this publication, we proposed a method that uses the so-called EOV-Sensitive PCs 
            as a surrogate of the Environmental and Operational variables driving 
            the non-stanionary behaviour in the DSFs. Hence, a regression model 
            using EOV-Sensitive PCs as predictors and remaining PCs as explained variables.

            Parameters
            ----------
            psd : 3D array
                Collection of Power Spectral Density arrays.
            damage_existence : 1D array (rows should correspond to different observations)
                0 equals undamaged conditions and 1 equals damaged conditions.
            eovs : 1D or 2D array (rows should correspond to different observations)
                Information regarding the environmental and/or operational conditions of a given observation.
            train_proportion : float
                Value between 0 and 1, which splits data between training and test.
            n_pcs : int
                Number of Principal Components extracted from the Power Spectral Density array.
            discarded_pcs : list
                List of PCs discarded
            max_order : int
                Highest order for the Polynomial Regression model to try.
            print_results : bool
                Option to print results from EOV Mitigation Procedure on terminal.
            save_results : bool
                Option to save results from EOV Mitigation Procedure in a .csv file
            plot_results : bool
                Plots helpful to explain the results

            Returns
            ----------
            damage_index: list
                List of damage indexes for each PSD.
        """

        data = self.value
        labels = np.hstack((eovs,damage_existence))

        #Flatten Power Spectral Density matrices if more than one accelerometer is used.
        data_flatten = data.reshape(-1,np.shape(data)[2])
        data_flatten = data_flatten.T

        data_u = []
        data_d = []
        labels_u = []
        labels_d = []

        #Split dataset between undamaged and damaged observations
        print('--------------FOR------------')
        for i in range(np.shape(labels)[0]):
            if damage_existence[i] == 1: #Undamaged
                if data_u == []:
                    data_u = data_flatten[i,:]
                    labels_u = labels[i]
                else:
                    data_u = np.vstack((data_u, data_flatten[i,:]))
                    labels_u = np.vstack((labels_u, labels[i,:]))
            elif damage_existence[i] == 0: #Damaged
                if data_d == []:
                    data_d = data_flatten[i,:]
                    labels_d = labels[i]
                else:
                    data_d = np.vstack((data_d, data_flatten[i,:]))
                    labels_d = np.vstack((labels_d, labels[i,:]))

        data_train, data_u_test, labels_train, labels_u_test = train_test_split(data_u,
                                                                                labels_u, 
                                                                                train_size = train_proportion, 
                                                                                random_state = 42,
                                                                                stratify = labels_u[:,:-1])

        data_test = np.vstack((data_u_test, data_d))
        labels_test = np.vstack((labels_u_test, labels_d))

        dataset = np.vstack((data_train, data_test))
        labels_dataset = np.vstack((labels_train, labels_test))

        #Use training data to establish normalization
        scaler = MinMaxScaler() 
        scaler.fit(data_train)

        #Normalize data
        data_train_scaled = scaler.transform(data_train)
        data_test_scaled = scaler.transform(data_test)
        data_validation_scaled = scaler.transform(data_u_test)

        #Apply Principal Components Analysis (PCA)
        pca = PCA(n_components=n_pcs)
        #Use training data to fit PCA, then apply to transform both datasets
        #Training and Testing
        pca.fit(data_train_scaled)

        train_pca = pca.transform(data_train_scaled)
        test_pca = pca.transform(data_test_scaled)
        validation_pca = pca.transform(data_validation_scaled) #Used only to validate results from regression model
        dataset_pca = np.vstack((train_pca, test_pca))

        data_train  = []
        dataset = []

        for eov in eov_sensitive_pcs: 
            for pca in range(len(eov_sensitive_pcs),np.shape(train_pca)[1]):
                mse_results = []
                order_regression = []

                X_train = train_pca[:,eov].reshape(-1,1)
                Y_train = train_pca[:,pca].reshape(-1,1)

                X_validation = validation_pca[:,eov].reshape(-1,1)
                Y_validation = validation_pca[:,pca].reshape(-1,1)

                X_dataset = dataset_pca[:,eov].reshape(-1,1)
                Y_dataset = dataset_pca[:,pca].reshape(-1,1)

                for order in range(max_order):
                    order = order + 1

                    poly_features = PolynomialFeatures(degree=order, include_bias=False)
                    X_train_poly = poly_features.fit_transform(X_train)
                    X_validation_poly = poly_features.transform(X_validation)

                    reg = LinearRegression()

                    reg = reg.fit(X_train_poly, Y_train)

                    X_vals = np.linspace(np.min(X_train), np.max(X_train),100).reshape(-1,1)
                    X_vals_poly = poly_features.transform(X_vals)
                    Y_vals = reg.predict(X_vals_poly)

                    Y_train_pred = reg.predict(X_train_poly)
                    Y_validation_pred = reg.predict(X_validation_poly)

                    mse = mean_squared_error(Y_train, Y_train_pred)

                    mse_results.append(mse)
                    order_regression.append(order)

                best_order = min(mse_results)
                index = mse_results.index(best_order)

                poly_features = PolynomialFeatures(degree=order_regression[index], include_bias = False)
                X_train_poly = poly_features.fit_transform(X_train)

                reg = LinearRegression()
                reg = reg.fit(X_train_poly, Y_train)

                X_dataset_poly = poly_features.transform(X_dataset)
                Y_dataset_pred = reg.predict(X_dataset_poly)

                corrected_train = Y_train - Y_train_pred
                corrected_dataset = Y_dataset - Y_dataset_pred

                if data_train == []:
                    data_train = corrected_train
                    dataset = corrected_dataset
                else:
                    data_train = np.hstack((data_train, corrected_train))
                    dataset = np.hstack((dataset, corrected_dataset))

        #Mahalanobis Distance
        sigma = np.linalg.inv(np.cov(data_train.T))
        di_array = []

        for observation in range(np.shape(dataset_pca)[0]):
            d = dataset[observation].reshape(1,-1) @ sigma @ dataset[observation].reshape(-1,1)

            if di_array == []:
                di_array = d
            else:
                di_array = np.append(di_array,d)

        if save_results == True:
            np.savetxt("di_explicit_pca_reg_npcs_" + str(n_pcs) + ".csv",di_array,delimiter=",")

        #Calculate F1 Score
        di_train = di_array[0:len(train_pca)]

        threshold = np.percentile(di_train,q=95)

        y_pred = di_array < threshold
        y_pred = y_pred.astype(int)

        u = 0
        ud = 0
        fa = 0
        d = 0

        damage_existence = labels_dataset[:,-1]

        for i in range(np.shape(dataset_pca)[0]):
            if y_pred[i] == 1 and damage_existence[i] == 1:
                u = u + 1
            elif y_pred[i] == 1 and damage_existence[i] == 0:
                ud = ud + 1
            elif y_pred[i] == 0 and damage_existence[i] == 1:
                fa = fa + 1
            elif y_pred[i] == 0 and damage_existence[i] == 0:
                d = d + 1

        f1 = f1_score(damage_existence, y_pred)

        if print_results == True:
            print('PC-INFORMED REGRESSION')
            print('---DATA---')
            print('How many PCs are used? ' + str(n_pcs))
            print('How many PCs have been used as surrogate variable? ' + str(len(eov_sensitive_pcs)))
            print('Maximum order used in Polynomial Regression Models? ' + str(max_order))

            print('---PREDICTIONS---')
            print('Undamaged:' + str(u))
            print('False Alarm:' + str(fa))
            print('Unnoticed Damagae:' + str(ud))
            print('Damage:' + str(d))

            print('---PERFORMANCE---')
            print('F1 Score: ' + str(f1))

        if plot_results ==True:
            pymodal.plot_control_chart(di_array,di_train,threshold,colors,"PC-Informed Regression",n_pcs)

        return di_array,f1,threshold,y_pred
