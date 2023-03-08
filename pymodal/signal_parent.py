import numpy as np
from warnings import warn, catch_warnings, filterwarnings
from typing import Optional
import numpy.typing as npt
import pymodal
from pint import UnitRegistry
from copy import deepcopy


ureg = UnitRegistry()


class _signal:
    def __init__(
        self,
        measurements: npt.NDArray[np.complex64],
        coordinates: Optional[npt.NDArray[np.float64]] = None,
        orientations: Optional[npt.NDArray[np.float64]] = None,
        dof: Optional[float] = None,
        domain_start: Optional[float] = 0,
        domain_end: Optional[float] = None,
        domain_span: Optional[float] = None,
        domain_resolution: Optional[float] = None,
        measurements_units: Optional[str] = None,
        space_units: Optional[str] = "millimeter",
        method: str = "SIMO",
        label: Optional[str] = None,
    ):
        """This class is intended as a parent class for any class involving the handling
        of a series of measurements relating to spatial coordinates, along a temporal
        domain quantifying the rate of change of the measured quantity.

        Parameters
        ----------
        measurements : numpy array of complexes
            A numpy array of up to three dimensions where the first one contains the
            measurements as they change along the temporal domain, and the rest are
            related to the system's degrees of freedom and the obtention method.
        coordinates : numpy array of floats, optional
            A two-dimensional array containing the spatial coordinates of the degrees of
            freedom of the measurements contained within the instance of this class,
            repeating as needed if measurements were taken for more than one orientation
            on the same spatial coordinates, by default None.
        orientations : numpy array of floats, optional
            A two dimensional array containing a unit vector representing the direction
            in which the measurement taken at a given coordinate was recorded, by
            default None.
        dof : float, optional
            How many degrees of freedom have been measured and are stored within the
            instance of this class, by default None.
        domain_start : float, optional
            Starting value of the temporal domain, by default 0.
        domain_end : float, optional
            Maximum value of the temporal domain, by default None.
        domain_span : float, optional
            Total duration of the temporal domain, by default None.
        domain_resolution : float, optional
            Temporal domain quantity between two consecutive measurement points, by
            default None.
        measurements_units : string, optional
            Units used for the measurements stored within the instance of this class,
            they are assumed to be Newtons, millimeters and seconds; taking "Newton" as
            the default for an excitation and "millimiter / second ** 2" as default for
            any output measurement, by default None.
        space_units : string, optional
            Units used for the spatial coordinates of the degrees of freedom, by default
            "millimeter"
        method : string, optional
            Whether the method used to get the measurements is Multiple Input Single
            Output (MISO), Single Input Multiple Output (SIMO), Multiple Input Multiple
            Output (MIMO), or a recording of the excitation inputs, by default "SIMO"
        label : string, optional
            An identifying label for the measurements stored in this instance of the
            signal class.
        """
        self.label = label
        # Measurement checks
        self.method = method
        assert self.method in ["MISO", "SIMO", "MIMO", "excitation"]
        # Set units to millimeter, second, Newton if not defined, else parse if a string
        # is given or set units to
        # whatever the user has setup.
        if measurements_units is None:
            if method == "excitation":
                measurements_units = ureg.parse_expression("newton")
            else:
                measurements_units = ureg.parse_expression("millimeter/second**2")
        elif type(measurements_units) is str:
            measurements_units = ureg.parse_expression(measurements_units)
        self.measurements_units = measurements_units
        self.measurements = np.asarray(measurements) * self.measurements_units
        # Make sure the measurements array is three-dimensional array.
        if self.measurements.ndim < 3:
            for _ in range(3 - self.measurements.ndim):
                self.measurements = self.measurements[..., np.newaxis]
        # Make sure the shape of the measurements array is coherent with the type of
        # system the user has specified:
        # - First axis is for the domain dimension.
        # - Second axis is for referencing output.
        # - Third axis is for referencing input.
        if self.method == "SIMO":
            self.measurements = self.measurements.reshape(
                (self.measurements.shape[0], -1, 1)
            )
        elif self.method == "MISO":
            self.measurements = self.measurements.reshape(
                (self.measurements.shape[0], 1, 1)
            )
        elif self.method == "excitation":
            self.measurements = self.measurements.reshape(
                (self.measurements.shape[0], 1, -1)
            )
        else:
            assert self.method == "MIMO"
            assert self.measurements.shape[1] == self.measurements.shape[2]
        if dof is None:
            self.dof = max(self.measurements.shape[1], self.measurements.shape[2])
        else:
            self.dof = dof
        # Coordinates and orientations checks
        if coordinates is None and orientations is None:
            warn(
                "Coordinates will be assumed to be points spaced one distance unit"
                " along the x axis.",
                UserWarning,
            )
            self.coordinates = np.vstack(
                (np.arange(self.dof).T, np.zeros((self.dof, 2)).T)
            ).T
            warn(
                "Orientations will be assumed to be unit vectors on the z axis.",
                UserWarning,
            )
            self.orientations = np.vstack(
                (np.zeros((self.dof, 2)).T, np.ones(self.dof).T)
            ).T
        elif coordinates is None:
            self.orientations = np.asarray(orientations)
            warn(
                "Coordinates will be assumed to be points spaced one distance unit"
                " along the x axis.",
                UserWarning,
            )
            self.coordinates = np.vstack(
                (np.arange(self.dof).T, np.zeros((self.dof, 2))).T
            ).T
        elif orientations is None:
            self.coordinates = np.asarray(coordinates)
            warn(
                "Coordinates will be assumed to be points spaced one distance unit"
                " along the x axis.",
                UserWarning,
            )
            self.orientations = np.vstack(
                (np.arange(self.dof).T, np.zeros((self.dof, 2)).T).T
            )
        else:
            if isinstance(coordinates, type(self.measurements)):
                with catch_warnings():
                    filterwarnings(
                        "ignore",
                        message="The unit of the quantity is stripped when downcasting"
                        " to ndarray.",
                    )
                    self.coordinates = np.asarray(coordinates)
            else:
                self.coordinates = np.asarray(coordinates)
            self.orientations = np.asarray(orientations)
        # Normalize orientations
        self.orientations = (
            self.orientations.T / np.linalg.norm(self.orientations, axis=1)
        ).T
        # Assign space units to coordinates.
        if type(space_units) is str:
            space_units = ureg.parse_expression(space_units)
        self.space_units = space_units
        self.coordinates = self.coordinates * self.space_units
        # Make sure coordinates-orientations pairs are unique and both them and
        # measurements' shapes are coherent with
        # the system type specified by the user.
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting to"
                " ndarray.",
            )
            combination = np.hstack((np.asarray(self.coordinates), self.orientations))
        _, cnt = np.unique(combination, axis=0, return_counts=True)
        assert np.all(cnt == 1)
        cnt = np.sum(cnt)
        if self.method == "SIMO":
            assert self.measurements.shape[1] == self.dof
            assert self.measurements.shape[2] == 1
            assert cnt == self.dof
        elif self.method == "MISO":
            assert self.measurements.shape[1] == 1
            assert self.measurements.shape[2] == 1
            assert cnt == 1
        elif self.method == "excitation":
            assert self.measurements.shape[1] == 1
            assert self.measurements.shape[2] == self.dof
            assert cnt == self.dof
        elif self.method == "MIMO":
            assert self.measurements.shape[1] == self.dof
            assert self.measurements.shape[2] == self.dof
            self.coordinates = np.tile(self.coordinates, (1, 1, self.dof))
            self.orientations = np.tile(self.orientations, (1, 1, self.dof))
            assert cnt == self.dof
        self.samples = self.measurements.shape[0]
        # Make sure domain parameters are coherent, calculate the missing domain
        # parameters
        self.domain_start = float(domain_start)
        self.domain_end = float(domain_end) if domain_end is not None else domain_end
        self.domain_span = (
            float(domain_span) if domain_span is not None else domain_span
        )
        self.domain_resolution = (
            float(domain_resolution)
            if domain_resolution is not None
            else domain_resolution
        )
        if self.domain_span is None:
            if self.domain_end is None:  # max, span are None, rate is defined
                if self.domain_resolution is None:
                    raise ValueError("Insufficient temporal domain parameters.")
                self.domain_span = (self.samples - 1) * self.domain_resolution
                self.domain_end = self.domain_start + self.domain_span
            else:
                self.domain_span = self.domain_end - self.domain_start
                if self.domain_resolution is None:
                    self.domain_resolution = self.domain_span / (self.samples - 1)
        else:  # max is defined, span is not, rate not considered
            self.domain_span = self.domain_end - self.domain_start
            if self.domain_resolution is None:  # span, rate are None, max is defined
                self.domain_resolution = self.domain_span / (self.samples - 1)
            else:  # span is None, max and rate are defined
                calculated_resolution = self.domain_span / (self.samples - 1)
                if not np.allclose(self.domain_resolution, calculated_resolution):
                    raise ValueError(
                        "The temporal domain parameters introduced are inconsistent."
                    )
                else:
                    if self.domain_end is None:  # max is None, span is defined.
                        self.domain_end = self.domain_start + self.domain_span
                        if (
                            self.domain_resolution is None
                        ):  # max and rate are None, span is defined
                            self.domain_resolution = self.domain_span / (
                                self.samples - 1
                            )
                        else:  # span is None, max and rate are defined
                            calculated_resolution = self.domain_span / (
                                self.samples - 1
                            )
                            if not np.allclose(
                                self.domain_resolution, calculated_resolution
                            ):
                                raise ValueError(
                                    "The temporal domain parameters introduced are"
                                    " inconsistent."
                                )
                    else:  # max and span are defined
                        calculated_end = self.domain_start + self.domain_span
                        if not np.allclose(self.domain_end, calculated_end):
                            raise ValueError(
                                "The temporal domain parameters introduced are"
                                " inconsistent."
                            )
                        if (
                            self.domain_resolution is None
                        ):  # rate is None, max and span are defined
                            self.domain_resolution = self.domain_span / (
                                self.samples - 1
                            )
                        else:  # everything is defined
                            calculated_resolution = self.domain_span / (
                                self.samples - 1
                            )
                            if not np.allclose(
                                self.domain_resolution, calculated_resolution
                            ):
                                raise ValueError(
                                    "The temporal domain parameters introduced are"
                                    " inconsistent."
                                )
        # Build the domain array and make sure it is coherent with the measurements.
        self.domain_array = np.arange(
            self.domain_start,
            self.domain_end + self.domain_resolution / 2,
            self.domain_resolution,
        )
        if not np.allclose(len(self.domain_array), self.samples):
            raise ValueError(
                "The temporal domain parameters introduced are inconsistent."
            )
        assert np.allclose(
            self.domain_resolution, np.average(np.diff(self.domain_array, axis=0))
        )

    def __len__(self):
        """This method describes what happens when len(_signal class object) is invoked.

        Returns
        -------
        float
            The amount of data points of the measurements.
        """
        return self.samples

    def __eq__(self, other):
        """This method describes what happens when an instance of this class is compared
        with another object to ascertain whether or not they are equal.

        Parameters
        ----------
        other
            An object of any type whose equality relative to a given instance of this
            class needs to be ascertained.

        Returns
        -------
        boolean
            Whether or not this instance of the signal class and the object against
            which its equality was being ascertained are, in fact, equal.
        """
        if isinstance(other, pymodal._signal):
            own_dict = dict(self.__dict__)
            own_arrays = []
            key_list = list(own_dict.keys())
            # Separate all arrays in current instance to a list
            for key in key_list:
                if isinstance(own_dict[key], np.ndarray):
                    own_arrays.append(own_dict[key])
                    del own_dict[key]
            other_dict = dict(other.__dict__)
            other_arrays = []
            key_list = list(other_dict.keys())
            # Separate all arrays in instance being compared to a list
            for key in key_list:
                if isinstance(other_dict[key], np.ndarray):
                    other_arrays.append(other_dict[key])
                    del other_dict[key]
            # Determine if arrays are equal
            equal_array = len(own_arrays) == len(other_arrays)
            for i, own_array in enumerate(own_arrays):
                if not (equal_array):
                    break
                equal_array = np.array_equal(own_array, other_arrays[i]) and equal_array
            # Instances are equal if both array and non-array parts are equal.
            return own_dict == other_dict and equal_array
        else:
            return False

    def __getitem__(self, key: tuple[slice]):
        self_copy = deepcopy(self)  # Make a deepcopy of self to work on it.
        # Make sure key is a list of slices. If it isn't, turn it into one.
        if type(key) is int:
            key = slice(key, key + 1)
        if type(key) is slice:
            key = [key]
        key = list(key)
        for i, index in enumerate(key):
            if type(index) is int:
                key[i] = slice(index, index + 1)
        # If only one key is provided, it is assumed to refer to an output selection,
        # unless the system type is supposed to have only one input, in which case it
        # will be assumed to refer to an input selection. If two keys are provided, the
        # first one is assumed to refer to an output, the second to an input.
        if len(key) == 1:
            if self.method in ["SIMO", "MIMO"]:
                self_copy.measurements = self.measurements[:, key[0], :]
                self_copy.coordinates = self.coordinates[:, key[0]]
                self_copy.orientations = self.orientations[:, key[0]]
            elif self.method in ["MISO", "excitation"]:
                self_copy.measurements = self.measurements[:, :, key[0]]
                self_copy.coordinates = self.coordinates[:, key[0]]
                self_copy.orientations = self.orientations[:, key[0]]
        elif len(key) == 2:
            self_copy.measurements = self.measurements[:, key[0], key[1]]
            self_copy.coordinates = self.coordinates[:, key[0], key[1]]
            self_copy.orientations = self.orientations[:, key[0], key[1]]
        else:
            raise ValueError("Too many keys provided.")
        return self_copy

    def change_domain_resolution(self, new_resolution: float):
        """Change the temporal resolution of an array of signals, assuming the temporal
        dimension of said signal is the first dimension of the array.

        Parameters
        ----------
        new_resolution : float
            The desired distance between any two adjacent values of the domain array.

        Returns
        -------
        _signal class object
            A hard copy of the class instance with the modifications pertinent to the
            method applied: a new temporal domain resolution and the data points
            corresponding to the new domain array.
        """
        new_domain_array, new_measurements_array = pymodal.change_domain_resolution(
            domain_array=self.domain_array,
            measurements_array=self.measurements,
            new_resolution=new_resolution,
        )
        self_copy = deepcopy(self)
        self_copy.measurements = new_measurements_array
        self_copy.domain_start = new_domain_array[0]
        self_copy.domain_end = new_domain_array[1]
        self_copy.domain_span = new_domain_array[1] - new_domain_array[0]
        self_copy.domain_resolution = new_resolution
        return self_copy

    def change_domain_span(
        self,
        new_min_domain: Optional[float] = None,
        new_max_domain: Optional[float] = None,
    ):
        """Change the span of the temporal domain of an array of signals, assuming the
        temporal dimension of said signal is the first dimension of the array.

        Parameters
        ----------
        new_min_domain : float, optional
            The desired new minimum value for the domain array, by default None.
        new_max_domain : float, optional
            The desired new maximum value for the domain array, by default None.

        Returns
        -------
        _signal class object
            A hard copy of the class instance with the modifications pertinent to the
            method applied: the new domain array without the values that fall outside
            the given range, and extended as necessary to comply with the given range,
            with the corresponding measurements values.
        """
        cut_domain_array, cut_measurements_array = pymodal.change_domain_span(
            domain_array=self.domain_array,
            measurements_array=self.measurements,
            new_min_domain=new_min_domain,
            new_max_domain=new_max_domain,
        )
        self_copy = deepcopy(self)
        self_copy.measurements = cut_measurements_array
        self_copy.domain_start = cut_domain_array[0]
        self_copy.domain_end = cut_domain_array[1]
        self_copy.domain_span = cut_domain_array[1] - cut_domain_array[0]
        self_copy.domain_resolution = self.domain_resolution
        self_copy.domain_array = cut_domain_array
        assert np.allclose(
            self_copy.domain_resolution,
            np.average(np.diff(self_copy.domain_array, axis=0)),
        )
        return self_copy
