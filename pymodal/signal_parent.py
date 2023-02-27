import numpy as np
from warnings import warn
from typing import Optional
import numpy.typing as npt
import pymodal
from pint import UnitRegistry
from copy import deepcopy


ureg = UnitRegistry()


class _signal():
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
        space_units: Optional[str] = None,
        system_type: str = "SIMO",
    ):
        # Measurement checks
        self.system_type = system_type
        assert self.system_type in ["MISO", "SIMO", "MIMO", "excitation"]
        if measurements_units is None:
            if system_type == "excitation":
                measurements_units = ureg.parse_expression("newton")
            else:
                measurements_units = ureg.parse_expression("millimeter/second**2")
        elif measurements_units is str:
            self.units = ureg.parse_expression(measurements_units)
        self.units = measurements_units
        self.measurements = np.asarray(measurements) * self.units
        if self.measurements.ndim < 3:
            for _ in range(3 - self.measurements.ndim):
                self.measurements = self.measurements[..., np.newaxis]
        if self.system_type == "SIMO":
            self.measurements = self.measurements.reshape(
                (self.measurements.shape[0], -1, 1)
            )
        elif self.system_type == "MISO" or self.system_type == "excitation":
            self.measurements = self.measurements.reshape(
                (self.measurements.shape[0], 1, -1)
            )
        else:
            assert self.system_type == "MIMO"
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
            self.coordinates = np.asarray(coordinates)
            self.orientations = np.asarray(orientations)
        self.orientations = (self.orientations.T / np.linalg.norm(self.orientations, axis=1)).T
        if space_units is None:
            space_units = ureg.parse_expression("millimeter")
        elif space_units is str:
            self.units = ureg.parse_expression(space_units)
        self.space_units = space_units
        self.coordinates = self.coordinates * self.space_units
        combination = np.hstack((np.asarray(self.coordinates), self.orientations))
        _, cnt = np.unique(combination, axis=0, return_counts=True)
        assert np.all(cnt == 1)
        cnt = cnt[0]
        if self.system_type == "SIMO":
            assert self.measurements.shape[1] == self.dof
            assert self.measurements.shape[2] == 1
        elif self.system_type == "MISO" or self.system_type == "excitation":
            assert self.measurements.shape[1] == 1
            assert self.measurements.shape[2] == self.dof
        elif self.system_type == "MIMO":
            assert self.measurements.shape[1] == self.dof
            assert self.measurements.shape[2] == self.dof
            self.coordinates = np.tile(self.coordinates, (1, 1, self.dof))
            self.orientations = np.tile(self.orientations, (1, 1, self.dof))

        self.samples = self.measurements.shape[0]

        # Create domain array
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
        return self.dof

    def __eq__(self, other):
        if isinstance(other, pymodal.signal):
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
        self_copy = deepcopy(self)
        if type(key) is int:
            key = slice(key, key + 1)
        if type(key) is slice:
            key = [key]
        key = list(key)
        for i, index in enumerate(key):
            if type(index) is int:
                key[i] = slice(index, index + 1)
        if len(key) == 1:
            if self.system_type in ["SIMO", "MIMO"]:
                self_copy.measurements = self.measurements[:, key[0], :]
                self_copy.coordinates = self.coordinates[:, key[0]]
                self_copy.orientations = self.orientations[:, key[0]]
            elif self.system_type in ["MISO", "excitation"]:
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

    def change_domain_resolution(self, new_resolution):
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
            self_copy.domain_resolution, np.average(np.diff(self_copy.domain_array, axis=0))
        )
        return self_copy
