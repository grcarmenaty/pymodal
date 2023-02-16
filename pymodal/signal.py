import numpy as np
from warnings import warn
from typing import Optional
import numpy.typing as npt
import pymodal


class _signal:
    def __init__(
        self,
        measurements: npt.NDArray[npt.complex64],
        coordinates: npt.NDArray[npt.float64] = None,
        orientations: npt.NDArray[npt.float64] = None,
        units: Optional[str] = None,
        system_type: str = "SIMO",
    ):
        assert system_type in ["MISO", "SIMO", "MIMO"]
        self.measurements = np.asarray(measurements)
        if self.measurements.ndim < 3:
            for _ in range(3 - self.measurements.ndim):
                self.measurements = [..., np.newaxis]
        if system_type is "SIMO":
            self.measurements.reshape((self.measurements.shape[0], -1, 1))
        elif system_type is "MISO":
            self.measurements.reshape((self.measurements.shape[0], 1, -1))
        else:
            assert system_type == "MIMO"
            assert self.measurements.shape[1] == self.measurements.shape[2]
        self.dof = max(self.measurements.shape[1], self.measurements.shape[2])
        if coordinates is None and orientations is None:
            warn(
                "Coordinates will be assumed to be points spaced one distance unit"
                " along the x axis.",
                UserWarning,
            )
            self.coordinates = np.vstack(np.arange(self.dof), np.zeros((self.dof, 2)))
            warn(
                "orientations will be assumed to be unit vectors on the z axis.",
                UserWarning,
            )
            self.orientations = np.vstack(np.zeros((self.dof, 2)), np.ones(self.dof))
        elif coordinates is None:
            self.orientations = np.asarray(orientations)
            warn(
                "Coordinates will be assumed to be points spaced one distance unit"
                " along the x axis.",
                UserWarning,
            )
            self.coordinates = np.vstack(np.arange(self.dof), np.zeros((self.dof, 2)))
        elif orientations is None:
            self.coordinates = np.asarray(coordinates)
            warn(
                "Coordinates will be assumed to be points spaced one distance unit"
                " along the x axis.",
                UserWarning,
            )
            self.orientations = np.vstack(np.arange(self.dof), np.zeros((self.dof, 2)))
        else:
            self.coordinates = np.asarray(coordinates)
            self.orientations = np.asarray(orientations)
        combination = np.hstack((self.coordinates, self.orientations))
        unq, cnt = np.unique(combination, axis=0)
        assert np.all(cnt == 1)
        cnt = cnt[0]
        if system_type is "SIMO":
            assert self.measurements.shape[1] == self.dof
            assert self.measurements.shape[2] == 1
        elif system_type is "MISO":
            assert self.measurements.shape[1] == 1
            assert self.measurements.shape[2] == self.dof
        elif system_type is "MIMO":
            assert self.measurements.shape[1] == self.dof
            assert self.measurements.shape[2] == self.dof
            self.coordinates = np.tile(self.coordinates, (1, 1, self.dof))
            self.orientations = np.tile(self.orientations, (1, 1, self.dof))
        if units is None:
            self.units = "mm, kg, s, °C"
            print("Units will be assumed to be mm, kg, s, °C.")
        else:
            self.units = units
        self.units = units
        self.samples = self.measurements.shape[0]

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
