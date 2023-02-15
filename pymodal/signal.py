import numpy as np
from warnings import warn
from typing import Optional
import numpy.typing as npt
import pymodal
from .utils import _check_coordinates_orientations


class _signal:
    def __init__(
        self,
        measurements: npt.NDArray[np.complex64],
        coordinates: npt.NDArray[np.float64] = None,
        orientations: npt.NDArray[np.float64] = None,
        units: Optional[str] = None,
    ):
        self.measurements = np.asarray(measurements)
        self.dof = max(self.measurements.shape[1], self.measurements.shape[2])
        (
            self.coordinates,
            self.orientations,
            self.dof,
            matrix_completion,
        ) = _check_coordinates_orientations(
            coordinates=coordinates,
            orientations=orientations,
            dof=self.dof
        )
        if matrix_completion == 0:
            assert self.measurements.shape[1] == self.measurements.shape[2]
        if matrix_completion == 1:
            assert self.measurements.shape[1] == self.dof
            assert self.measurements.shape[2] == 1
        if matrix_completion == 2:
            assert self.measurements.shape[2] == self.dof
            assert self.measurements.shape[1] == 1
        if units is None:
            self.units = "mm, kg, s, °C"
            warn("Units will be assumed to be mm, kg, s, °C.", UserWarning)
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
