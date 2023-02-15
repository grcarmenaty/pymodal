import numpy as np
from warnings import warn
from typing import Optional
import numpy.typing as npt

class signal:

    def __init__(
        self,
        measurements: npt.NDArray[np.complex64],
        coordinates: Optional[npt.NDArray[np.float64]] = None,
        orientations: Optional[npt.NDArray[np.float64]] = None,
        units: Optional[str] = None,
    ):
        self.measurements = np.asarray(measurements)
        if self.measurements.ndim > 3:
            raise ValueError("Measurements must be, at most, a three-dimensional array")
        elif self.measurements.ndim < 3:
            for _ in range(3 - self.measurements.ndim):
                self.measurements = [..., np.newaxis]
        if units is None:
            self.units = "mm, kg, s, °C"
            warn("Units will be assumed to be mm, kg, s, °C.", UserWarning)
        else:
            self.units = units
        self.units = units
        self.samples = self.measurements.shape[0]
        self.degrees_of_freedom = self.measurements.shape[1]
        if self.coordinates is None:
            warn(
                "Coordinates will be assumed to be points spaced one distance"
                " unit along the x axis.",
                UserWarning,
            )
            self.coordinates = np.vstack(
                np.arange(self.degrees_of_freedom),
                np.zeros((self.degrees_of_freedom, 2)),
            )
        else:
            self.coordinates = np.array(coordinates, dtype=np.float64)
            # Check for right number of numpy array dimensions.
            if self.coordinates.ndim != 2:
                raise ValueError(
                    f"Coordinates must be a two-dimensional array"
                    f", but it is a {self.coordinates.ndim}-"
                    f"dimensional array."
                )
            # Check for right dimensions of coordinates
            dimension_difference = self.coordinates.shape[1] - 3
            if dimension_difference > 0:
                warn(
                    f"Your coordinates are {self.coordinates.shape[1]}"
                    f"-dimensional, Only first three dimensions of"
                    f" coordinates will be used.",
                    UserWarning,
                )
                self.coordinates = self.coordinates[:, :2]
            elif dimension_difference < 0:
                warn(
                    f"Your coordinates are {self.coordinates.shape[1]}"
                    f"-dimensional, Less than three dimensions provided,"
                    f" missing dimensions are assumed to be 0 for all"
                    f" coordinates.",
                    UserWarning,
                )
                self.coordinates = np.vstack(
                    self.coordinates,
                    np.zeros((self.degrees_of_freedom, abs(dimension_difference))),
                )
            # Check for right amount of coordinates
            if len(self.coordinates) > self.degrees_of_freedom:
                raise ValueError("Too many coordinates were provided.")
            elif len(self.coordinates) < self.degrees_of_freedom:
                raise ValueError("Too few coordinates were provided.")
        if self.orientations is None:
            warn(
                "orientations will be assumed to be unit vectors on the z" " axis.",
                UserWarning,
            )
            self.orientations = np.vstack(
                np.zeros((self.degrees_of_freedom, 2)), np.ones(self.degrees_of_freedom)
            )
        else:
            self.orientations = np.array(orientations, dtype=np.float64)
            # Check for right number of numpy array dimensions.
            if self.orientations.ndim != 2:
                raise ValueError(
                    f"orientations must be a two-dimensional"
                    f" array, but it is a"
                    f" {self.orientations.ndim}-dimensional"
                    f" array."
                )
            # Check for right dimensions of orientations
            dimension_difference = self.orientations.shape[1] - 3
            if dimension_difference > 0:
                warn(
                    f"Your orientations are {self.orientations.shape[1]}"
                    f"-dimensional, Only first three dimensions of"
                    f" orientations will be used.",
                    UserWarning,
                )
                self.orientations = self.orientations[:, :2]
            elif dimension_difference < 0:
                warn(
                    f"Your orientations are {self.orientations.shape[1]}"
                    f"-dimensional, Less than three dimensions provided,"
                    f" missing dimensions are assumed to be 0 for all"
                    f" orientations.",
                    UserWarning,
                )
                self.orientations = np.vstack(
                    self.orientations,
                    np.zeros((self.degrees_of_freedom, abs(dimension_difference))),
                )
            # Check for right amount of orientations
            if len(self.orientations) > self.degrees_of_freedom:
                raise ValueError("Too many orientations were provided.")
            elif len(self.orientations) < self.degrees_of_freedom:
                raise ValueError("Too few orientations were provided.")


