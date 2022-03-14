import numpy as np
from warning import warn
import pymodal

class signal():
    """
    Store and manipulate spectral signals.
    """

    def __init__(self,
                 amplitude: np.ndarray,
                 sample_rate: float = None,
                 elapsed_time: float = None,
                 max_time: float = None,
                 min_time: float = 0,
                 label: list = None,
                 coordinates: list[tuple[float, float, float]] = None,
                 orientation: list[tuple[float, float, float]] = None,
                 units: str = "N·mm/s^2"):

        self.amplitude = amplitude
        self.units = units
        self.samples = self.amplitude.shape[0]

        self.label = label
        if self.label is None:
            self.label = []
        for i in range(self.samples):
            self.label.append(f"signal {i}")
        
        self.min_time = min_time  # Minimum frequency is assumed to be 0
        if elapsed_time is None:
            if max_time is None:
                # If neither elapsed_time nor maximum frequency are defined
                self.sample_rate = sample_rate  # Then sample_rate MUST be defined
                # So elapsed_time and maximum frequency can be calculated
                self.elapsed_time = (self.samples - 1) * self.sample_rate
                self.max_time = self.min_time + self.elapsed_time
            else:  # If elapsed_time is not defined but maximum frequency is
                self.max_time = max_time
                self.elapsed_time = self.max_time - self.min_time
                if sample_rate is None:
                    self.sample_rate = (self.elapsed_time /
                                       (self.samples - 1))
                else:
                    self.sample_rate = sample_rate
        else:
            self.elapsed_time = elapsed_time
            if max_time is None:
                self.max_time = self.min_time + self.elapsed_time
                if sample_rate is None:
                    self.sample_rate = (self.elapsed_time /
                                       (self.data_points - 1))
                else:
                    self.sample_rate = sample_rate
            else:
                self.max_time = max_time
                if sample_rate is None:
                    self.sample_rate = (self.elapsed_time /
                                       (self.data_points - 1))
                else:
                    self.sample_rate = sample_rate

        # In case the user inputs more values than is necessary and those
        # values don't make sense together, raise an exception.
        calculated_sample_rate = self.elapsed_time / (self.data_points - 1)
        sample_rate_error = not(self.sample_rate == calculated_sample_rate)
        calculated_elapsed_time = self.max_time - self.min_time
        elapsed_time_error = not(self.elapsed_time == calculated_elapsed_time)
        if sample_rate_error or elapsed_time_error:
            raise Exception((
                f"The sample_rate ({sample_rate} Hz), elapsed_time ({elapsed_time} s)"
                f", min_time ({min_time} s) and/or max_time ({max_time} s)"
                f" values introduced were not coherent with each other.\n\n"
                f"elapsed_time = (data_points - 1) * sample_rate\n"
                f"max_time = min_time + elapsed_time"
            ))
        
        self.time_vector = np.arange(
            self.min_time, self.max_time + self.sample_rate / 2, self.sample_rate
        )

        self.coordinates = coordinates
        if self.coordinates is None:
            warn("You will not be able to convert this to frequential.")
        else:
            for element in coordinates:
                if len(element) != 3:
                    raise Exception("At least one coordinate has too many elements.")
                if len(self.coordinates) > len(self):
                    raise Exception("Too many coordinates were provided.")

        if len(self.label) > len(self):
            raise Exception("Too many names were provided.")

        self.orientation = orientation
        if self.orientation is None:
            warn("You will not be able to convert this to frequential.")
        else:
            for element in coordinates:
                if len(element) != 3:
                    raise Exception("At least one orientation has too many elements.")
                if len(self.orientation) > len(self):
                    raise Exception("Too many orientations were provided.")

    def __len__(self):
        return self.amplitude.shape[1]


    def __repr__(self):
        return f"{self.__class__} ({self.__dict__})"


    def __eq__(self, other):
        if isinstance(other, pymodal.signal):
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
        return signal(amplitude = self.amplitude[:, index.start:index.stop:index.step],
                      sample_rate = self.sample_rate,
                      elapsed_time = self.elapsed_time,
                      max_time = self.max_time,
                      min_time = self.min_time,
                      label = self.label,
                      coordinates = self.coordinates[index.start:index.stop:index.step],
                      orientation = self.orientation[index.start:index.stop:index.step],
                      units = self.units)