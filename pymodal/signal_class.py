import numpy as np
from warnings import warn
import pymodal
from scipy import interpolate

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
                 coordinates: np.ndarray = None,
                 orientations: np.ndarray = None,
                 units: str = "mm/s^2"):

        self.amplitude = np.array(amplitude)
        self.units = units
        self.samples = self.amplitude.shape[0]
        self.degrees_of_freedom = self.amplitude.shape[1]
        if self.amplitude.ndim > 2:
            raise ValueError(f"Coordinates must be a two-dimensional array"
                             f", but it is a {self.amplitude.ndim}-dimensional"
                             f" array.")

        self.label = label
        # If no label is provided, set it ro "Unnamed label"
        if self.label is None:
            self.label = "Unnamed signal"

        self.min_time = min_time  # Minimum time is assumed to be 0
        if elapsed_time is None:
            if max_time is None:
                # If neither elapsed_time nor maximum time are defined then
                # sample_rate MUST be defined
                self.sample_rate = sample_rate
                # So elapsed_time and maximum time can be calculated
                self.elapsed_time = (self.samples - 1) * self.sample_rate
                self.max_time = self.min_time + self.elapsed_time
            else:  # If elapsed_time is not defined but maximum time is
                self.max_time = max_time  # Max time MUST be defined
                self.elapsed_time = self.max_time - self.min_time
                if sample_rate is None:
                    self.sample_rate = (self.elapsed_time /
                                       (self.samples - 1))
                else:
                    self.sample_rate = sample_rate
        else:  # If elapsed_time is defined
            self.elapsed_time = elapsed_time
            if max_time is None: # but max_time is not.
                self.max_time = self.min_time + self.elapsed_time
                if sample_rate is None: # and sample_rate isn't either
                    self.sample_rate = (self.elapsed_time /
                                       (self.data_points - 1))
                else: # If sample_rate is defined
                    self.sample_rate = sample_rate
            else: # On the other hand, if max_time is defined
                self.max_time = max_time
                if sample_rate is None: # and sample_rate is not
                    self.sample_rate = (self.elapsed_time /
                                       (self.data_points - 1))
                else: # but if sample_rate is defined
                    self.sample_rate = sample_rate

        # In case the user inputs more values than is necessary and those
        # values don't make sense together, raise a value error.
        calculated_sample_rate = self.elapsed_time / (self.data_points - 1)
        sample_rate_error = not(self.sample_rate == calculated_sample_rate)
        calculated_elapsed_time = self.max_time - self.min_time
        elapsed_time_error = not(self.elapsed_time == calculated_elapsed_time)
        if sample_rate_error or elapsed_time_error:
            raise ValueError((
                f"The sample_rate ({sample_rate} Hz), elapsed_time"
                f" ({elapsed_time} s), min_time ({min_time} s) and/or max_time"
                f" ({max_time} s) values introduced were not coherent with"
                f" each other.\n\nelapsed_time = (data_points - 1) *"
                f" sample_rate\nmax_time = min_time + elapsed_time"
            ))
        
        self.time_vector = np.arange(
            self.min_time,
            self.max_time + self.sample_rate / 2,
            self.sample_rate
        )

        if self.coordinates is None:
            warn("Coordinates will be assumed to be points spaced one distance"
                 " unit along the x axis.",
                 UserWarning)
            self.coordinates = np.vstack(
                np.arange(self.degrees_of_freedom),
                np.zeros((self.degrees_of_freedom, 2))
            )
        else:
            self.coordinates = np.array(coordinates, dtype=np.float64)
            # Check for right number of numpy array dimensions.
            if self.coordinates.ndim != 2:
                raise ValueError(f"Coordinates must be a two-dimensional array"
                                 f", but it is a {self.coordinates.ndim}-"
                                 f"dimensional array.")
            # Check for right dimensions of coordinates
            dimension_difference = self.coordinates.shape[1] - 3
            if dimension_difference > 0:
                warn(
                    f"Your coordinates are {self.coordinates.shape[1]}"
                    f"-dimensional, Only first three dimensions of coordinates"
                    f"will be used.",
                    UserWarning
                )
                self.coordinates = self.coordinates[:, :2]
            elif dimension_difference < 0:
                warn(
                    f"Your coordinates are {self.coordinates.shape[1]}"
                    f"-dimensional, Less than three dimensions provided,"
                    f" missing dimensions are assumed to be 0 for all"
                    f" coordinates.",
                    UserWarning)
                self.coordinates = np.vstack(
                    self.coordinates,
                    np.zeros((self.degrees_of_freedom,
                              abs(dimension_difference)))
                )
            # Check for right amount of coordinates
            if len(self.coordinates) > self.degrees_of_freedom:
                raise ValueError("Too many coordinates were provided.")
            elif len(self.coordinates) < self.degrees_of_freedom:
                raise ValueError("Too few coordinates were provided.")

        if self.orientations is None:
            warn("orientations will be assumed to be points spaced one"
                 " distance unit along the x axis.",
                 UserWarning)
            self.orientations = np.vstack(
                np.arange(self.degrees_of_freedom),
                np.zeros((self.degrees_of_freedom, 2))
            )
        else:
            self.orientations = np.array(orientations, dtype=np.float64)
            # Check for right number of numpy array dimensions.
            if self.orientations.ndim != 2:
                raise ValueError(f"orientations must be a two-dimensional"
                                 f" array, but it is a"
                                 f" {self.orientations.ndim}-dimensional"
                                 f" array.")
            # Check for right dimensions of orientations
            dimension_difference = self.orientations.shape[1] - 3
            if dimension_difference > 0:
                warn(
                    f"Your orientations are {self.orientations.shape[1]}"
                    f"-dimensional, Only first three dimensions of"
                    f" orientations will be used.",
                    UserWarning
                )
                self.orientations = self.orientations[:, :2]
            elif dimension_difference < 0:
                warn(
                    f"Your orientations are {self.orientations.shape[1]}"
                    f"-dimensional, Less than three dimensions provided,"
                    f" missing dimensions are assumed to be 0 for all"
                    f" orientations.",
                    UserWarning)
                self.orientations = np.vstack(
                    self.orientations,
                    np.zeros((self.degrees_of_freedom,
                              abs(dimension_difference)))
                )
            # Check for right amount of orientations
            if len(self.orientations) > self.degrees_of_freedom:
                raise ValueError("Too many orientations were provided.")
            elif len(self.orientations) < self.degrees_of_freedom:
                raise ValueError("Too few orientations were provided.")


    def __len__(self):
        return self.degrees_of_freedom


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
        return signal(
            amplitude = self.amplitude[:, index.start:index.stop:index.step],
            sample_rate = self.sample_rate,
            elapsed_time = self.elapsed_time,
            max_time = self.max_time,
            min_time = self.min_time,
            label = self.label[index.start:index.stop:index.step],
            coordinates = self.coordinates[index.start:index.stop:index.step],
            orientations = self.orientations[index.start:index.stop:index.step],
            units = self.units
        )


    def extend(self, new_signal):
        assert self.sample_rate == new_signal.sample_rate
        assert self.min_time == new_signal.min_time
        assert self.max_time == new_signal.max_time
        assert self.units == new_signal.units
        return signal(
            amplitude = np.vstack((self.amplitude, new_signal.amplitude)),
            sample_rate = self.sample_rate,
            elapsed_time = self.elapsed_time,
            max_time = self.max_time,
            min_time = self.min_time,
            label = np.vstack((self.label, new_signal.label)),
            coordinates = np.vstack((self.coordinates,
                                     new_signal.coordinates)),
            orientations = np.vstack((self.amplitude, new_signal.amplitude)),
            units = self.units
        )

    
    def change_sample_rate(self, new_sample_rate: float):
        if new_sample_rate % self.sample_rate != 0:
            warn("The resulting signal will be interpolated according to the"
                 " desired new sample_rate.")
            new_time_vector = np.arange(
                self.min_freq, self.max_freq+new_sample_rate, new_sample_rate
            )
            new_value = []
            for i in range(self.amplitude.shape[-1]):
                signal = self.amplitude[..., i]
                new_signal = interpolate.interp1d(new_time_vector, signal)
                new_value.append(new_signal)
            new_value = np.array(new_value)
        else:
            step = int(new_sample_rate / self.sample_rate)
            new_value = self.amplitude[0::step, :, :]

        return signal(signal=new_value,
                      sample_rate=new_sample_rate,
                      elapsed_time=max(new_time_vector)-min(new_time_vector),
                      max_time=np.amax(new_time_vector),
                      min_time=np.amin(new_time_vector),
                      label=self.label,
                      coordinates=self.coordinates,
                      orientations=self.orientations,
                      units=self.units)

            
    def change_sample_rate(self, new_min_time: float = None,
                           new_max_time: float = None):
        if new_min_time is None:
            new_min_time = self.min_time
        if new_max_time is None:
            new_max_time = self.max_time
        new_time_vector = self.time_vector
        closest_max_time_index = (
            np.abs(new_time_vector - new_max_time)
        ).argmin()
        closest_max_time = new_time_vector[closest_max_time_index]
        closest_min_time_index = (
            np.abs(new_time_vector - new_min_time)
        ).argmin()
        closest_min_time = new_time_vector[closest_min_time_index]
        new_elapsed_time = closest_max_time - closest_min_time
        new_sample_rate = (
            new_elapsed_time / (closest_max_time_index-closest_min_time_index)
        )
        if new_sample_rate != self.sample_rate:
            new_signal = self.change_sample_rate(new_sample_rate)
        else:
            new_signal = self.change_sample_rate(self.sample_rate)
        new_time_vector = new_signal.time_vector

        if new_max_time > new_signal.max_time:
            time_extension = np.arange(
                new_signal.max_time,
                new_max_time+new_signal.sample_rate,
                new_signal.sample_rate
            )
            new_time_vector = np.hstack((new_time_vector, time_extension))
            amplitude_extension = np.zeros((
                time_extension.shape[0], new_signal.amplitude.shape[-1]
            ))
            new_amplitude = np.hstack((
                new_signal.amplitude, amplitude_extension
            ))
        else:
            new_time_vector = new_time_vector[
                0:(np.abs(new_time_vector - new_max_time)).argmin()
            ]
            new_amplitude = new_amplitude[
                0:(np.abs(new_time_vector - new_max_time)).argmin(), :
            ]

        if new_min_time < new_signal.min_time:
            time_extension = np.arange(
                new_min_time,
                new_signal.min_time+new_signal.sample_rate,
                new_signal.sample_rate
            )
            new_time_vector = np.hstack((time_extension, new_time_vector))
            amplitude_extension = np.zeros((
                time_extension.shape[0], new_signal.amplitude.shape[-1]
            ))
            new_amplitude = np.hstack((
                amplitude_extension,
                new_signal.amplitude
            ))
            if new_min_time < 0:
                new_time_vector = new_time_vector + abs(new_min_time)
        else:
            new_time_vector = new_time_vector[
                (np.abs(new_time_vector - new_max_time)).argmin():
            ]
            new_amplitude = new_amplitude[
                (np.abs(new_time_vector - new_max_time)).argmin():, :
            ]
        return signal(signal=new_amplitude,
                      sample_rate=new_sample_rate,
                      elapsed_time=max(new_time_vector)-min(new_time_vector),
                      max_time=np.amax(new_time_vector),
                      min_time=np.amin(new_time_vector),
                      label=self.label,
                      coordinates=self.coordinates,
                      orientations=self.orientations,
                      units=self.units)


    def apply_window(self, window: str):
        if isinstance(window, str):
            if window == "hanning":
                self.amplitude = self.amplitude * np.tile(
                    np.hanning(self.samples), (1, len(self))
                )
            elif window == "hamming":
                self.amplitude = self.amplitude * np.tile(
                    np.hamming(self.samples), (1, len(self))
                )
        else:
            window = interpolate.interp1d(self.time_vector, window)
            self.amplitude = self.amplitude * np.tile(window, (1, len(self)))