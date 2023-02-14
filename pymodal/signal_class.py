import numpy as np
from warnings import warn
import pymodal
from scipy import interpolate

class signal():

    """
    Store and manipulate vibration signals recorded under the same conditions
    from the same structure.
    """

    def __init__(self,
                 amplitude: np.ndarray,
                 sample_rate: float = None,
                 elapsed_time: float = None,
                 max_time: float = None,
                 min_time: float = 0,
                 label: list[str] = None,
                 coordinates: np.ndarray = None,
                 orientations: np.ndarray = None,
                 units: str = None):

        """
        Constructor for the signal class

        Parameters
        ----------
        amplitude: numpy array
            2-dimensional vector with the amplitudes of as many excitation
            signals as have been recorded using the same parameters on the same
            structure; it is assumed they all correspond to independent degrees
            of freedom.
        sample_rate: float, optional if at least one of max_time, elapsed_time
                     is provided
            How much time passes between one data point and the next in the
            signals.
        elapsed_time: float, optional if at least one of max_time, sample_rate
                      is provided
            How much time passes between the start of the signal recording and
            the end of the signal recording.
        max_time: float, optional if at least one of elapsed_time, sample_rate
                      is provided
            Time at the end of the recording of the signal.
        min_time: float, default 0
            Time at the start of the recording of the signal.
        label: list of str, optional
            A list of identifying labels for each signal stored within the
            signal object. If not provided, ordinal labels will be defined.
        coordinates: numpy array, optional
            A 2-dimensional array with the spatial coordinates of every degree
            of freedom. If not provided, it is assumed all degrees of freedom
            are distributed along the x axis every 1 unit of distance.
        orientations: numpy array, optional
            A 2-dimensional array with the vector defining the positive
            direction of the signal to which it is associated. If not provided,
            it is assumed all orientations are a unit vector along the z axis. 

        Returns
        -------
        out: signal class
            An object designed to store and manipulate vibration signals
            recorded under the same conditions from the same structure.

        Notes
        -----
        """

        self.amplitude = np.array(amplitude)
        if self.amplitude.ndim > 3:
            raise ValueError(f"Amplitude must be, at most, a three-dimensional"
                             f" array, but it is a {self.amplitude.ndim}"
                             f"-dimensional array.")
        elif self.amplitude.ndim < 3:
            for _ in range(3-self.amplitude.ndim):
                self.amplitude = [..., np.newaxis]
        if units is None:
            self.units = "mm, kg, s, °C"
            warn("Units will be assumed to be mm, kg, s, °C.", UserWarning)
        else:
            self.units = units
        self.units = units
        self.samples = self.amplitude.shape[0]
        self.degrees_of_freedom = self.amplitude.shape[1]

        self.label = label
        # If no label is provided, set it to "Unnamed label {i}" for each label
        if self.label is None:
            self.label = [
                f"Signal {i}" for i in range(self.degrees_of_freedom)
            ]
        else:
            self.label = list(self.label) # Make sure label is a list

        self.min_time = min_time  # Minimum time is assumed to be 0
        if elapsed_time is None:
            if max_time is None:
                # If neither elapsed_time nor maximum time are defined then
                # sample_rate MUST be defined
                if sample_rate is None:
                    raise ValueError(f"At least one of elapsed time, max time"
                                     f" or sample rate must be provided.")
                self.sample_rate = sample_rate
                # So elapsed_time and maximum time can be calculated
                self.elapsed_time = (self.samples - 1) * self.sample_rate
                self.max_time = self.min_time + self.elapsed_time
            else:  # If elapsed_time is not defined but maximum time is
                self.max_time = max_time
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

        self.time_array = np.arange(
            self.min_time,
            self.max_time + self.sample_rate / 2,
            self.sample_rate
        )
        if self.time_array[-1] != self.max_time:
            if self.time_array[-1] > self.max_time:
                self.time_array = self.time_array[:-1]
            self.max_time = self.time_array[-1]
            warn(f"The resulting max time will be {self.max_time}",
                 UserWarning)

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
                warn(f"Your coordinates are {self.coordinates.shape[1]}"
                     f"-dimensional, Only first three dimensions of"
                     f" coordinates will be used.",
                     UserWarning)
                self.coordinates = self.coordinates[:, :2]
            elif dimension_difference < 0:
                warn(f"Your coordinates are {self.coordinates.shape[1]}"
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
            warn("orientations will be assumed to be unit vectors on the z"
                 " axis.",
                 UserWarning)
            self.orientations = np.vstack(
                np.zeros((self.degrees_of_freedom, 2)),
                np.ones(self.degrees_of_freedom)
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
                if not(equal_array):
                    break
                equal_array = (np.array_equal(own_array, other_arrays[i])
                                and equal_array)
            # Instances are equal if both array and non-array parts are equal.
            return own_dict == other_dict and equal_array
        else:
            return False


    def __getitem__(self, index: slice):
        # Make sure argument is a slice.
        if not isinstance(index, slice):
            index = slice(index, index + 1, None)
        return signal(
            amplitude = self.amplitude[:, index.start:index.stop:index.step],
            sample_rate = self.sample_rate,
            elapsed_time = self.elapsed_time,
            max_time = self.max_time,
            min_time = self.min_time,
            label = self.label[index.start:index.stop:index.step],
            coordinates = self.coordinates[
                :, index.start:index.stop:index.step
            ],
            orientations = self.orientations[
                :, index.start:index.stop:index.step
            ],
            units = self.units
        )


    def append(self, new_signal):
        if self.sample_rate != new_signal.sample_rate:
            raise ValueError(f"Sample rate for the signals to be added is"
                             f" {new_signal.sample_rate}. It should be equal"
                             f" to the base signal's sample rate:"
                             f" {self.sample_rate}")
        if self.min_time != new_signal.min_time:
            raise ValueError(f"Minimum time for the signals to be added is"
                             f" {new_signal.min_time}. It should be equal"
                             f" to the base signal's minimum time:"
                             f" {self.min_time}")
        if self.max_time != new_signal.max_time:
            raise ValueError(f"Maximum time for the signals to be added is"
                             f" {new_signal.max_time}. It should be equal"
                             f" to the base signal's maximum time:"
                             f" {self.max_time}")
        if self.units != new_signal.units:
            raise ValueError(f"Units for the signals to be added is"
                             f" {new_signal.units}. It should be equal"
                             f" to the base signal's units:"
                             f" {self.units}")
        return signal(
            amplitude = np.vstack((self.amplitude, new_signal.amplitude)),
            sample_rate = self.sample_rate,
            elapsed_time = self.elapsed_time,
            max_time = self.max_time,
            min_time = self.min_time,
            label = np.vstack((self.label, new_signal.label)),
            coordinates = np.vstack((self.coordinates,
                                     new_signal.coordinates)),
            orientations = np.vstack((self.orientations,
                                      new_signal.orientations)),
            units = self.units
        )


    def change_sample_rate(self, new_sample_rate: float):
        new_time_array = np.arange(
            self.min_time, self.max_time+new_sample_rate/2, new_sample_rate
        )
        if new_time_array[-1] != self.max_time:
            if new_time_array[-1] > self.max_time:
                new_time_array = new_time_array[:-1]
            warn(f"The resulting max time will be {new_time_array[-1]}",
                 UserWarning)
        # Check if new sample rate is multiple of old sample rate
        if new_sample_rate % self.sample_rate != 0:
            warn("The resulting signal will be interpolated according to the"
                 " desired new sample_rate.", UserWarning)
            # Interpolate the values for each signal according to the new time
            # vector
            new_amplitude = []
            for i in range(self.amplitude.shape[-1]):
                signal = self.amplitude[..., i]
                new_signal = interpolate.interp1d(new_time_array, signal)
                new_amplitude.append(new_signal)
            new_amplitude = np.array(new_amplitude)
        else:
            # Keep values corresponding to the new sample rate
            step = int(new_sample_rate / self.sample_rate)
            new_amplitude = self.amplitude[0::step, :, :]

        return signal(amplitude=new_amplitude,
                      sample_rate=new_sample_rate,
                      elapsed_time=(new_time_array[-1]-new_time_array[0]),
                      max_time=new_time_array[-1],
                      min_time=new_time_array[0],
                      label=self.label,
                      coordinates=self.coordinates,
                      orientations=self.orientations,
                      units=self.units)

            
    def change_time(self,
                    new_min_time: float = None,
                    new_max_time: float = None):
        # Create a copy of the current object to work on it
        new_signal = signal(amplitude=self.amplitude,
                            sample_rate=self.sample_rate,
                            elapsed_time=self.elapsed_time,
                            max_time=self.max_time,
                            min_time=self.min_time,
                            label=self.label,
                            coordinates=self.coordinates,
                            orientations=self.orientations,
                            units=self.units)
        # Make sure both new max and min times exist
        if new_min_time is None:
            new_min_time = new_signal.min_time
        if new_max_time is None:
            new_max_time = new_signal.max_time
        # Add a tail of 0s if max time is greater than the current max time
        if new_max_time > new_signal.max_time:
            time_extension = np.arange(
                new_signal.max_time,
                new_max_time+new_signal.sample_rate/2,
                new_signal.sample_rate
            )
            # Make sure the last time is coherent with sample rate and not
            # greater than the new max time desired
            if time_extension[-1] != new_max_time:
                if time_extension[-1] > new_max_time:
                    time_extension = time_extension[:-1]
                warn(f"To keep sample rate constant, the resulting max time"
                     f" will be {time_extension[-1]}", UserWarning)
            new_time_array = np.hstack((new_time_array, time_extension)) # noqa
            new_signal.time_array = new_time_array
            new_signal.max_time = new_time_array[-1]
            # Add as many amplitude points as time points were created
            amplitude_extension = np.zeros((
                time_extension.shape[0], new_signal.amplitude.shape[-1]
            ))
            new_signal.amplitude = np.hstack((
                new_signal.amplitude, amplitude_extension
            ))
        else:
            max_time_index = (
                np.abs(new_signal.time_array - new_max_time)
            ).argmin()
            new_time_array = new_signal.time_array[0:max_time_index]
            # Make sure the last time is coherent with sample rate and not
            # greater than the new max time desired
            if new_time_array[-1] != new_max_time:
                if new_time_array[-1] > new_max_time:
                    new_time_array = new_time_array[:-1]
                warn(f"To keep sample rate constant, the resulting max time"
                     f" will be {new_time_array[-1]}", UserWarning)
            new_signal.time_array = new_time_array
            new_signal.max_time = new_time_array[-1]
            # Cut the signals to the new max time
            new_signal.amplitude = new_signal.amplitude[
                0:len(new_time_array)-1, :
            ]
        # Add a head of 0s to the signals if the new min time is smaller than
        # the precious min time
        if new_min_time < new_signal.min_time:
            time_extension = np.arange(
                new_min_time,
                new_signal.min_time+new_signal.sample_rate/2,
                new_signal.sample_rate
            )
            # Make sure the time extension is compatible with the previous time
            # vector
            if time_extension[-1] != new_signal.min_time:
                time_extension = time_extension + (
                    new_signal.min_time-time_extension[-1]
                )
                warn(f"To keep sample rate constant, the resulting min time"
                     f" will be {time_extension[0]}", UserWarning)
            new_time_array = np.hstack((time_extension, new_time_array))
            if new_time_array[0] < 0:
                new_time_array = new_time_array + abs(new_time_array[0])
            new_signal.time_array = new_time_array
            new_signal.max_time = new_time_array[-1]
            # Add as many amplitude points as time points were created
            amplitude_extension = np.zeros((
                time_extension.shape[0], new_signal.amplitude.shape[-1]
            ))
            new_signal.amplitude = np.hstack((amplitude_extension,
                                       new_signal.amplitude))
        else:
            min_time_index = (np.abs(new_time_array - new_max_time)).argmin()
            new_time_array = new_signal.time_array[min_time_index:]
            # Make sure the new min time is the closest to the one specified
            # by the user.
            if new_time_array[0] != new_min_time:
                warn(f"To keep sample rate constant, the resulting min time"
                     f" will be {new_time_array[0]}", UserWarning)
            new_signal.time_array = new_time_array
            new_signal.max_time = new_time_array[-1]
            # Cut the signal from the new min time
            new_signal.amplitude = new_signal.amplitude[
                min_time_index:, :
            ]
        return new_signal


    def apply_window(self, window: str):
        # Each windowing method should be individually programmed
        if isinstance(window, str):
            if window == "hanning":
                new_amplitude = self.amplitude * np.tile(
                    np.hanning(self.samples), (1, len(self))
                )
            elif window == "hamming":
                new_amplitude = self.amplitude * np.tile(
                    np.hamming(self.samples), (1, len(self))
                )
        else:
            # If an iterable object is provided, then it can be applied as
            # window to all signals
            window = interpolate.interp1d(self.time_array, np.array(window))
            new_amplitude = self.amplitude * np.tile(window, (1, len(self)))
        return signal(amplitude=new_amplitude,
                      sample_rate=self.sample_rate,
                      elapsed_time=self.elapsed_time,
                      max_time=self.max_time,
                      min_time=self.min_time,
                      label=self.label,
                      coordinates=self.coordinates,
                      orientations=self.orientations,
                      units=self.units)
        
    
    # def to_frf(self, excitation_signal: pymodal.signal):
    #     return frf_object