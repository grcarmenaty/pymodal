import numpy as np
import numpy.typing as npt
from warnings import warn
from decimal import Decimal


def change_resolution(
    domain_array: npt.NDArray[np.float64],
    amplitude_array: npt.NDArray[np.complex64],
    new_resolution: float,
):
    """Change the resolution of an array of signals, assuming the temporal dimension of
    said signal is the first dimension of the array.

    Parameters
    ----------
    domain_array: np.ndarray of float
        An array containing the temporal dimension, which measures the rate of physical
        change, be it by using domain, frequency or any other suitable quantity.
    amplitude_array: np.ndarray of complex
        An array with the amplitude of the signal recorded along the domain array.
    new_resolution: float
        The desired distance between any two adjacent values of the domain array.

    Returns
    -------
    A numpy array of floats
        An array containing the new temporal dimension, which measures the rate of
        physical change, be it by using domain, frequency or any other suitable quantity;
        with the new resolution.
    A numpy array of complexes
        An array with the new amplitude of the signal recorded along the domain array,
        with the values corresponding to the values of the new domain array.

    """

    domain_array = np.asarray(domain_array)
    amplitude_array = np.asarray(amplitude_array)
    new_resolution = float(new_resolution)
    if domain_array.shape[0] != amplitude_array.shape[0]:
        raise ValueError(
            "Both the domain array and the amplitude array should be of the same"
            " length along their first axis."
        )
    if np.all(np.diff(domain_array) < 0):
        raise ValueError(
            "Domain array values should be strictly increasing."
        )
    
    new_domain_array = np.arange(
        domain_array[0], domain_array[-1] + new_resolution / 2, new_resolution
    )
    # Determine the amount of decimal places the user desires from the amount
    # of decimals in the desired resolution.
    decimal_places = abs(Decimal(str(new_resolution)).as_tuple().exponent)
    np.around(new_domain_array, decimals=decimal_places)
    # Infer the current resolution from the average of differences between
    # elements in the domain array.
    domain_diff = np.diff(domain_array)
    resolution = np.average(domain_diff)
    # Make sure the new max domain value is smaller than the previous max
    # domain value.
    domain_max_value = domain_array[-1]
    if not np.allclose(new_domain_array[-1], domain_max_value):
        if new_domain_array[-1] > domain_max_value:
            new_domain_array = new_domain_array[:-1]
        warn(
            (
                "The resulting max domain will be different to accommodate for the"
                " new resolution."
            ),
            UserWarning,
        )
    else:
        new_domain_array[-1] = domain_max_value

    # Check if new resolution is multiple of old resolution or resolution is
    # not constant
    is_multiple = np.allclose(new_resolution % resolution, 0)
    is_constant = np.allclose(domain_diff, np.ones(domain_diff.shape) * resolution)
    if not is_multiple or not is_constant:
        warn(
            (
                "The resulting signal will be interpolated according to the"
                " desired new resolution."
            ),
            UserWarning,
        )
        # Interpolate the values for each signal according to the new domain
        # vector
        amplitude_shape = list(amplitude_array.shape)
        amplitude_shape[0] = new_domain_array.shape[0]
        amplitude_shape = tuple(amplitude_shape)
        flat_amplitude = amplitude_array.reshape((amplitude_array.shape[0], -1))
        new_amplitude_array = np.empty(
            (new_domain_array.shape[0], flat_amplitude.shape[-1]),
            dtype=amplitude_array.dtype,
        )
        for i in range(flat_amplitude.shape[-1]):
            new_amplitude_array[:, i] = np.interp(
                new_domain_array, domain_array, flat_amplitude[:, i]
            )
        new_amplitude_array = new_amplitude_array.reshape(amplitude_shape)
    else:
        # Keep values corresponding to the new resolution
        step = int(new_resolution / resolution)
        new_amplitude_array = amplitude_array[0::step, ...]

    return new_domain_array, new_amplitude_array


def change_domain(
    domain_array: npt.NDArray[np.float64],
    amplitude_array: npt.NDArray[np.complex64],
    new_min_domain: float = None,
    new_max_domain: float = None,
):
    # Make sure both new max and min domains exist
    if new_min_domain is None:
        new_min_domain = domain_array[0]
    if new_max_domain is None:
        new_max_domain = domain_array[-1]
    # Get current resolution
    domain_diff = np.diff(domain_array)
    resolution = np.average(domain_diff)
    
    # Add a tail of 0s if max domain is greater than the current max domain
    if new_max_domain > domain_array[-1]:
        domain_extension = np.arange(
            domain_array[-1],
            new_max_domain + resolution / 2,
            resolution,
        )[1:]
        # Make sure the last domain is coherent with resolution and not
        # greater than the new max domain desired
        if not np.allclose(domain_extension[-1], new_max_domain):
            if domain_extension[-1] > new_max_domain:
                domain_extension = domain_extension[:-1]
            warn(
                f"Max domain will be changed to keep sample rate constant", UserWarning
            )
        new_domain_array = np.hstack((domain_array, domain_extension))
        # Add as many amplitude points as domain points were created
        amplitude_extension_shape = amplitude_array.shape
        amplitude_extension_shape[0] = domain_extension.shape[0]
        amplitude_extension = np.zeros(amplitude_extension_shape)
        new_amplitude_array = np.hstack((amplitude_array, amplitude_extension))
    else:
        max_domain_index = (np.abs(domain_array - new_max_domain)).argmin()
        new_domain_array = domain_array[0:max_domain_index]
        # Make sure the last domain is coherent with sample rate and not
        # greater than the new max domain desired
        if new_domain_array[-1] != new_max_domain:
            if new_domain_array[-1] > new_max_domain:
                new_domain_array = new_domain_array[:-1]
            warn(
                f"Max domain will be changed to keep sample rate constant", UserWarning
            )
        # Cut the signals to the new max domain
        new_amplitude_array = amplitude_array[0:max_domain_index, ...]
    
    # Add a head of 0s to the signals if the new min domain is smaller than
    # the precious min domain
    if new_min_domain < domain_array[0]:
        domain_extension = np.arange(
            new_min_domain,
            domain_array[0] + resolution / 2,
            resolution,
        )
        # Make sure the domain extension is compatible with the previous domain
        # vector
        if not np.allclose(domain_extension[-1], domain_array[0]):
            domain_extension = domain_extension + (
                domain_array[0] - domain_extension[-1]
            )
            warn(
                f"Min domain will be changed to keep sample rate constant", UserWarning
            )
        new_domain_array = np.hstack((domain_extension, new_domain_array))
        if new_domain_array[0] < 0:
            new_domain_array = new_domain_array + abs(new_domain_array[0])
        # Add as many amplitude points as domain points were created
        amplitude_extension_shape = amplitude_array.shape
        amplitude_extension_shape[0] = domain_extension.shape[0]
        amplitude_extension = np.zeros(amplitude_extension_shape)
        new_amplitude_array = np.hstack((amplitude_extension, amplitude_array))
    else:
        min_domain_index = (np.abs(new_domain_array - new_max_domain)).argmin()
        new_domain_array = new_signal.domain_array[min_domain_index:]
        # Make sure the new min domain is the closest to the one specified
        # by the user.
        if new_domain_array[0] != new_min_domain:
            warn(
                f"To keep sample rate constant, the resulting min domain"
                f" will be {new_domain_array[0]}",
                UserWarning,
            )
        new_signal.domain_array = new_domain_array
        domain_array[-1] = new_domain_array[-1]
        # Cut the signal from the new min domain
        new_signal.amplitude = new_signal.amplitude[min_domain_index:, :]
    return new_signal


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Create a sinusoidal signal from a domain array
    domain_array = np.arange(0, 2 * np.pi, 0.1)
    amplitude_array = np.sin(domain_array)
    # Change the resolution to half the original resolution
    new_domain_array, new_amplitude_array = change_resolution(
        domain_array=domain_array, amplitude_array=amplitude_array, new_resolution=0.2
    )
    # You should be able to see the difference in the peaks.
    plt.plot(domain_array, amplitude_array)
    plt.plot(new_domain_array, new_amplitude_array)
    plt.show()
