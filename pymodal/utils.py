import numpy as np
import numpy.typing as npt
from warnings import warn
from decimal import Decimal
from matplotlib import pyplot as plt


def change_resolution(
    domain_array: npt.NDArray[np.float64],
    amplitude_array: npt.NDArray[np.complex64],
    new_resolution: float,
):
    """
    Change the resolution of an array of signals, assuming the temporal
    dimension of said signal is the first dimension of the array.

    Parameters
    ----------
    domain_array: np.ndarray of float
        An array containing the temporal dimension, which measures the rate of
        physical change, be it by using time, frequency or any other suitable
        quantity.
    amplitude_array: np.ndarray of complex
        An array with the amplitude of the signal recorded along the domain
        array.
    new_resolution: float
        The desired distance between any two adjacent values of the domain
        array.

    Returns
    -------
    A numpy array of floats
        An array containing the new temporal dimension, which measures the rate
        of physical change, be it by using time, frequency or any other
        suitable quantity; with the new resolution.
    A numpy array of complexes
        An array with the new amplitude of the signal recorded along the domain
        array, with the values corresponding to the values of the new domain
        array.
    """

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
                "The resulting max time will be different to accommodate for the"
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


if __name__ == "__main__":
    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = np.cos(domain_array)
    new_domain_array, new_amplitude_array = change_resolution(
        domain_array=domain_array, amplitude_array=amplitude_array, new_resolution=0.07
    )
    plt.plot(domain_array, amplitude_array)
    plt.plot(new_domain_array, new_amplitude_array)
    plt.show()
