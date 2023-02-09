import numpy.typing as npt
import numpy as np
import pymodal
from scipy import interpolate
import pytest
import warnings
from decimal import Decimal


def amplitude_array_constructor(domain_array: npt.NDArray[np.float64]):
    """
    This function creates a list of four numpy arrays, going from a one-
    dimensional array to a four-dimensional array. Each of them consists of
    a combination of cosine signals displaced one sixth of a turn from the
    provided domain array.

    Args:
        domain_array (numpy array of floats):
            A one-dimensional numpy array containing the values for a supposed
            temporal dimension.

    Returns:
        list of numpy arrays of floats: 
            A list of four numpy arrays, going from a one-dimensional array to
            a four-dimensional array.
    """
    
    amplitude_array_4d = np.empty((domain_array.shape[0], 2, 3, 4))
    amplitude_array_4d[:, 0, 0, 0] = np.cos(domain_array)
    amplitude_array_4d[:, 0, 1, 0] = np.cos(domain_array + 2*np.pi/6)
    amplitude_array_4d[:, 0, 2, 0] = np.cos(domain_array + 2*2*np.pi/6)
    amplitude_array_4d[:, 1, 0, 0] = np.cos(domain_array + 3*2*np.pi/6)
    amplitude_array_4d[:, 1, 1, 0] = np.cos(domain_array + 4*2*np.pi/6)
    amplitude_array_4d[:, 1, 2, 0] = np.cos(domain_array + 5*2*np.pi/6)
    for i in range(4):
        amplitude_array_4d[:, ..., i] = amplitude_array_4d[:, ..., 0] + 10*i
    amplitude_array_3d = amplitude_array_4d[:, ..., 0]
    amplitude_array_2d = amplitude_array_3d.reshape((domain_array.shape[0], 6))
    amplitude_array_1d = amplitude_array_2d[:, 4]
    return [amplitude_array_1d,
            amplitude_array_2d,
            amplitude_array_3d,
            amplitude_array_4d]


def test_change_resolution():
    
    def test_change_resolution_internal_loop(domain_array,
                                             amplitude_array,
                                             resolution):
        decimal_places = abs(
            Decimal(str(resolution)).as_tuple().exponent
        )
        new_domain_array, new_amplitude_array = (
            pymodal.change_resolution(domain_array,
                                      amplitude_array,
                                      resolution)
        )
        resolution_warning = (
            not np.allclose(resolution % 0.1, 0) or domain_array[-1] > 200
        )
        max_time_warning = not np.allclose(domain_array[-1],
                                           new_domain_array[-1])
        if resolution_warning and max_time_warning:
            with pytest.warns(UserWarning) as record:
                new_domain_array, new_amplitude_array = (
                    pymodal.change_resolution(domain_array,
                                              amplitude_array,
                                              resolution)
                )
            assert len(record) == 2
            new_max_domain_value = new_domain_array[-1]
            assert record[0].message.args[0] == (
                f"The resulting max time will be"
                f" {new_max_domain_value:.{decimal_places}f}."
            )
            assert record[1].message.args[0] == (
                "The resulting signal will be interpolated according to the"
                " desired new resolution."
            )
        elif resolution_warning and not max_time_warning:
            with pytest.warns(UserWarning) as record:
                new_domain_array, new_amplitude_array = (
                    pymodal.change_resolution(domain_array,
                                              amplitude_array,
                                              resolution)
                )
            assert len(record) == 1
            assert record[0].message.args[0] == (
                "The resulting signal will be interpolated according to the"
                " desired new resolution."
            )
        elif not resolution_warning and max_time_warning:
            with pytest.warns(UserWarning) as record:
                new_domain_array, new_amplitude_array = (
                    pymodal.change_resolution(domain_array,
                                              amplitude_array,
                                              resolution)
                )
            assert len(record) == 1
            assert record[0].message.args[0] == (
                f"The resulting max time will be"
                f" {new_max_domain_value:.{decimal_places}f}."
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                new_domain_array, new_amplitude_array = (
                    pymodal.change_resolution(domain_array,
                                              amplitude_array,
                                              resolution)
                )
        amplitude_array = amplitude_array.reshape((
            amplitude_array.shape[0], -1
        ))
        new_amplitude_array = new_amplitude_array.reshape((
            new_amplitude_array.shape[0], -1
        ))
        
        for i in range(amplitude_array.shape[-1]):
            f = interpolate.interp1d(domain_array, amplitude_array[:, i])
            new_f = interpolate.interp1d(new_domain_array,
                                         new_amplitude_array[:, i])
            reference = f(np.arange(50, 60, 0.01))
            new_reference = new_f(np.arange(50, 60, 0.01))
            assert np.allclose(reference, new_reference, atol=1e-2, rtol=1e-2)
    
    # Create a set of domain arrays
    array_collection = np.vstack((
        np.arange(0, 120.05, 0.1),
        np.arange(0, 120.05, 0.1) + 10,
        np.hstack((np.arange(0, 60.05, 0.1), np.arange(60.25, 210.125, 0.25))),
        np.hstack((np.arange(0, 60.05, 0.1),
                   np.arange(60.25, 210.125, 0.25))) + 10
    ))
    # Get the amplitude arrays for said domain arrays
    array_collection = [
        (array, amplitude_array_constructor(array)) for array in array_collection
    ]
    for array in array_collection:
        domain_array = array[0]
        for base_amplitude_array in array[1]:
            amplitude_arrays = [base_amplitude_array,
                                base_amplitude_array+base_amplitude_array*1j]
            for amplitude_array in amplitude_arrays:
                resolutions = [0.2, 0.07, 0.13, 0.15]
                for resolution in resolutions:
                    test_change_resolution_internal_loop(domain_array,
                                                          amplitude_array,
                                                          resolution)

if __name__ == "__main__":
    test_change_resolution()