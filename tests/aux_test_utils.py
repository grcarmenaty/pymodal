import numpy as np
import numpy.typing as npt

def amp_array_constructor(
    domain_array: npt.NDArray[np.float64],
    dimensions: str = "1d",
    return_complex: bool = False,
):
    """This function can create a series of four numpy arrays, going from a one-
    dimensional array to a four-dimensional array. Each of them consists of a
    combination of cosine signals displaced one sixth of a turn from the provided
    domain array.

    Parameters
    ----------
    domain_array (numpy array of floats):
        A one-dimensional numpy array containing the values for a supposed domain
        dimension.
    dimensions (string), "1d" as default:
        One of four options among "1d", "2d", "3d" or "4d" which selects how many
        dimensions the output numpy array will have.
    return_complex (bool), False as default:
        Whether ot not the resulting array should be complex or not by adding the same
        amplitude array.

    Returns
    -------
    numpy array of floats or complexes:
        A numpy array, of between a one-dimensional array to a four-dimensional
        array, complex or not as specified.

    """

    assert dimensions in ["1d", "2d", "3d", "4d"]

    amplitude_array = np.empty((domain_array.shape[0], 2, 3, 4))
    amplitude_array[:, 0, 0, 0] = np.cos(domain_array)
    amplitude_array[:, 0, 1, 0] = np.cos(domain_array + 2 * np.pi / 6)
    amplitude_array[:, 0, 2, 0] = np.cos(domain_array + 2 * 2 * np.pi / 6)
    amplitude_array[:, 1, 0, 0] = np.cos(domain_array + 3 * 2 * np.pi / 6)
    amplitude_array[:, 1, 1, 0] = np.cos(domain_array + 4 * 2 * np.pi / 6)
    amplitude_array[:, 1, 2, 0] = np.cos(domain_array + 5 * 2 * np.pi / 6)
    for i in range(4):
        amplitude_array[:, ..., i] = amplitude_array[:, ..., 0] + 10 * i

    if dimensions == "3d":
        amplitude_array = amplitude_array[:, ..., 0]

    elif dimensions == "2d":
        amplitude_array = amplitude_array[:, ..., 0]
        amplitude_array = amplitude_array.reshape((domain_array.shape[0], 6))

    elif dimensions == "1d":
        amplitude_array = amplitude_array[:, ..., 0]
        amplitude_array = amplitude_array.reshape((domain_array.shape[0], 6))
        amplitude_array = amplitude_array[:, 4]

    if return_complex:
        amplitude_array = amplitude_array + amplitude_array * 1j
    return amplitude_array