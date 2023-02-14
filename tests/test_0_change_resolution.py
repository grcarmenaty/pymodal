import numpy.typing as npt
import numpy as np
import pymodal
import pytest
import warnings
from typing import Optional


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


def _test_change_resolution(
    domain_array: npt.NDArray[np.float64],
    amplitude_array: npt.NDArray[np.complex64],
    new_resolution: float,
    expected_warnings: int = 0,
    warning_messages: Optional[list[str]] = None,
):
    """This function tests the change_resolution function given a certain combination
    of domain, amplitude and desired new resolution. If warnings are expected, it checks
    that the right amount are triggered, and that their messages are as expected.

    Parameters
    ----------
    domain_array: numpy array of float
        An array containing the temporal dimension, which measures the rate of physical
        change, be it by using domain, frequency or any other suitable quantity.
    amplitude_array: numpy array of complex
        An array with the amplitude of the signal recorded along the domain array.
    new_resolution: float
        The desired distance between any two adjacent values of the domain array.
    expected_warnings: int
        The amount of warnings which are expected to be raised during the execution of
        the change_resolution function given the input parameters.
    warning_messages (optional): list of strings:
        The messages that should be contained withing the warnings being raised, if any.
    """

    if expected_warnings == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            new_domain_array, new_amplitude_array = pymodal.change_resolution(
                domain_array, amplitude_array, new_resolution
            )
    else:
        with pytest.warns(UserWarning) as records:
            new_domain_array, new_amplitude_array = pymodal.change_resolution(
                domain_array, amplitude_array, new_resolution
            )
        assert amplitude_array.dtype == new_amplitude_array.dtype
        assert len(records) == expected_warnings
        for i, record in enumerate(records):
            assert record.message.args[0] == warning_messages[i]

    amplitude_array = amplitude_array.reshape((amplitude_array.shape[0], -1))
    new_amplitude_array = new_amplitude_array.reshape(
        (new_amplitude_array.shape[0], -1)
    )
    for i in range(amplitude_array.shape[-1]):
        reference = np.interp(
            np.arange(50, 60, 0.01), domain_array, amplitude_array[:, i]
        )
        result = np.interp(
            np.arange(50, 60, 0.01), new_domain_array, new_amplitude_array[:, i]
        )
        assert np.allclose(reference, result, atol=1e-2, rtol=1e-2)


def test_change_resolution_0to120_0·1to0·2_1d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 1-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=False
    )
    new_resolution = 0.2
    _test_change_resolution(domain_array, amplitude_array, new_resolution)


def test_change_resolution_0to120_0·1to0·2_1d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 1-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=True
    )
    new_resolution = 0.2
    _test_change_resolution(domain_array, amplitude_array, new_resolution)


def test_change_resolution_0to120_0·1to0·2_2d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 2-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=False
    )
    new_resolution = 0.2
    _test_change_resolution(domain_array, amplitude_array, new_resolution)


def test_change_resolution_0to120_0·1to0·2_2d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 2-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=True
    )
    new_resolution = 0.2
    _test_change_resolution(domain_array, amplitude_array, new_resolution)


def test_change_resolution_0to120_0·1to0·2_3d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 3-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=False
    )
    new_resolution = 0.2
    _test_change_resolution(domain_array, amplitude_array, new_resolution)


def test_change_resolution_0to120_0·1to0·2_3d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 3-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=True
    )
    new_resolution = 0.2
    _test_change_resolution(domain_array, amplitude_array, new_resolution)


def test_change_resolution_0to120_0·1to0·2_4d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 4-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=False
    )
    new_resolution = 0.2
    _test_change_resolution(domain_array, amplitude_array, new_resolution)


def test_change_resolution_0to120_0·1to0·2_4d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 4-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=True
    )
    new_resolution = 0.2
    _test_change_resolution(domain_array, amplitude_array, new_resolution)


def test_change_resolution_0to120_0·1to0·07_1d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 1-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.07.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=False
    )
    new_resolution = 0.07
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·07_1d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 1-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.07.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=True
    )
    new_resolution = 0.07
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·07_2d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 2-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.07.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=False
    )
    new_resolution = 0.07
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·07_2d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 2-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.07.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=True
    )
    new_resolution = 0.07
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·07_3d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 3-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.07.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=False
    )
    new_resolution = 0.07
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·07_3d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 3-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.07.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=True
    )
    new_resolution = 0.07
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·07_4d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 4-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.07.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=False
    )
    new_resolution = 0.07
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·07_4d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 4-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.07.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=True
    )
    new_resolution = 0.07
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·13_1d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 1-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=False
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·13_1d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 1-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=True
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·13_2d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 2-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=False
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·13_2d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 2-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=True
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·13_3d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 3-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=False
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·13_3d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 3-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=True
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·13_4d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 4-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=False
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·13_4d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 4-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=True
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to120_0·1to0·25_1d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 1-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.25.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=False
    )
    new_resolution = 0.25
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to120_0·1to0·25_1d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 1-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.25.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=True
    )
    new_resolution = 0.25
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to120_0·1to0·25_2d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 2-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.25.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=False
    )
    new_resolution = 0.25
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to120_0·1to0·25_2d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 2-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.25.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=True
    )
    new_resolution = 0.25
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to120_0·1to0·25_3d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 3-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.25.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=False
    )
    new_resolution = 0.25
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to120_0·1to0·25_3d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 3-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.25.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=True
    )
    new_resolution = 0.25
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to120_0·1to0·25_4d():
    """This tests the change_resolution function using a domain vector going from 0 to
    120 with a resolution of 0.1, creating a 4-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.25.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=False
    )
    new_resolution = 0.25
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to120_0·1to0·25_4d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a resolution of 0.1, creating a 4-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.25.
    """

    domain_array = np.arange(0, 120.05, 0.1)
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=True
    )
    new_resolution = 0.25
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to240_mixedto0·13_1d():
    """This tests the change_resolution function using a domain vector going from 0
    to 120 with a mixed resolution, creating a 1-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=False
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to240_mixedto0·13_1d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a mixed resolution, creating a 1-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=True
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to240_mixedto0·13_2d():
    """This tests the change_resolution function using a domain vector going from 0
    to 120 with a mixed resolution, creating a 2-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=False
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to240_mixedto0·13_2d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a mixed resolution, creating a 2-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=True
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to240_mixedto0·13_3d():
    """This tests the change_resolution function using a domain vector going from 0
    to 120 with a mixed resolution, creating a 3-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=False
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to240_mixedto0·13_3d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a mixed resolution, creating a 3-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=True
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to240_mixedto0·13_4d():
    """This tests the change_resolution function using a domain vector going from 0
    to 120 with a mixed resolution, creating a 4-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=False
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to240_mixedto0·13_4d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a mixed resolution, creating a 4-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.13.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=True
    )
    new_resolution = 0.13
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=2,
        warning_messages=[
            (
                "The resulting max time will be different to"
                " accommodate for the new resolution."
            ),
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            ),
        ],
    )


def test_change_resolution_0to240_mixedto0·2_1d():
    """This tests the change_resolution function using a domain vector going from 0
    to 120 with a mixed resolution, creating a 1-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=False
    )
    new_resolution = 0.2
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to240_mixedto0·2_1d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a mixed resolution, creating a 1-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="1d", return_complex=True
    )
    new_resolution = 0.2
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to240_mixedto0·2_2d():
    """This tests the change_resolution function using a domain vector going from 0
    to 120 with a mixed resolution, creating a 2-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=False
    )
    new_resolution = 0.2
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to240_mixedto0·2_2d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a mixed resolution, creating a 2-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="2d", return_complex=True
    )
    new_resolution = 0.2
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to240_mixedto0·2_3d():
    """This tests the change_resolution function using a domain vector going from 0
    to 120 with a mixed resolution, creating a 3-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=False
    )
    new_resolution = 0.2
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to240_mixedto0·2_3d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a mixed resolution, creating a 3-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="3d", return_complex=True
    )
    new_resolution = 0.2
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to240_mixedto0·2_4d():
    """This tests the change_resolution function using a domain vector going from 0
    to 120 with a mixed resolution, creating a 4-dimensional amplitude array of floats
    and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=False
    )
    new_resolution = 0.2
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


def test_change_resolution_0to240_mixedto0·2_4d_complex():
    """This tests the change_resolution function using a domain vector going tofrom 0
    to 120 with a mixed resolution, creating a 4-dimensional amplitude array of
    complexes and changing the resolution of the domain-amplitude pair to 0.2.
    """

    domain_array = np.hstack(
        (np.arange(0, 120.05, 0.2), np.arange(120.25, 240.05, 0.25))
    )
    amplitude_array = amp_array_constructor(
        domain_array=domain_array, dimensions="4d", return_complex=True
    )
    new_resolution = 0.2
    _test_change_resolution(
        domain_array,
        amplitude_array,
        new_resolution,
        expected_warnings=1,
        warning_messages=[
            (
                "The resulting signal will be interpolated"
                " according to the desired new resolution."
            )
        ],
    )


if __name__ == "__main__":
    pass
