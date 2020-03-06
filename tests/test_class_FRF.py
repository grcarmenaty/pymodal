import numpy as np
import ntpath
from pathlib import Path
import pymodal
import pytest

# Build a list of all FRF.mat files in the data folder. All of the used FRF
# files have 81 lines of 0Hz to 3200Hz bands with a resolution of 0.5Hz, which
# implies the shape of the arrays is (6401, 81)
path = (Path(__file__).parent / 'data' / 'FRF').glob('**/*')
files = [file for file in path if file.is_file()]
length = len(files)
# This is expected to be the list of names in each FRF object in which a list
# of files is used as input
name = [ntpath.split(item)[-1] for item in files]
unknown_name = [f'Unknown name {i + 1}' for i in range(length)]
# This is the expected value to be stored in a FRF object using the file list
array_list = [pymodal.load_array(file) for file in files]
array_3d = np.dstack(array_list)


def test_init():

    """

    This tests initialization values using the preferred input method:
    an array list.
    """

    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5, name=name)
    value_test = [(frf.value[i] == array_list[i]).all() for i in range(length)]
    assert np.asarray(value_test).all()
    assert frf.resolution == 0.5
    assert frf.max_freq == 3200
    assert frf.min_freq == 0
    assert frf.bandwidth == 3200
    assert frf.name == name
    assert frf.part == 'complex'
    frf = pymodal.frf.FRF(frf=array_list, min_freq=1000, max_freq=7400)
    # Values have not changed, so they should be the same as before
    value_test = [(frf.value[i] == array_list[i]).all() for i in range(length)]
    assert np.asarray(value_test).all()
    assert frf.resolution == 1
    assert frf.max_freq == 7400
    assert frf.min_freq == 1000
    assert frf.bandwidth == 6400
    assert frf.name == unknown_name  # No names specified
    assert frf.part == 'complex'
    # Make sure an impossible definition raises an error
    try:
        frf = pymodal.frf.FRF(frf=array_list, resolution=0.5, max_freq=6400)
        assert False
    except Exception as __:
        assert True
    # Try to give more names than needed
    try:
        frf = pymodal.frf.FRF(frf=array_list[0:2], resolution=0.5, name=name)
        assert False
    except Exception as __:
        assert True
    # Give less names and make sure it still generates names
    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5, name=name[0:2])
    assert len(frf.name) == len(array_list)


def test_equality():

    """

    This tests that an object of the class is equal to exactly the same
    object and not equal to any object with any different value.
    """

    frf = pymodal.frf.FRF(frf=array_list[0:4], resolution=0.5)
    same_object = pymodal.frf.FRF(frf=array_list[0:4], resolution=0.5)
    other_frf = pymodal.frf.FRF(frf=array_list[0:2], resolution=0.5)
    other_resolution = pymodal.frf.FRF(frf=array_list[0:4], resolution=1)
    other_max_freq = pymodal.frf.FRF(frf=array_list[0:4], max_freq=6400)
    other_min_freq = pymodal.frf.FRF(frf=array_list[0:4], resolution=0.5,
                                 min_freq=9)
    other_bandwidth = pymodal.frf.FRF(frf=array_list[0:4], bandwidth=4200)
    other_name = pymodal.frf.FRF(frf=array_list[0:4], resolution=0.5,
                             name=name[0:4])
    # Doing this is strongly not recommended, part and value should always be
    # related (e.g. if values correspond to the phase, part should be phase,
    # defining the class this way can lead to easily made mistakes)
    other_part = pymodal.frf.FRF(frf=array_list[0:4], resolution=0.5, part='real')
    assert frf == same_object
    assert frf != other_frf
    assert frf != other_resolution
    assert frf != other_max_freq
    assert frf != other_min_freq
    assert frf != other_bandwidth
    assert frf != other_name
    assert frf != other_part


def test_input():

    """

    This tests that a function created using the four possible input
    types in the FRF field (array list, list of .mat file objects, 3D
    array, 2D array and single .mat file object) yields the same object.
    """

    # All objects are created with the names defined so that they can be equal
    # to using file path list as input.
    frf_array_list = pymodal.frf.FRF(frf=array_list, resolution=0.5, name=name)
    frf_file_list = pymodal.frf.FRF(frf=files, resolution=0.5)
    frf_3d_array = pymodal.frf.FRF(frf=array_3d, resolution=0.5, name=name)
    frf_array_list_single = pymodal.frf.FRF(frf=[array_list[2]], resolution=0.5,
                                        name=name[2])
    frf_2d_array = pymodal.frf.FRF(frf=array_list[2], resolution=0.5, name=name[2])
    frf_file_object = pymodal.frf.FRF(frf=files[2], resolution=0.5)
    assert frf_array_list == frf_file_list
    assert frf_array_list == frf_3d_array
    assert frf_array_list_single == frf_2d_array
    assert frf_array_list_single == frf_file_object


def test_getitem():

    """

    This tests that slicing an object has the same values as the sliced
    expected values.
    """

    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5, name=name)
    frf_selected = pymodal.frf.FRF(frf=array_list[5], resolution=0.5, name=name[5])
    frf_sliced = pymodal.frf.FRF(frf=array_list[5:15], resolution=0.5,
                             name=name[5:15])
    assert frf[5] == frf_selected
    assert frf[5:15] == frf_sliced


def test_len():

    """

    This tests that the length of the object is the same as the length
    of the value.
    """

    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5, name=name)
    assert len(frf) == len(array_list)


def test_extend():

    """

    This tests that the extended object is the same as an object with
    the values which were added.
    """
    extended_array_list = list(array_list)
    extended_array_list.extend(array_list[0:2])
    extended_name = list(name)
    extended_name.extend(name[0:2])

    # With names
    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5, name=name)
    frf_extended = pymodal.frf.FRF(frf=extended_array_list, resolution=0.5,
                               name=extended_name)
    frf.extend(array_list[0:2], name[0:2])
    assert frf == frf_extended

    # Without names
    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5)
    frf_extended = pymodal.frf.FRF(frf=extended_array_list, resolution=0.5)
    frf.extend(array_list[0:2])
    assert frf == frf_extended

    # With some names defined in parent object
    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5, name=name[0:15])
    frf_extended = pymodal.frf.FRF(frf=extended_array_list, resolution=0.5,
                               name=name[0:15])
    frf.extend(array_list[0:2])
    assert frf == frf_extended

    # With names defined only in extension
    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5)
    extended_unknown_name = list(unknown_name)
    extended_unknown_name.extend(name[0:2])
    frf_extended = pymodal.frf.FRF(frf=extended_array_list, resolution=0.5,
                               name=extended_unknown_name)
    frf.extend(array_list[0:2], name[0:2])
    assert frf == frf_extended


def test_change_resolution():

    """

    This tests whether change resolution actually changes when the
    method is applied.
    """

    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5)
    with pytest.warns(UserWarning):
        frf_changed = frf.change_resolution(1.2)
    changed_array_list = list(array_list)
    for index, item in enumerate(changed_array_list):
        changed_array_list[index] = item[0::int(np.around(1.2 / 0.5)), :]
    value_test = [(frf_changed.value[i] == changed_array_list[i]).all()
                  for i in range(length)]
    assert frf_changed.resolution == 1
    assert all(value_test)
    assert all([element.shape == (3201, 81) for element in frf_changed.value])


def test_change_lines():

    """

    This tests whether change resolution actually changes when the
    method is applied.
    """

    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5)
    new_lines = list(range(10, 40, 5))
    new_lines.append(62)
    frf_changed = frf.change_lines(new_lines)
    changed_array_list = list(array_list)
    for index, item in enumerate(changed_array_list):
        changed_array_list[index] = [item[:, i] for i in new_lines]
        changed_array_list[index] = np.asarray(
            changed_array_list[index]).conj().T
    value_test = [(frf_changed.value[i] == changed_array_list[i]).all()
                  for i in range(length)]
    assert all(value_test)
    assert all([element.shape == (6401, 7) for element in frf_changed.value])


def test_change_frequencies():

    """

    This tests whether change frequency actually changes when the
    method is applied.
    """

    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5)

    try:  # Length 3 entry should raise an error
        frf_changed = frf.change_frequencies([1000, 2000, 3000])
        assert False
    except Exception as __:
        assert True

    try:  # Length 1 entry should raise an error
        frf_changed = frf.change_frequencies([1000])
        assert False
    except Exception as __:
        assert True

    try:  # Non-iterable entry should raise an error
        frf_changed = frf.change_frequencies(1000)
        assert False
    except Exception as __:
        assert True

    frf_changed = frf.change_frequencies((1000, 2000))
    changed_array_list = list(array_list)
    for index, item in enumerate(changed_array_list):
        changed_array_list[index] = item[2000:4001, :]
    value_test = [(frf_changed.value[i] == changed_array_list[i]).all()
                  for i in range(length)]
    assert frf_changed.min_freq == 1000
    assert frf_changed.max_freq == 2000
    assert frf_changed.bandwidth == 1000
    assert all(value_test)
    assert all([element.shape == (2001, 81) for element in frf_changed.value])


def test_part():

    """

    This tests whether the correct part is selected when the part method
    is applied.
    """

    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5)
    frf_real = pymodal.frf.FRF(frf=[np.absolute(element.real)
                                for element in array_list],
                           resolution=0.5,
                           part='real')
    frf_imag = pymodal.frf.FRF(frf=[np.absolute(element.imag)
                                for element in array_list],
                           resolution=0.5,
                           part='imag')
    frf_abs = pymodal.frf.FRF(frf=[np.absolute(element) for element in array_list],
                          resolution=0.5,
                          part='abs')
    frf_phase = pymodal.frf.FRF(frf=[np.angle(element)
                                 for element in array_list],
                            resolution=0.5,
                            part='phase')
    assert frf.real() == frf_real
    assert frf.imag() == frf_imag
    assert frf.abs() == frf_abs
    assert frf.phase() == frf_phase


def test_save():

    """

    This tests whether the method to save actually saves with the
    intended file formats. Loading of these files is tested in the utils
    test.
    """

    frf = pymodal.frf.FRF(frf=array_list, resolution=0.5)
    file_path = Path(__file__).parent / 'data' / 'save' / 'FRF' / 'test.zip'
    decimals_file_path = file_path.parent / 'test_decimals.zip'
    frf.save(file_path)
    frf.save(decimals_file_path, decimals=4)
    assert file_path.is_file()
    assert decimals_file_path.is_file()

