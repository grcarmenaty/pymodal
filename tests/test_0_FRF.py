import numpy as np
import ntpath
import pathlib
import pymodal
import pytest
import matplotlib.pyplot as plt
import admin
import os

# I synch this folder with Google Drive, so I need to run it as admin because
# of the way the save function is written, which needs to delete files that are
# being synched.
if os.name == 'nt':
    if not admin.isUserAdmin():
            admin.runAsAdmin()

# Build a list of all FRF.mat files in the data folder. All of the used FRF
# files have 81 lines of 0Hz to 3200Hz bands with a resolution of 0.5Hz, which
# implies the shape of the arrays is (6401, 81)
data_path = (pathlib.Path(__file__).parent / 'data' / 'FRF').glob('**/*')
files = [file for file in data_path if file.is_file()]
length = len(files)
# This is expected to be the list of names in each FRF object in which a list
# of files is used as input
name = [ntpath.split(item)[-1] for item in files]
unknown_name = [f'Unknown name {i + 1}' for i in range(length)]
# This is the expected value to be stored in a FRF object using the file list
array_3d = [pymodal.load_array(file) for file in files]
array_3d = np.dstack(array_3d)


def test_init():
    """
    Check every value assigned at initialization has the value it is
    believed to have.
    """
    frf = pymodal.FRF(frf=array_3d, resolution=0.5, name=name)
    assert np.array_equal(frf.value, array_3d)
    assert frf.resolution == 0.5
    assert frf.max_freq == 3200
    assert frf.min_freq == 0
    assert frf.bandwidth == 3200
    assert frf.name == name
    assert frf.lines == 81
    freq_vector = np.arange(0, 3200 + 0.5 / 2, 0.5)
    assert np.array_equal(freq_vector, frf.freq_vector)
    assert frf.part == 'complex'
    frf = pymodal.FRF(frf=array_3d, min_freq=1000, max_freq=7400)
    # Values have not changed, so they should be the same as before
    assert np.array_equal(frf.value, array_3d)
    assert frf.resolution == 1
    assert frf.max_freq == 7400
    assert frf.min_freq == 1000
    assert frf.bandwidth == 6400
    assert frf.name == unknown_name  # No names specified
    assert frf.lines == 81
    freq_vector = np.arange(1000, 7400 + 1 / 2, 1)
    assert np.array_equal(freq_vector, frf.freq_vector)
    assert frf.part == 'complex'
    # Make sure an impossible definition raises an error
    try:
        frf = pymodal.FRF(frf=array_3d, resolution=0.5, max_freq=6400)
        assert False
    except Exception as __:  # noqa F841
        assert True
    # Try to give more names than needed
    try:
        frf = pymodal.FRF(frf=array_3d[0:2], resolution=0.5, name=name)
        assert False
    except Exception as __:  # noqa F841
        assert True
    # Give less names and make sure it still generates names
    frf = pymodal.FRF(frf=array_3d, resolution=0.5, name=name[0:2])
    assert len(frf.name) == array_3d.shape[2]


def test_equality():
    """
    Check that two instances equally created are equal and changing any
    of the arguments' default creates a not equal object.
    """
    frf = pymodal.FRF(frf=array_3d[:, :, 0:4], resolution=0.5)
    same_frf = pymodal.FRF(frf=array_3d[:, :, 0:4], resolution=0.5)
    other_frf = pymodal.FRF(frf=array_3d[:, :, 0:2], resolution=0.5)
    other_resolution = pymodal.FRF(frf=array_3d[:, :, 0:4], resolution=1)
    other_max_freq = pymodal.FRF(frf=array_3d[:, :, 0:4], max_freq=6400)
    other_min_freq = pymodal.FRF(frf=array_3d[:, :, 0:4], resolution=0.5,
                                     min_freq=9)
    other_bandwidth = pymodal.FRF(frf=array_3d[:, :, 0:4], bandwidth=4200)
    other_name = pymodal.FRF(frf=array_3d[:, :, 0:4], resolution=0.5,
                                 name=name[0:4])
    # Doing this is strongly not recommended, part and value should always be
    # related (e.g. if values correspond to the phase, part should be phase,
    # defining the class this way can lead to easily made mistakes)
    other_part = pymodal.FRF(frf=array_3d[:, :, 0:4],
                                 resolution=0.5, part='real')
    assert frf == same_frf
    assert frf != other_frf
    assert frf != other_resolution
    assert frf != other_max_freq
    assert frf != other_min_freq
    assert frf != other_bandwidth
    assert frf != other_name
    assert frf != other_part


def test_slice():
    """
    Check that a slice or an index yield the same as a slice or index of
    the input FRF.
    """
    frf = pymodal.FRF(frf=array_3d, resolution=0.5, name=name)
    frf_selected = pymodal.FRF(frf=array_3d[:, :, 5],
                                   resolution=0.5,
                                   name=[name[5]])
    frf_sliced = pymodal.FRF(frf=array_3d[:, :, 5:15], resolution=0.5,
                                 name=name[5:15])
    assert frf[5] == frf_selected
    assert frf[5:15] == frf_sliced


def test_len():
    """
    Check that the length of the object is the same as the length of the
    input.
    """
    frf = pymodal.FRF(frf=array_3d, resolution=0.5, name=name)
    assert len(frf) == array_3d.shape[2]


def test_extend():
    """
    Check that extending an FRF in the same way a list is extended is
    the same as extending the input and the name list, and that this
    works whether ot not names are defined in various combinations.
    """
    extended_array_3d = np.dstack((array_3d, array_3d[:, :, 0:2]))
    extended_name = list(name)
    extended_name.extend(name[0:2])

    # With names
    frf = pymodal.FRF(frf=array_3d, resolution=0.5, name=name)
    frf_extended = pymodal.FRF(frf=extended_array_3d, resolution=0.5,
                                   name=extended_name)
    frf = frf.extend(array_3d[:, :, 0:2], name[0:2])
    assert frf == frf_extended

    # Without names
    frf = pymodal.FRF(frf=array_3d, resolution=0.5)
    frf_extended = pymodal.FRF(frf=extended_array_3d, resolution=0.5)
    frf = frf.extend(array_3d[:, :, 0:2])
    assert frf == frf_extended

    # With some names defined in parent object
    frf = pymodal.FRF(frf=array_3d, resolution=0.5, name=name[0:15])
    frf_extended = pymodal.FRF(frf=extended_array_3d, resolution=0.5,
                                   name=name[0:15])
    frf = frf.extend(array_3d[:, :, 0:2])
    assert frf == frf_extended

    # With names defined only in extension
    frf = pymodal.FRF(frf=array_3d, resolution=0.5)
    extended_unknown_name = list(unknown_name)
    extended_unknown_name.extend(name[0:2])
    frf_extended = pymodal.FRF(frf=extended_array_3d, resolution=0.5,
                                   name=extended_unknown_name)
    frf = frf.extend(array_3d[:, :, 0:2], name[0:2])
    assert frf == frf_extended


def test_normalize():
    frf = pymodal.FRF(frf=array_3d, resolution=0.5)
    try:
        frf.real().normalize()
        assert False
    except Exception as __:
        assert True
    frf_normalized = np.array(array_3d)
    for i in range(frf_normalized.shape[2]):
        frf_normalized[:, :, i] = (frf_normalized[:, :, i] /
            np.amax(np.abs(frf_normalized[:, :, i])))
    assert np.array_equal(frf_normalized, frf.normalize().value)


def test_change_resolution():
    """
    Check that changing the resolution works the same as changing the
    resolution of the FRF manually.
    """
    frf = pymodal.FRF(frf=array_3d, resolution=0.5)
    with pytest.warns(UserWarning):
        frf_changed = frf.change_resolution(1.2)
    changed_array_3d = array_3d[0::int(np.around(1.2 / 0.5)), :, :]
    freq_vector = np.arange(0, 3200 + 1 / 2, 1)
    assert np.array_equal(freq_vector, frf_changed.freq_vector)
    assert frf_changed.resolution == 1
    assert np.array_equal(frf_changed.value, changed_array_3d)
    assert frf_changed.value.shape == (3201, 81, len(frf_changed))


def test_change_lines():
    """
    Check that reducing the amount of mesh points works the same as
    doing the same to the input and then creating the object.
    """
    frf = pymodal.FRF(frf=array_3d, resolution=0.5)
    new_lines = list(range(10, 40, 5))
    new_lines.append(62)
    frf_changed = frf.change_lines(new_lines)
    changed_array_3d = array_3d[:, new_lines, :]
    value_test = [(frf_changed.value[i] == changed_array_3d[i]).all()
                  for i in range(length)]
    assert np.array_equal(frf_changed.value, changed_array_3d)
    assert frf_changed.value.shape == (6401, 7, len(frf_changed))
    assert frf_changed.lines == 7


def test_change_frequencies():
    """
    Check that changing the frequency range works the same as doing the
    same to the input and then creating the object.
    """
    frf = pymodal.FRF(frf=array_3d, resolution=0.5)

    try:  # Length 3 entry should raise an error
        frf_changed = frf.change_frequencies([1000, 2000, 3000])
        assert False
    except Exception as __:  # noqa F841
        assert True

    try:  # Length 1 entry should raise an error
        frf_changed = frf.change_frequencies([1000])
        assert False
    except Exception as __:  # noqa F841
        assert True

    try:  # Non-iterable entry should raise an error
        frf_changed = frf.change_frequencies(1000)
        assert False
    except Exception as __:  # noqa F841
        assert True

    frf_changed = frf.change_frequencies((1000, 2000))
    changed_array_3d = array_3d[2000:4001, :, :]
    assert frf_changed.min_freq == 1000
    assert frf_changed.max_freq == 2000
    assert frf_changed.bandwidth == 1000
    assert frf_changed.resolution == 0.5
    freq_vector = np.arange(1000, 2000 + 0.5 / 2, 0.5)
    assert np.array_equal(freq_vector, frf_changed.freq_vector)
    assert np.array_equal(frf_changed.value, changed_array_3d)
    assert frf_changed.value.shape == (2001, 81, len(frf_changed))


def test_part():
    """
    Check that selecting some part of the FRF as complex works the same
    as doing the same to the input and then creating the object.
    """

    frf = pymodal.FRF(frf=array_3d, resolution=0.5)
    frf.plot()
    frf_real = pymodal.FRF(frf=array_3d.real,
                           resolution=0.5,
                           part='real')
    frf_imag = pymodal.FRF(frf=array_3d.imag,
                           resolution=0.5,
                           part='imag')
    frf_abs = pymodal.FRF(frf=np.absolute(array_3d),
                          resolution=0.5,
                          part='abs')
    frf_phase = pymodal.FRF(np.angle(array_3d),
                            resolution=0.5,
                            part='phase')
    assert frf.real() == frf_real
    assert frf.imag() == frf_imag
    assert frf.abs() == frf_abs
    assert frf.phase() == frf_phase


def test_get_cfdac():
    frf = pymodal.FRF(frf=array_3d, resolution=0.5).change_resolution(10)
    cfdac = np.dstack((pymodal.value_CFDAC(frf[1].value[:, :, 0],
                                           frf[0].value[:, :, 0]),
                       pymodal.value_CFDAC(frf[1].value[:, :, 0],
                                           frf[1].value[:, :, 0]),
                       pymodal.value_CFDAC(frf[1].value[:, :, 0],
                                           frf[2].value[:, :, 0])))
    cfdac_method = frf[0:3].get_CFDAC(1)
    assert np.array_equal(cfdac_method, cfdac)


def test_get_sci():
    #! This should test all possible SCIs: abs, real, imag
    frf = pymodal.FRF(frf=array_3d, resolution=0.5).change_resolution(10)
    pristine = np.abs(pymodal.value_CFDAC(frf[1].value[:, :, 0],
                                   frf[1].value[:, :, 0]))
    sci = np.array([pymodal.SCI(pristine,
                                np.abs(
                                    pymodal.value_CFDAC(frf[1].value[:, :, 0],
                                    frf[0].value[:, :, 0])
                                )),
                    pymodal.SCI(pristine,
                                np.abs(
                                    pymodal.value_CFDAC(frf[1].value[:, :, 0],
                                    frf[1].value[:, :, 0])
                                )),
                    pymodal.SCI(pristine,
                                np.abs(
                                    pymodal.value_CFDAC(frf[1].value[:, :, 0],
                                                    frf[2].value[:, :, 0])
                                ))])
    sci_method = frf[0:3].get_SCI(1)
    assert np.array_equal(sci_method, sci)


def test_plot():
    """
    Check the plots that are created by the class object method. Should
    be checked manually.
    """
    save_dir = pathlib.Path(__file__).parents[1] / 'result_images' / 'frf_plot'
    frf = pymodal.FRF(frf=array_3d, resolution=0.5)

    frf[0].plot()
    plt.close()
    
    frf[0].plot()
    file_path = save_dir / 'one_frf.png'
    plt.savefig(file_path)
    plt.close()
    assert file_path.is_file()

    frf[0:2].real().plot()
    file_path = save_dir / 'two_frf.png'
    plt.savefig(file_path)
    plt.close()
    assert file_path.is_file()

    frf[0:3].imag().plot()
    file_path = save_dir / 'three_frf.png'
    plt.savefig(file_path)
    plt.close()
    assert file_path.is_file()

    fig, ax = plt.subplots(2, 1, figsize=(10,10))
    plt.title = 'Frequency Response Function'
    frf[0].abs().plot(ax=ax[0], title='Magnitude')
    frf[0].phase().plot(ax=ax[1], title='Phase')
    plt.tight_layout()
    file_path = save_dir / 'frf_mag_phase.png'
    plt.savefig(file_path)
    plt.close()
    assert file_path.is_file()


def test_save():
    """
    Check that a file is created in the intended file path.
    """
    frf = pymodal.FRF(frf=array_3d, resolution=0.5)
    file_path = pathlib.Path(__file__).parent
    file_path = file_path / 'data' / 'save' / 'FRF' / 'test.zip'
    decimals_file_path = file_path.parent / 'test_decimals.zip'
    frf.save(file_path)
    frf.save(decimals_file_path, decimals=4)
    assert file_path.is_file()
    assert decimals_file_path.is_file()
