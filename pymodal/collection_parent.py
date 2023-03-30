import numpy as np
from pymodal import _signal
from itertools import h5py
from pathlib import Path


class _collection():
    def __init__(self, exp_list: list[_signal], path: Path):

        def __all_equal(iterator):
            iterator = iter(iterator)
            try:
                first = next(iterator)
            except StopIteration:
                return True
            return np.all(first == x for x in iterator)

        self.label = list([exp.label for exp in exp_list])
        assert __all_equal([exp.measurements_units for exp in exp_list])
        self.measurements_units = exp_list[0].measurements_units
        assert __all_equal([exp.method for exp in exp_list])
        self.method = exp_list[0].method
        assert __all_equal([exp.dof for exp in exp_list])
        self.dof = exp_list[0].dof
        assert __all_equal([exp.orientations for exp in exp_list])
        self.orientations = exp_list[0].orientations
        assert __all_equal([exp.coordinates for exp in exp_list])
        self.coordinates = exp_list[0].coordinates
        assert __all_equal([exp.space_units for exp in exp_list])
        self.space_units = exp_list[0].space_units
        assert __all_equal([exp.domain_start for exp in exp_list])
        self.domain_start = exp_list[0].domain_start
        assert __all_equal([exp.domain_end for exp in exp_list])
        self.domain_end = exp_list[0].domain_end
        assert __all_equal([exp.domain_span for exp in exp_list])
        self.domain_span = exp_list[0].domain_span
        assert __all_equal([exp.domain_resolution for exp in exp_list])
        self.domain_resolution = exp_list[0].domain_resolution
        assert __all_equal([exp.domain_array for exp in exp_list])
        self.domain_array = exp_list[0].domain_array
        assert __all_equal([exp.samples for exp in exp_list])
        self.samples = exp_list[0].samples

        self.measurements = h5py.File(path, "w")
        self.measurements = np.array([exp.measurements.m for exp in exp_list]) * self.measurements_units
        

if __name__ == "__main__":
    from pymodal import frf
    
    time = np.arange(0, 30 + 0.05, 0.1)
    signal = np.sin(1 * time)
    signal = np.vstack((signal, np.sin(2 * time)))
    signal = np.vstack((signal, np.sin(3 * time)))
    signal = np.vstack((signal, np.sin(4 * time)))
    signal = np.vstack((signal, np.sin(5 * time)))
    signal = signal.reshape((time.shape[0], -1))
    signal = np.fft.fft(signal, axis=0)
    signal_1 = signal + 1
    signal_2 = signal + 2
    test_object_0 = frf(signal, freq_end=5)
    test_object_1 = frf(signal_1, freq_end=5)
    test_object_2 = frf(signal_2, freq_end=5)
    test_collection = _collection([test_object_0, test_object_1, test_object_2])
    print(test_collection.samples)