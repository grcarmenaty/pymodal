import numpy as np
from pymodal import _signal
import h5py
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pint
import os
import inspect
from copy import deepcopy

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def save_array(array_info):
    array, dataset_name, file_name = array_info
    with h5py.File(file_name, "a") as hf:
        hf.create_dataset(dataset_name, data=array)


def add_suffix(strings):
    counter = {}
    result = []

    for string in strings:
        if string in counter:
            count = counter[string]
            new_string = f"{string}_{count}"
            counter[string] += 1
            result.append(new_string)
        else:
            counter[string] = 1
            result.append(string)

    return result


def get_attributes(obj):
    attributes = []
    for name, value in inspect.getmembers(obj):
        if (
            not name.startswith("__")
            and not inspect.ismethod(value)
        ):
            attributes.append(name)
    return attributes


num_processes = cpu_count()

# Check if specified attributes match
def attributes_match(instance1, instance2, attributes_to_match):
    for attribute in attributes_to_match:
        value1 = getattr(instance1, attribute)
        value2 = getattr(instance2, attribute)

        if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
            if not np.array_equal(value1, value2):
                return False
        elif isinstance(value1, pint.Quantity) and isinstance(value2, pint.Quantity):
            if not np.array_equal(value1.magnitude, value2.magnitude):
                return False
        elif value1 != value2:
            return False

    return True


def worker(pair):
    instance1, instance2, attributes_to_match = pair
    return attributes_match(instance1, instance2, attributes_to_match)


def parallel_attributes_match(instances, attributes_to_match):
    first_instance = instances[0]
    remaining_instances = instances[1:]

    with Pool(num_processes) as pool:
        results = pool.map(
            worker,
            [
                (first_instance, instance, attributes_to_match)
                for instance in remaining_instances
            ],
        )

    if np.all(results):
        return True
    return False


class _collection:
    def __init__(self, exp_list: list[_signal], path: Path = Path("temp.h5")):

        self.path = Path(path)
        if self.path.exists():
            self.path.unlink()

        self.label = add_suffix(list([exp.label for exp in exp_list]))

        self.attributes = get_attributes(exp_list[0])
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("label")
        assert parallel_attributes_match(exp_list, attributes_to_match)
        for attribute in attributes_to_match:
            setattr(self, attribute, getattr(exp_list[0], attribute))
        array_info = [
            (array.magnitude, f"measurements/{self.label[i]}", self.path)
            for i, array in enumerate([exp.measurements for exp in exp_list])
        ]
        with Pool(num_processes) as pool:
            pool.map(save_array, array_info)
        exp_list = exp_list[0]
        self.file = h5py.File(self.path, "a")
        self.measurements = list([self.file[f"measurements/{label}"] for label in self.label])
        self.collection_class = exp_list
        for attribute in self.attributes:
            if attribute not in ["measurements", "label"]:
                self.file["measurements"].attrs[attribute] = getattr(exp_list, attribute)
            setattr(self.collection_class, attribute, None)
        del exp_list
        
    # def __getitem__(self, key: tuple[slice]):
    #     self.file.close()
    #     self_copy = deepcopy(self)  # Make a deepcopy of self to work on it.
    #     # Make sure key is a list of slices. If it isn't, turn it into one.
    #     self.label = 
    #     self.file = h5py.File(self.path, "r")
    #     self.measurements = list([self.file[f"measurements/{label}"] for label in self.label])
    #     if type(key) is int:
    #         key = slice(key, key + 1)
    #     if type(key) is slice:
    #         key = [key]
    #     key = list(key)
    #     for i, index in enumerate(key):
    #         if type(index) is int:
    #             key[i] = slice(index, index + 1)
    #     # If only one key is provided, it is assumed to refer to an output selection,
    #     # unless the system type is supposed to have only one input, in which case it
    #     # will be assumed to refer to an input selection. If two keys are provided, the
    #     # first one is assumed to refer to an output, the second to an input.
    #     if len(key) == 1:
    #         if self.method in ["SIMO", "MIMO"]:
    #             self_copy.measurements = self.measurements[:, key[0], :]
    #             self_copy.coordinates = self.coordinates[:, key[0]]
    #             self_copy.orientations = self.orientations[:, key[0]]
    #         elif self.method in ["MISO", "excitation"]:
    #             self_copy.measurements = self.measurements[:, :, key[0]]
    #             self_copy.coordinates = self.coordinates[:, key[0]]
    #             self_copy.orientations = self.orientations[:, key[0]]
    #     elif len(key) == 2:
    #         self_copy.measurements = self.measurements[:, key[0], key[1]]
    #         self_copy.coordinates = self.coordinates[:, key[0], key[1]]
    #         self_copy.orientations = self.orientations[:, key[0], key[1]]
    #     else:
    #         raise ValueError("Too many keys provided.")
    #     return self_copy
    
    def close(self, keep: bool = False):

        self.file.close()
        if not keep:
            self.path.unlink()


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
    test_object_0 = _signal(signal, domain_end=5)
    test_object_1 = _signal(signal_1, domain_end=5)
    test_object_2 = _signal(signal_2, domain_end=5)
    test_collection = _collection([test_object_0, test_object_1, test_object_2])
    print(test_collection.measurements)
    print(list(test_collection.file["measurements"].attrs.items()))
    test_collection.close()
