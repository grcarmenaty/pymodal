import numpy as np
from pymodal import _signal
import h5py
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pint
import os
import inspect
from copy import deepcopy
from warnings import warn, catch_warnings, filterwarnings


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def save_array(array_info):
    array, dataset_name, file_name = array_info
    with h5py.File(file_name, "a") as hf:
        if dataset_name in hf:
            del hf[dataset_name]
        hf[dataset_name] = array


def add_suffix(strings):
    counter = {}
    result = []

    for string in strings:
        flag = string in counter
        if flag:
            while flag:
                count = counter[string]
                new_string = f"{string}_{count}"
                counter[string] += 1
                flag = new_string in counter
                if flag:
                    counter[string] += 1
            result.append(new_string)
        else:
            counter[string] = 1
            result.append(string)

    return result


def get_attributes(obj):
    attributes = []
    for name, value in inspect.getmembers(obj):
        if not name.startswith("__") and not inspect.ismethod(value):
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
    results = []
    for instance in remaining_instances:
        results.append(worker((first_instance, instance, attributes_to_match)))
    # with Pool(num_processes) as pool:
    #     results = pool.map(
    #         worker,
    #         [
    #             (first_instance, instance, attributes_to_match)
    #             for instance in remaining_instances
    #         ],
    #     )

    if np.all(results):
        return True
    return False


class _signal_collection:
    def __init__(self, exp_list: list[_signal], path: Path = Path("temp.h5")):

        self.path = Path(path)
        if self.path.exists():
            self.path.unlink()

        self.name = add_suffix(list([exp.name for exp in exp_list]))

        self.attributes = get_attributes(exp_list[0])
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("name")
        assert parallel_attributes_match(exp_list, attributes_to_match)
        for attribute in attributes_to_match:
            setattr(self, attribute, getattr(exp_list[0], attribute))
        array_info = [
            (array.magnitude, f"measurements/{self.name[i]}", self.path)
            for i, array in enumerate([exp.measurements for exp in exp_list])
        ]
        # save_array(array_info[0])
        # with Pool(num_processes) as pool:
        #     pool.map(save_array, array_info[1:])
        for array in array_info:
            save_array(array)
        exp_list = exp_list[0]
        self.file = h5py.File(self.path, "a")
        self.measurements = list(
            [self.file[f"measurements/{name}"] for name in self.name]
        )
        self.collection_class = exp_list
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting"
                " to ndarray.",
            )
            for attribute in self.attributes:
                if attribute not in ["measurements", "name"]:
                    self.file["measurements"].attrs[attribute] = getattr(
                        exp_list, attribute
                    )
                setattr(self.collection_class, attribute, None)
        del exp_list

    def __len__(self):
        return len(self.name)

    def __getitem__(self, key: tuple[slice]):
        if type(key) is str:
            key = [key]
        if type(key) is set or type(key) is list:
            self.name = list(key)
            self.measurements = list(
                [self.file[f"measurements/{name}"] for name in self.name]
            )
        else:
            if type(key) is int:
                key = slice(key, key + 1)
            if type(key) is slice:
                key = [key]
            key = list(key)
            for i, index in enumerate(key):
                if type(index) is int:
                    key[i] = slice(index, index + 1)
            # If only one key is provided, it is assumed to refer to an output selection,
            # unless the system type is supposed to have only one input, in which case it
            # will be assumed to refer to an input selection. If two keys are provided, the
            # first one is assumed to refer to an output, the second to an input.
            if len(key) == 1:
                if self.method in ["SIMO"]:
                    for i, measurement in enumerate(self.measurements):
                        del self.file[f"measurements/{self.name[i]}"]
                        self.file[f"measurements/{self.name[i]}"] = measurement[
                            :, key[0], :
                        ]
                    self.coordinates = self.coordinates[key[0], :]
                    self.orientations = self.orientations[key[0], :]
                elif self.method in ["MIMO"]:
                    for i, measurement in enumerate(self.measurements):
                        del self.file[f"measurements/{self.name[i]}"]
                        self.file[f"measurements/{self.name[i]}"] = measurement[
                            :, key[0], :
                        ]
                    self.coordinates = self.coordinates[key[0], :, :]
                    self.orientations = self.orientations[key[0], :, :]
                elif self.method in ["MISO", "excitation"]:
                    for i, measurement in enumerate(self.measurements):
                        del self.file[f"measurements/{self.name[i]}"]
                        self.file[f"measurements/{self.name[i]}"] = measurement[
                            :, :, key[0]
                        ]
                    self.coordinates = self.coordinates[:, key[0]]
                    self.orientations = self.orientations[:, key[0]]
                self.measurements = list(
                    [self.file[f"measurements/{name}"] for name in self.name]
                )
            elif len(key) == 2:
                for i, measurement in enumerate(self.measurements):
                    del self.file[f"measurements/{self.name[i]}"]
                    self.file[f"measurements/{self.name[i]}"] = measurement[
                        :, key[0], key[1]
                    ]
                self.coordinates = self.coordinates[:, key[0], key[1]]
                self.orientations = self.orientations[:, key[0], key[1]]
                self.measurements = list(
                    [self.file[f"measurements/{name}"] for name in self.name]
                )
            else:
                raise ValueError("Too many keys provided.")
            self.dof = max(self.measurements[0].shape[1], self.measurements[0].shape[2])
            self.file["measurements"].attrs["dof"] = self.dof
            self.file["measurements"].attrs["coordinates"] = self.coordinates
            self.file["measurements"].attrs["orientations"] = self.orientations
        return self

    def close(self, keep: bool = False):

        self.file.close()
        if not keep:
            self.path.unlink()

    def append(self, signal: _signal):
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("name")
        assert attributes_match(self, signal, attributes_to_match)
        self.name.append(signal.name)
        self.name = add_suffix(self.name)
        self.file[f"measurements/{self.name[-1]}"] = signal.measurements
        self.measurements.append(self.file[f"measurements/{self.name[-1]}"])
        return self


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
    signal_1 = signal * 2
    signal_2 = signal * 4
    signal_3 = signal * 6
    test_object_0 = _signal(signal, domain_end=5)
    test_object_1 = _signal(signal_1, domain_end=5)
    test_object_2 = _signal(signal_2, domain_end=5)
    test_object_3 = _signal(signal_2, domain_end=5)
    test_collection = _signal_collection([test_object_0, test_object_1, test_object_2])
    print(test_collection.measurements)
    print(test_collection.append(test_object_3).measurements)
    print(list(test_collection.file["measurements"].attrs.items()))
    print(test_collection[["Vibrational data", "Vibrational data_3"]].measurements)
    print(test_collection[1:-1].measurements)
    print(test_collection["Vibrational data"].measurements)
    test_collection.close()