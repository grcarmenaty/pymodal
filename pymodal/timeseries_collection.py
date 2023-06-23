from pymodal import _collection, timeseries
from pathlib import Path
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import h5py
from warnings import warn, catch_warnings, filterwarnings

num_processes = cpu_count()


def change_time_span(var):
    collection, i, new_min_time, new_max_time = var
    working_instance = deepcopy(collection.collection_class)
    for attribute in collection.attributes:
        if attribute == "label":
            setattr(working_instance, attribute, collection.label[i])
        elif attribute == "measurements":
            with h5py.File(collection.path, "r") as f:
                setattr(
                    working_instance,
                    attribute,
                    f[collection.label[i]][()] * collection.measurements_units,
                )
        else:
            setattr(working_instance, attribute, getattr(collection, attribute))
    with h5py.File(collection.path, "a") as f:
        del f[collection.label[i]]
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting"
                " to ndarray.",
            )
            working_instance = working_instance.change_time_span(
                new_min_time, new_max_time
            )
            f[collection.label[i]] = working_instance.measurements
    del working_instance.measurements
    return working_instance


def change_sampling_rate(var):
    collection, i, new_sampling_rate = var
    working_instance = deepcopy(collection.collection_class)
    for attribute in collection.attributes:
        if attribute == "label":
            setattr(working_instance, attribute, collection.label[i])
        elif attribute == "measurements":
            with h5py.File(collection.path, "r") as f:
                setattr(
                    working_instance,
                    attribute,
                    f[collection.label[i]][()] * collection.measurements_units,
                )
        else:
            setattr(working_instance, attribute, getattr(collection, attribute))
    with h5py.File(collection.path, "a") as f:
        del f[collection.label[i]]
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting"
                " to ndarray.",
            )
            working_instance = working_instance.change_sampling_rate(new_sampling_rate)
            f[collection.label[i]] = working_instance.measurements
    del working_instance.measurements
    return working_instance


class timeseries_collection(_collection):
    def __init__(self, exp_list: list[timeseries], path: Path = Path("temp.h5")):
        super().__init__(exp_list=exp_list, path=path)
        del exp_list

    def change_time_span(self, new_min_time=None, new_max_time=None):
        vars = []
        for i in range(len(self.measurements)):
            vars.append((self, i, new_min_time, new_max_time))
        self.file.close()
        del self.file
        del self.measurements
        # for var in vars:
        #     change_time_span(var)
        with Pool(num_processes) as pool:
            working_instance = pool.map(change_time_span, vars)
        working_instance = working_instance[0]
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("label")
        for attribute in attributes_to_match:
            setattr(self, attribute, getattr(working_instance, attribute))
        self.file = h5py.File(self.path, "r")
        self.measurements = list([self.file[f"measurements/{label}"] for label in self.label])

    def change_sampling_rate(self, new_sampling_rate):
        vars = []
        for i in range(len(self.measurements)):
            vars.append((self, i, new_sampling_rate))
        self.file.close()
        del self.file
        del self.measurements
        with Pool(num_processes) as pool:
            working_instance = pool.map(change_sampling_rate, vars)
        working_instance = working_instance[0]
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("label")
        for attribute in attributes_to_match:
            setattr(self, attribute, getattr(working_instance, attribute))
        self.file = h5py.File(self.path, "r")
        self.measurements = list([self.file[f"measurements/{label}"] for label in self.label])


if __name__ == "__main__":

    time = np.arange(0, 30 + 0.05, 0.1)
    signal = np.sin(1 * time)
    signal = np.vstack((signal, np.sin(2 * time)))
    signal = np.vstack((signal, np.sin(3 * time)))
    signal = np.vstack((signal, np.sin(4 * time)))
    signal = np.vstack((signal, np.sin(5 * time)))
    signal = signal.reshape((time.shape[0], -1))
    signal_1 = signal + 1
    signal_2 = signal + 2
    test_object = timeseries(signal, time_end=30)
    test_object_1 = timeseries(signal_1, time_end=30)
    test_object_2 = timeseries(signal_2, time_end=30)
    test_collection = timeseries_collection([test_object, test_object_1, test_object_2])
    print(test_collection.measurements)
    test_collection.change_time_span(new_max_time=20)
    print(test_collection.measurements)
    test_collection.change_sampling_rate(new_sampling_rate=0.2)
    print(test_collection.measurements)
    test_collection.close()
