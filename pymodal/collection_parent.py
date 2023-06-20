import numpy as np
from pymodal import _signal
import h5py
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pint

def save_array(array_info):
    array, dataset_name, file_name = array_info
    with h5py.File(file_name, "a") as hf:
        hf.create_dataset(dataset_name, data=array)

def process_strings(args):
    strings, start_index, end_index = args
    counter = {}
    result = []

    for i in range(start_index, end_index):
        string = strings[i]
        if string in counter:
            count = counter[string]
            new_string = f"{string}_{count}"
            counter[string] += 1
            result.append(new_string)
        else:
            counter[string] = 1
            result.append(string)

    return result

num_processes = cpu_count()
attributes_to_match = ["measurements_units", "method", "dof", "orientations", "coordinates", "space_units", "domain_start", "domain_end", "domain_span", "domain_resolution", "domain_array", "samples"]

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
    instance1, instance2 = pair
    return attributes_match(instance1, instance2, attributes_to_match)

def parallel_attributes_match(instances):
    first_instance = instances[0]
    remaining_instances = instances[1:]

    with Pool(num_processes) as pool:
        results = pool.map(worker, [(first_instance, instance) for instance in remaining_instances])

    if np.all(results):
        return True
    return False

class _collection():
    def __init__(self, exp_list: list[_signal], path: Path = Path("temp.h5")):

        self.path = Path(path)
        if self.path.exists():
            self.path.unlink()

        self.label = list([exp.label for exp in exp_list])

        string_count = len(self.label)
        chunk_size = string_count // num_processes
        start_indices = [i * chunk_size for i in range(num_processes)]
        end_indices = start_indices[1:] + [string_count]

        pool = Pool(processes=num_processes)
        args = [(self.label, start, end) for start, end in zip(start_indices, end_indices)]
        results = pool.map(process_strings, args)
        pool.close()
        pool.join()

        self.label = list([string for sublist in results for string in sublist])
        
        assert parallel_attributes_match(exp_list)
        
        array_info = [(array, f"{self.label[i]}", self.path) for i, array in enumerate([exp.measurements for exp in exp_list])]
        with Pool(num_processes) as pool:
            pool.map(save_array, array_info)
        pool.close()
        pool.join()
        self.file = h5py.File(self.path, "r")
        self.measurements = list([self.file[f"{label}"] for label in self.label])

    def close(self):
        self.file.close()
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
    test_object_0 = frf(signal, freq_end=5)
    test_object_1 = frf(signal_1, freq_end=5)
    test_object_2 = frf(signal_2, freq_end=5)
    test_collection = _collection([test_object_0, test_object_1, test_object_2])
    print(test_collection.measurements[0])
    test_collection.close()