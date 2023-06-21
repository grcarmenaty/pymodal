from pymodal import _collection, timeseries
from pathlib import Path
import numpy as np
from copy import deepcopy

def change_sampling_rate(var):
    collection, i, new_sampling_rate = var
    working_instance = deepcopy(collection.collection_class)
    for attribute in collection.attributes:
        if attribute=="label":
            setattr(working_instance, attribute, collection.label[i])
        elif attribute=="measurements":
            setattr(working_instance, attribute, collection.measurements[i][()] * collection.measurements_units)
        else:
            setattr(working_instance, attribute, getattr(collection, attribute))
    del collection.file[f"{collection.label[i]}"]
    collection.file[f"{collection.label[i]}"] = working_instance.change_sampling_rate(new_sampling_rate).measurements
    

class timeseries_collection(_collection):
    def __init__(self, exp_list: list[timeseries], path: Path = Path("temp.h5")):
        super().__init__(exp_list=exp_list, path=path)
        del exp_list

    def change_sampling_rate(self, new_sampling_rate):
        vars = []
        for i in range(len(self.measurements)):
            vars.append((self, i, new_sampling_rate))

        for var in vars:
            change_sampling_rate(var)
        
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
    test_collection.change_sampling_rate(new_sampling_rate=0.2)
    print(test_collection.measurements)
    test_collection.close()