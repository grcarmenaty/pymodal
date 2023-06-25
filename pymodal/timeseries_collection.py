from pymodal import _collection, timeseries
from pathlib import Path
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import h5py
from warnings import warn, catch_warnings, filterwarnings
import matplotlib.pyplot as plt

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
                    f[f"measurements/{collection.label[i]}"][()]
                    * collection.measurements_units,
                )
        else:
            setattr(working_instance, attribute, getattr(collection, attribute))
    with h5py.File(collection.path, "a") as f:
        del f[f"measurements/{collection.label[i]}"]
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting"
                " to ndarray.",
            )
            working_instance = working_instance.change_time_span(
                new_min_time, new_max_time
            )
            f[f"measurements/{collection.label[i]}"] = working_instance.measurements
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
                    f[f"measurements/{collection.label[i]}"][()]
                    * collection.measurements_units,
                )
        else:
            setattr(working_instance, attribute, getattr(collection, attribute))
    with h5py.File(collection.path, "a") as f:
        del f[f"measurements/{collection.label[i]}"]
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting"
                " to ndarray.",
            )
            working_instance = working_instance.change_sampling_rate(new_sampling_rate)
            f[f"measurements/{collection.label[i]}"] = working_instance.measurements
    del working_instance.measurements
    return working_instance


class timeseries_collection(_collection):
    def __init__(self, exp_list: list[timeseries], path: Path = Path("temp.h5")):
        super().__init__(exp_list=exp_list, path=path)
        del exp_list

    def change_time_span(self, new_min_time=None, new_max_time=None):
        vars = []
        for i in range(len(self)):
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
        self.file = h5py.File(self.path, "a")
        self.measurements = list(
            [self.file[f"measurements/{label}"] for label in self.label]
        )
        for attribute in attributes_to_match:
            self.file["measurements"].attrs[attribute] = getattr(
                working_instance, attribute
            )
            setattr(self, attribute, getattr(working_instance, attribute))
        del working_instance
        return self

    def change_sampling_rate(self, new_sampling_rate):
        vars = []
        for i in range(len(self)):
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
        self.file = h5py.File(self.path, "a")
        self.measurements = list(
            [self.file[f"measurements/{label}"] for label in self.label]
        )
        for attribute in attributes_to_match:
            self.file["measurements"].attrs[attribute] = getattr(
                working_instance, attribute
            )
            setattr(self, attribute, getattr(working_instance, attribute))
        del working_instance
        return self

    def plot(
            self,
            ax: plt.Axes = None,
            fontname: str = "DejaVu Serif",
            fontsize: float = 12,
            title: str = None,
            title_size: float = 12,
            major_y_locator: int = 4,
            minor_y_locator: int = 4,
            major_x_locator: int = 4,
            minor_x_locator: int = 4,
            color=plt.cm.rainbow,
            linestyle: str = "-",
            ylabel: str = None,
            xlabel: str = None,
            decimals_y: int = 2,
            decimals_x: int = 2,
            bottom_ylim: float = None,
            top_ylim: float = None,
            grid: bool = True,
        ):
        color = iter(color(np.linspace(0, 1, len(self))))
        working_instance = deepcopy(self.collection_class)
        for attribute in self.attributes:
            if attribute == "label":
                setattr(working_instance, attribute, self.label[0])
            elif attribute == "measurements":
                setattr(
                    working_instance,
                    attribute,
                    self.measurements[0][()] * self.measurements_units,
                )
            else:
                setattr(working_instance, attribute, getattr(self, attribute))
        ax, img = working_instance.plot(
            ax=ax,
            fontname=fontname,
            fontsize=fontsize,
            title=title,
            title_size=title_size,
            major_y_locator=major_y_locator,
            minor_y_locator=minor_y_locator,
            major_x_locator=major_x_locator,
            minor_x_locator=minor_x_locator,
            color=next(color),
            linestyle=linestyle,
            ylabel=ylabel,
            xlabel=xlabel,
            decimals_y=decimals_y,
            decimals_x=decimals_x,
            bottom_ylim=bottom_ylim,
            top_ylim=top_ylim,
            grid=grid,
        )
        old_bottom_ylim, old_top_ylim = ax.get_ylim()
        for i, label in enumerate(self.label):
            if i > 0:
                working_instance = deepcopy(self.collection_class)
                for attribute in self.attributes:
                    if attribute == "label":
                        setattr(working_instance, attribute, label)
                    elif attribute == "measurements":
                        setattr(
                            working_instance,
                            attribute,
                            self.measurements[i][()] * self.measurements_units,
                        )
                    else:
                        setattr(working_instance, attribute, getattr(self, attribute))
                ax, img = working_instance.plot(
                    ax=ax,
                    fontname=fontname,
                    fontsize=fontsize,
                    title=title,
                    title_size=title_size,
                    major_y_locator=major_y_locator,
                    minor_y_locator=minor_y_locator,
                    major_x_locator=major_x_locator,
                    minor_x_locator=minor_x_locator,
                    color=next(color),
                    linestyle=linestyle,
                    ylabel=ylabel,
                    xlabel=xlabel,
                    decimals_y=decimals_y,
                    decimals_x=decimals_x,
                    bottom_ylim=bottom_ylim,
                    top_ylim=top_ylim,
                    grid=grid,
                )
                new_bottom_ylim, new_top_ylim = ax.get_ylim()
                if new_bottom_ylim > old_bottom_ylim:
                    ax.set_ylim(bottom=old_bottom_ylim)
                else:
                    old_bottom_ylim = new_bottom_ylim
                if new_top_ylim < old_top_ylim:
                    ax.set_ylim(bottom=old_top_ylim)
                else:
                    old_top_ylim = new_top_ylim
        return ax, img


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
    test_collection.plot()
    plt.show()
    print(test_collection.change_time_span(new_max_time=20).measurements)
    print(test_collection.change_sampling_rate(new_sampling_rate=0.2).measurements)
    print(test_collection[["Vibrational data", "Vibrational data_2"]].measurements)
    print(test_collection[1:-1].measurements)
    print(test_collection["Vibrational data"].measurements)
    test_collection.close()
