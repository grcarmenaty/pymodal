from pymodal import _signal_collection, timeseries, frf_collection, timeseries_collection
from pathlib import Path
import numpy as np
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import h5py
from warnings import warn, catch_warnings, filterwarnings
import matplotlib.pyplot as plt
from typing import Optional
from audiomentations import Compose, AddGaussianNoise
import random

num_processes = cpu_count()


def change_time_span(var):
    collection, i, new_min_time, new_max_time = var
    working_instance = deepcopy(collection.collection_class)
    for attribute in collection.attributes:
        if attribute == "name":
            setattr(working_instance, attribute, collection.name[i])
        elif attribute == "measurements":
            with h5py.File(collection.path, "r") as f:
                setattr(
                    working_instance,
                    attribute,
                    f[f"measurements/{collection.name[i]}/data"][()]
                    * collection.measurements_units,
                )
        else:
            setattr(working_instance, attribute, getattr(collection, attribute))
    with h5py.File(collection.path, "a") as f:
        del f[f"measurements/{collection.name[i]}/data"]
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting"
                " to ndarray.",
            )
            working_instance = working_instance.change_time_span(
                new_min_time, new_max_time
            )
            f[f"measurements/{collection.name[i]}/data"] = working_instance.measurements
    del working_instance.measurements
    return working_instance


def change_sampling_rate(var):
    collection, i, new_sampling_rate = var
    working_instance = deepcopy(collection.collection_class)
    for attribute in collection.attributes:
        if attribute == "name":
            setattr(working_instance, attribute, collection.name[i])
        elif attribute == "measurements":
            with h5py.File(collection.path, "r") as f:
                setattr(
                    working_instance,
                    attribute,
                    f[f"measurements/{collection.name[i]}/data"][()]
                    * collection.measurements_units,
                )
        else:
            setattr(working_instance, attribute, getattr(collection, attribute))
    with h5py.File(collection.path, "a") as f:
        del f[f"measurements/{collection.name[i]}/data"]
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting"
                " to ndarray.",
            )
            working_instance = working_instance.change_sampling_rate(new_sampling_rate)
            f[f"measurements/{collection.name[i]}/data"] = working_instance.measurements
    del working_instance.measurements
    return working_instance


class timeseries_collection(_signal_collection):
    def __init__(self, exp_list: list[timeseries], labels: Optional[list[float]] = None, path: Path = Path("temp.h5")):
        super().__init__(exp_list=exp_list, labels=labels, path=path)
        del exp_list

    def change_time_span(self, new_min_time=None, new_max_time=None):
        vars = []
        for i in range(len(self)):
            vars.append((self, i, new_min_time, new_max_time))
        self.file.close()
        del self.file
        del self.measurements
        for var in vars:
            working_instance = change_time_span(var)
        # with Pool(num_processes) as pool:
        #     working_instance = pool.map(change_time_span, vars)
        # working_instance = working_instance[0]
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("name")
        self.file = h5py.File(self.path, "a")
        self.measurements = list(
            [self.file[f"measurements/{name}/data"] for name in self.name]
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
        for var in vars:
            working_instance = change_sampling_rate(var)
        # with Pool(num_processes) as pool:
        #     working_instance = pool.map(change_sampling_rate, vars)
        # working_instance = working_instance[0]
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("name")
        self.file = h5py.File(self.path, "a")
        self.measurements = list(
            [self.file[f"measurements/{name}/data"] for name in self.name]
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
            if attribute == "name":
                setattr(working_instance, attribute, self.name[0])
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
        for i, name in enumerate(self.name):
            if i > 0:
                working_instance = deepcopy(self.collection_class)
                for attribute in self.attributes:
                    if attribute == "name":
                        setattr(working_instance, attribute, name)
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
                    ax.set_ylim(top=old_top_ylim)
                else:
                    old_top_ylim = new_top_ylim
        return ax, img

    def to_FRF(
        self,
        excitation: timeseries_collection,
        FRF_type: str = "H1",
        resp_delay: int = 0,
        new_path: Optional[Path] = None,
    ):
        if new_path is None:
            new_path = self.path.parent / f"{self.path.stem}_frf.h5"
        working_instance = deepcopy(self.collection_class)
        for attribute in self.attributes:
            if attribute == "name":
                setattr(working_instance, attribute, self.name[0])
            elif attribute == "measurements":
                setattr(
                    working_instance,
                    attribute,
                    self.measurements[0][()] * self.measurements_units,
                )
            else:
                setattr(working_instance, attribute, getattr(self, attribute))
        working_excitation = deepcopy(excitation.collection_class)
        for attribute in excitation.attributes:
            if attribute == "name":
                setattr(working_excitation, attribute, excitation.name[0])
            elif attribute == "measurements":
                setattr(
                    working_excitation,
                    attribute,
                    excitation.measurements[0][()] * excitation.measurements_units,
                )
            else:
                setattr(working_excitation, attribute, getattr(excitation, attribute))
        from pymodal import frf_collection
        frf_collection_instance = frf_collection(
            [
                working_instance.to_FRF(
                    excitation=working_excitation,
                    FRF_type=FRF_type,
                    resp_delay=resp_delay,
                )
            ],
            new_path,
        )
        for i, name in enumerate(self.name):
            if i > 0:
                working_instance = deepcopy(self.collection_class)
                for attribute in self.attributes:
                    if attribute == "name":
                        setattr(working_instance, attribute, name)
                    elif attribute == "measurements":
                        setattr(
                            working_instance,
                            attribute,
                            self.measurements[i][()] * self.measurements_units,
                        )
                    else:
                        setattr(working_instance, attribute, getattr(self, attribute))
                working_excitation = deepcopy(excitation.collection_class)
                for attribute in excitation.attributes:
                    if attribute == "name":
                        setattr(working_excitation, attribute, excitation.name[i])
                    elif attribute == "measurements":
                        setattr(
                            working_excitation,
                            attribute,
                            excitation.measurements[i][()]
                            * excitation.measurements_units,
                        )
                    else:
                        setattr(
                            working_excitation,
                            attribute,
                            getattr(excitation, attribute),
                        )
                frf_collection_instance.append(
                    working_instance.to_FRF(
                        excitation=working_excitation,
                        FRF_type=FRF_type,
                        resp_delay=resp_delay,
                    )
                )
        return frf_collection_instance

    def AddGaussianNoise(
        self, min_amplitude=0.001, max_amplitude=0.015, sample: Optional[float] = None
    ):
        if sample is None:
            sample = 1.0
        if type(sample) is float:
            n = int(np.floor(len(self) * sample))
            sample = random.sample(self.name, n)
        working_instance = deepcopy(self.collection_class)
        for attribute in self.attributes:
            if attribute != "name" and attribute != "measurements":
                setattr(working_instance, attribute, getattr(self, attribute))
        augmenter = Compose(
            [
                AddGaussianNoise(
                    min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=1.0
                )
            ]
        )
        for i, name in enumerate(self.name):
            if name in sample:
                array = self.measurements[i][()]
                augmented_samples = np.empty(array.shape)
                for j in range(array.shape[1]):
                    for k in range(array.shape[2]):
                        augmented_samples[:, j, k] = augmenter(
                            samples=array[:, j, k], sample_rate=self.sampling_rate
                        )
                working_instance.name = f"{name}_augmented"
                working_instance.measurements = augmented_samples
                try:
                    self.append(working_instance, self.labels[i])
                except Exception as _:
                    self.append(working_instance)
        return self


if __name__ == "__main__":
    time = np.arange(0, 30 + 0.05, 0.1)
    signal = np.sin(1 * time)
    signal = np.vstack((signal, np.sin(2 * time)))
    signal = np.vstack((signal, np.sin(3 * time)))
    signal = np.vstack((signal, np.sin(4 * time)))
    signal = np.vstack((signal, np.sin(5 * time)))
    signal = signal.reshape((time.shape[0], -1))
    signal_1 = signal * 2
    signal_2 = signal * 4
    signal_3 = signal * 6
    test_object = timeseries(signal, time_end=30)
    test_object_1 = timeseries(signal_1, time_end=30)
    test_object_2 = timeseries(signal_2, time_end=30)
    test_object_3 = timeseries(signal_3, time_end=30)
    test_collection = timeseries_collection([test_object, test_object_1, test_object_2], labels=[0, 1, 2])
    print(test_collection.measurements)
    test_collection.plot()
    plt.show()
    excitation_test = timeseries(np.sin(1 * time), time_end=30, method="excitation")
    excitation_test = timeseries_collection(
        [excitation_test, excitation_test, excitation_test], path="test_exc.h5"
    )
    frf_test = test_collection.to_FRF(excitation_test)
    frf_test.plot(format="mod-phase")
    plt.show()
    test_collection.AddGaussianNoise(min_amplitude=0.4, max_amplitude=0.6).plot()
    plt.show()
    print(test_collection.torch_dataset().dataset.get_data_infos("data"))
    print(test_collection.dataset.get_data("data", -1))
    print(test_collection.dataset.get_data("label", -1))
    import torch
    loader = torch.utils.data.DataLoader(test_collection.dataset, num_workers=2)
    print(next(iter(loader)))
    print(next(iter(loader)))
    test_collection.open()
    print(test_collection.append(test_object_3, 2).measurements)
    print(test_collection.change_time_span(new_max_time=20).measurements)
    print(test_collection.change_sampling_rate(new_sampling_rate=0.2).measurements)
    print(
        test_collection[
            ["Vibrational data", "Vibrational data_1", "Vibrational data_2"]
        ].measurements
    )
    print(test_collection[1:-1].measurements)
    print(test_collection.select_all().measurements)
    test_collection.close()
    excitation_test.close()
    frf_test.close()
