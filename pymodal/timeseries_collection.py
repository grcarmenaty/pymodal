from pymodal import _signal_collection, timeseries
from pathlib import Path
import numpy as np
from copy import deepcopy
import h5py
from warnings import catch_warnings, filterwarnings
import matplotlib.pyplot as plt
from typing import Optional
from audiomentations import Compose, AddGaussianNoise
import random


def change_time_span(var):
    """Worker that applies ``timeseries.change_time_span`` to a single element and
    writes the result back to the HDF5 file.

    Parameters
    ----------
    var : tuple
        ``(collection, i, new_min_time, new_max_time)`` where ``i`` is the
        zero-based index of the signal to process.

    Returns
    -------
    timeseries
        Modified instance with ``measurements`` deleted (data already in HDF5).
    """
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
    """Worker that applies ``timeseries.change_sampling_rate`` to a single element
    and writes the result back to the HDF5 file.

    Parameters
    ----------
    var : tuple
        ``(collection, i, new_sampling_rate)`` where ``i`` is the zero-based index
        of the signal to process.

    Returns
    -------
    timeseries
        Modified instance with ``measurements`` deleted (data already in HDF5).
    """
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
    """HDF5-backed collection of :class:`timeseries` objects.

    Inherits all storage and selection behaviour from :class:`_signal_collection`
    and adds time-domain batch operations, overlay plotting, batch FRF estimation,
    and Gaussian noise augmentation.
    """

    def __init__(
        self,
        exp_list: list[timeseries],
        labels: Optional[list[float]] = None,
        path: Optional[Path] = None,
    ):
        """Create a collection from a list of :class:`timeseries` instances.

        Parameters
        ----------
        exp_list : list of timeseries
            Time-series objects to store. All must share the same non-measurement
            attributes.
        labels : list of float, optional
            Numeric label for each signal (e.g. damage state).
        path : Path or str, optional
            Path for the backing HDF5 file. Auto-generated if None.
        """
        super().__init__(exp_list=exp_list, labels=labels, path=path)
        del exp_list

    def change_time_span(self, new_min_time=None, new_max_time=None):
        """Apply :meth:`timeseries.change_time_span` to every signal in the collection.

        Parameters
        ----------
        new_min_time : float, optional
            New start time. Unchanged if None.
        new_max_time : float, optional
            New end time. Unchanged if None.

        Returns
        -------
        self
        """
        vars = []
        for i in range(len(self)):
            vars.append((self, i, new_min_time, new_max_time))
        self.file.close()
        del self.file
        del self.measurements
        for var in vars:
            working_instance = change_time_span(var)
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
        """Apply :meth:`timeseries.change_sampling_rate` to every signal in the collection.

        Parameters
        ----------
        new_sampling_rate : float
            Target sampling frequency in Hz (fs = 1 / Δt).

        Returns
        -------
        self
        """
        vars = []
        for i in range(len(self)):
            vars.append((self, i, new_sampling_rate))
        self.file.close()
        del self.file
        del self.measurements
        for var in vars:
            working_instance = change_sampling_rate(var)
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
        """Overlay all time-series in the collection on a single plot using a rainbow
        colormap.

        Each signal is rendered using :meth:`timeseries.plot`. Y-axis limits are
        expanded progressively to accommodate every curve.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to draw on. Created automatically if None.
        color : matplotlib colormap, optional
            Colormap used to generate one colour per signal, default ``plt.cm.rainbow``.

        Returns
        -------
        ax : plt.Axes
        img : list of Line2D
        """
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
        excitation: "_signal_collection",
        FRF_type: str = "H1",
        resp_delay: int = 0,
        new_path: Optional[Path] = None,
    ):
        """Compute FRFs for every signal in the collection and return an
        :class:`frf_collection`.

        Parameters
        ----------
        excitation : timeseries_collection
            Matching collection of excitation signals. Must have the same length as
            ``self`` and be indexed in the same order.
        FRF_type : str, optional
            Estimator type passed to :meth:`timeseries.to_FRF`. One of
            ``"H1"``, ``"H2"``, ``"Hv"``, ``"vector"``, ``"ODS"``. Default ``"H1"``.
        resp_delay : int, optional
            Response delay in samples passed to :meth:`timeseries.to_FRF`, default 0.
        new_path : Path or str, optional
            HDF5 path for the resulting :class:`frf_collection`. Auto-generated if None.

        Returns
        -------
        frf_collection
            Collection of computed FRFs, preserving the labels from ``self``.
        """
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

        if self.labels is not None:
            labels = [label[()] for label in self.labels]
        else:
            labels = None
        frf_collection_instance = frf_collection(
            [
                working_instance.to_FRF(
                    excitation=working_excitation,
                    FRF_type=FRF_type,
                    resp_delay=resp_delay,
                )
            ],
            labels=[labels[0]] if labels is not None else None,
            path=new_path,
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
                    ),
                    labels[i] if labels is not None else None,
                )
        return frf_collection_instance

    def AddGaussianNoise(
        self, min_amplitude=0.001, max_amplitude=0.015, sample: Optional[float] = None
    ):
        """Augment the collection by adding Gaussian noise to selected signals.

        A noisy copy of each selected signal is appended to the collection with
        the suffix ``_augmented`` on its name. Uses the ``audiomentations`` library.

        Parameters
        ----------
        min_amplitude : float, optional
            Minimum noise amplitude relative to the signal, default 0.001.
        max_amplitude : float, optional
            Maximum noise amplitude relative to the signal, default 0.015.
        sample : None, float, or list of str, optional
            Which signals to augment.
            - ``None`` or ``1.0``: all signals.
            - float in (0, 1): random proportion of the collection.
            - list of str: specific signal names.

        Returns
        -------
        self
        """
        if sample is None:
            sample = 1.0
        if isinstance(sample, float):
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
                label = self.labels[i] if self.labels is not None else None
                self.append(working_instance, label)
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
    test_collection = timeseries_collection(
        [test_object, test_object_1, test_object_2], labels=[0, 1, 2]
    )
    print(test_collection.measurements)
    test_collection.plot()
    plt.show()
    excitation_test = timeseries(np.sin(1 * time), time_end=30, method="excitation")
    excitation_test = timeseries_collection(
        [excitation_test, excitation_test, excitation_test]
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
    print(test_collection.change_sampling_rate(new_sampling_rate=5.0).measurements)
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
