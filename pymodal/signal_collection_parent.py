import numpy as np
from pymodal import _signal, HDF5Dataset
import h5py
from pathlib import Path
import pint
import os
import inspect
from copy import deepcopy
from warnings import catch_warnings, filterwarnings
from typing import Optional
import time


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def save_array(array_info):
    """Write a single numpy array to an HDF5 dataset, replacing it if it exists.

    Parameters
    ----------
    array_info : tuple
        ``(array, dataset_name, file_name)`` where ``dataset_name`` is the full
        HDF5 path (e.g. ``"measurements/signal_0/data"``) and ``file_name`` is
        the path to the HDF5 file.
    """
    array, dataset_name, file_name = array_info
    # Downcast complex128 → complex64 to halve storage and GPU transfer cost.
    if np.iscomplexobj(array) and array.dtype == np.complex128:
        array = array.astype(np.complex64)
    with h5py.File(file_name, "a") as hf:
        if dataset_name in hf:
            del hf[dataset_name]
        hf[dataset_name] = array


def add_suffix(strings):
    """Return a copy of ``strings`` where duplicate values are made unique by
    appending ``_<n>`` suffixes.

    Parameters
    ----------
    strings : list of str
        Input list, possibly containing duplicates.

    Returns
    -------
    list of str
        List of the same length with all values unique.
    """
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
    """Return a list of public, non-method attribute names of ``obj``.

    Parameters
    ----------
    obj : object
        Any Python object.

    Returns
    -------
    list of str
        Names of attributes that do not start with ``__`` and are not methods.
    """
    attributes = []
    for name, value in inspect.getmembers(obj):
        if not name.startswith("__") and not inspect.ismethod(value):
            attributes.append(name)
    return attributes


# Check if specified attributes match
def attributes_match(instance1, instance2, attributes_to_match):
    """Return True if all listed attributes are equal between two objects.

    numpy arrays are compared with ``np.array_equal``; pint Quantities by
    comparing their magnitudes; everything else with ``==``.

    Parameters
    ----------
    instance1, instance2 : object
        Objects to compare.
    attributes_to_match : list of str
        Attribute names to check.

    Returns
    -------
    bool
    """
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
    """Unpack a ``(instance1, instance2, attributes_to_match)`` tuple and call
    ``attributes_match``. Used as a multiprocessing worker target."""
    instance1, instance2, attributes_to_match = pair
    return attributes_match(instance1, instance2, attributes_to_match)


def parallel_attributes_match(instances, attributes_to_match):
    """Check that all instances in ``instances`` share the same attribute values
    as the first element, for each attribute in ``attributes_to_match``.

    Currently runs sequentially (multiprocessing pool is commented out).

    Parameters
    ----------
    instances : list of _signal
        All signals intended for the same collection.
    attributes_to_match : list of str
        Attribute names that must be identical across all signals.

    Returns
    -------
    bool
        True only if every instance matches the first one on all listed attributes.
    """
    first_instance = instances[0]
    remaining_instances = instances[1:]
    results = [
        worker((first_instance, instance, attributes_to_match))
        for instance in remaining_instances
    ]
    return bool(np.all(results))


class _signal_collection:
    """HDF5-backed container for a list of signals that share all non-measurement
    attributes (coordinates, units, method, domain parameters, etc.).

    Measurements are written to an HDF5 file under ``measurements/{name}/data``.
    The file is kept open for the lifetime of the collection; call ``close()``
    to release it.  All batch operations (``change_*``, ``plot``) return ``self``
    for method chaining.
    """

    def __init__(
        self,
        exp_list: list[_signal],
        labels: Optional[list[float]] = None,
        path: Optional[Path] = None,
    ):
        """Initialise the collection from a list of compatible signals.

        Parameters
        ----------
        exp_list : list of _signal
            Signals to store. All must have identical non-measurement attributes.
        labels : list of float, optional
            Numeric label for each signal (e.g. damage state). Must have the same
            length as ``exp_list`` if provided.
        path : Path or str, optional
            Path of the HDF5 file to create. A timestamp-based filename is used
            if None. Any existing file at that path is deleted first.

        Raises
        ------
        AssertionError
            If any two signals differ in a non-measurement attribute.
        """
        if path is None:
            path = f"{int(time.time()*1000)}.h5"
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
            (array.magnitude, f"measurements/{self.name[i]}/data", self.path)
            for i, array in enumerate([exp.measurements for exp in exp_list])
        ]
        for array in array_info:
            save_array(array)
        exp_list = exp_list[0]
        self.file = h5py.File(self.path, "a")
        self.measurements = list(
            [self.file[f"measurements/{name}/data"] for name in self.name]
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
        if labels is not None:
            for i, label in enumerate(labels):
                self.file[f"measurements/{self.name[i]}/label"] = label
            self.labels = list(
                [self.file[f"measurements/{name}/label"] for name in self.name]
            )
        else:
            self.labels = labels

    def __len__(self):
        """Return the number of signals currently selected in the collection."""
        return len(self.name)

    def __getitem__(self, key: tuple[slice]):
        """Select a subset of signals or spatial degrees of freedom in-place.

        String or list/set of strings
            Restrict the active selection to the named signals.
        int or slice
            Restrict the active selection to a contiguous range of signals.
        One slice applied spatially
            Slice the output (SIMO/MIMO) or input (MISO/excitation) axis of every
            measurement in the HDF5 file.
        Two slices applied spatially
            Slice both the output and input axes (MIMO).

        Spatial slicing modifies the HDF5 datasets in-place and updates
        ``coordinates``, ``orientations``, and ``dof``.

        Returns
        -------
        self
        """
        if isinstance(key, str):
            key = [key]
        if isinstance(key, (set, list)):
            self.name = list(key)
            self.measurements = list(
                [self.file[f"measurements/{name}/data"] for name in self.name]
            )
            try:
                self.labels = list(
                    [self.file[f"measurements/{name}/label"] for name in self.name]
                )
            except Exception:
                pass
        else:
            if isinstance(key, int):
                key = slice(key, key + 1)
            if isinstance(key, slice):
                key = [key]
            key = list(key)
            for i, index in enumerate(key):
                if isinstance(index, int):
                    key[i] = slice(index, index + 1)
            # If only one key is provided, it is assumed to refer to an output
            # selection, unless the system type is supposed to have only one
            # input, in which case it will be assumed to refer to an input selection.
            # If two keys are provided, the first one is assumed to refer to an
            # output, the second to an input.
            if len(key) == 1:
                if self.method in ["SIMO"]:
                    for i, measurement in enumerate(self.measurements):
                        del self.file[f"measurements/{self.name[i]}/data"]
                        self.file[f"measurements/{self.name[i]}/data"] = measurement[
                            :, key[0], :
                        ]
                    self.coordinates = self.coordinates[key[0], :]
                    self.orientations = self.orientations[key[0], :]
                elif self.method in ["MIMO"]:
                    for i, measurement in enumerate(self.measurements):
                        del self.file[f"measurements/{self.name[i]}/data"]
                        self.file[f"measurements/{self.name[i]}/data"] = measurement[
                            :, key[0], :
                        ]
                    self.coordinates = self.coordinates[key[0], :, :]
                    self.orientations = self.orientations[key[0], :, :]
                elif self.method in ["MISO", "excitation"]:
                    for i, measurement in enumerate(self.measurements):
                        del self.file[f"measurements/{self.name[i]}/data"]
                        self.file[f"measurements/{self.name[i]}/data"] = measurement[
                            :, :, key[0]
                        ]
                    self.coordinates = self.coordinates[:, key[0]]
                    self.orientations = self.orientations[:, key[0]]
                self.measurements = list(
                    [self.file[f"measurements/{name}/data"] for name in self.name]
                )
            elif len(key) == 2:
                for i, measurement in enumerate(self.measurements):
                    del self.file[f"measurements/{self.name[i]}/data"]
                    self.file[f"measurements/{self.name[i]}/data"] = measurement[
                        :, key[0], key[1]
                    ]
                self.coordinates = self.coordinates[:, key[0], key[1]]
                self.orientations = self.orientations[:, key[0], key[1]]
                self.measurements = list(
                    [self.file[f"measurements/{name}/data"] for name in self.name]
                )
            else:
                raise ValueError("Too many keys provided.")
            self.dof = max(self.measurements[0].shape[1], self.measurements[0].shape[2])
            self.file["measurements"].attrs["dof"] = self.dof
            self.file["measurements"].attrs["coordinates"] = self.coordinates
            self.file["measurements"].attrs["orientations"] = self.orientations
        return self

    def select_all(self):
        """Reset the active selection to include every signal stored in the HDF5 file.

        Returns
        -------
        self
        """
        self = self[
            list(element[0] for element in list(list(self.file.items())[0][1].items()))
        ]
        return self

    def close(self, keep: bool = False):
        """Close the HDF5 file handle and optionally delete the file.

        Parameters
        ----------
        keep : bool, optional
            If False (default) the HDF5 file is deleted after closing.
            Pass True to retain the file on disk (e.g. before calling
            ``torch_dataset()``).
        """
        self.file.close()
        if not keep:
            self.path.unlink()

    def append(self, signal: _signal, label=None):
        """Add a signal to the collection.

        Parameters
        ----------
        signal : _signal
            Signal to add. Its non-measurement attributes must match those of the
            existing collection members.
        label : float, optional
            Numeric label for this signal.

        Returns
        -------
        self

        Raises
        ------
        AssertionError
            If ``signal`` has incompatible attributes.
        """
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("name")
        assert attributes_match(self, signal, attributes_to_match)
        self.name.append(signal.name)
        self.name = add_suffix(self.name)
        self.file[f"measurements/{self.name[-1]}/data"] = signal.measurements
        self.measurements.append(self.file[f"measurements/{self.name[-1]}/data"])
        if label is not None:
            self.file[f"measurements/{self.name[-1]}/label"] = label
            if self.labels is None:
                self.labels = []
            self.labels.append(self.file[f"measurements/{self.name[-1]}/label"])
        return self

    def torch_dataset(self):
        """Convert the collection to a PyTorch ``HDF5Dataset`` and store it in
        ``self.dataset``.

        The HDF5 file is closed with ``keep=True`` before wrapping, so the file
        remains on disk. Call ``open()`` to reacquire the file handle afterwards.

        Returns
        -------
        self
        """
        self.close(keep=True)
        self.measurements = None
        if self.labels is not None:
            self.labels = None
        self.dataset = HDF5Dataset(self.path)
        return self

    def split(self, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42):
        """Create a stratified train / val / test split and store the indices.

        Indices for each subset are stored as ``self.train_indices``,
        ``self.val_indices``, and ``self.test_indices``.  Stratification
        ensures each class is represented proportionally in every subset.

        Parameters
        ----------
        train_frac : float, optional
            Fraction for training, default 0.70.
        val_frac : float, optional
            Fraction for validation, default 0.15.
        test_frac : float, optional
            Fraction for testing, default 0.15.
        seed : int, optional
            Random seed for reproducibility, default 42.

        Returns
        -------
        tuple of (list, list, list)
            ``(train_indices, val_indices, test_indices)``
        """
        assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, \
            "train_frac + val_frac + test_frac must equal 1.0"
        rng = np.random.default_rng(seed)
        labels_list = self.labels if self.labels is not None else []
        label_groups: dict = {}
        for i, lbl in enumerate(labels_list):
            key = int(lbl[()] if hasattr(lbl, "__getitem__") else lbl)
            label_groups.setdefault(key, []).append(i)
        train_idx, val_idx, test_idx = [], [], []
        for key in sorted(label_groups):
            arr = np.array(label_groups[key])
            rng.shuffle(arr)
            n = len(arr)
            n_train = int(round(n * train_frac))
            n_val = int(round(n * val_frac))
            train_idx.extend(arr[:n_train].tolist())
            val_idx.extend(arr[n_train:n_train + n_val].tolist())
            test_idx.extend(arr[n_train + n_val:].tolist())
        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx
        return train_idx, val_idx, test_idx

    def open(self):
        """Reopen the HDF5 file and refresh the ``measurements`` handle list.

        Use this after ``torch_dataset()`` or any external operation that closed
        the file.

        Returns
        -------
        self
        """
        self.file = h5py.File(self.path, "a")
        self.measurements = list(
            [self.file[f"measurements/{name}/data"] for name in self.name]
        )
        return self


if __name__ == "__main__":

    t = np.arange(0, 30 + 0.05, 0.1)
    signal = np.sin(1 * t)
    signal = np.vstack((signal, np.sin(2 * t)))
    signal = np.vstack((signal, np.sin(3 * t)))
    signal = np.vstack((signal, np.sin(4 * t)))
    signal = np.vstack((signal, np.sin(5 * t)))
    signal = signal.reshape((t.shape[0], -1))
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
