import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data

# https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5


class HDF5Dataset(data.Dataset):
    """PyTorch Dataset wrapper over a single HDF5 file written by
    :class:`_signal_collection`.

    Expects the HDF5 structure ``measurements/{name}/data`` and optionally
    ``measurements/{name}/label``. Supports lazy loading with an LRU file-level
    cache (``data_cache_size`` files kept in memory simultaneously).
    """

    def __init__(self, file_path, load_data=False, data_cache_size=3, transform=None):
        """Initialise the dataset from a single HDF5 file.

        Parameters
        ----------
        file_path : str or Path
            Path to the HDF5 file produced by :class:`_signal_collection`.
        load_data : bool, optional
            If True all data is loaded into RAM immediately. Leave False for lazy
            loading when the dataset does not fit in memory. Default False.
        data_cache_size : int, optional
            Maximum number of HDF5 files kept in the in-memory cache simultaneously.
            Default 3.
        transform : callable, optional
            PyTorch transform applied to every data sample in ``__getitem__``.
        """
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)

        self._add_data_infos(str(p.resolve()), load_data)

    def __getitem__(self, index):
        """Return the ``(data, label)`` pair at ``index`` as PyTorch tensors.

        Parameters
        ----------
        index : int
            Index into the list of ``"data"`` datasets.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            ``(data, label)`` where data has the shape of the stored measurement
            array and label is a scalar tensor.
        """
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(np.array(y))
        return (x, y)

    def __len__(self):
        """Return the total number of data samples in the dataset."""
        return len(self.get_data_infos("data"))

    def _add_data_infos(self, file_path, load_data):
        """Scan the HDF5 file and populate ``self.data_info`` with metadata for every
        dataset found under ``/measurements/{name}/{type}``.

        Parameters
        ----------
        file_path : str
            Absolute path to the HDF5 file.
        load_data : bool
            If True, immediately load each dataset into the cache.
        """
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for _, group in h5_file.items():
                for gname, group in group.items():
                    for dname, ds in group.items():
                        # if data is not loaded its cache index is -1
                        idx = -1
                        if load_data:
                            # add data to the data cache
                            idx = self._add_to_cache(ds[()], file_path)

                        # type is derived from the name of the dataset; we expect the
                        # dataset name to have a name such as 'data' or 'label' to
                        # identify its type we also store the shape of the data in
                        # case we need it
                        self.data_info.append(
                            {
                                "file_path": file_path,
                                "name": gname,
                                "type": dname,
                                "shape": ds.shape,
                                "cache_idx": idx,
                            }
                        )

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for _, group in h5_file.items():
                for gname, group in group.items():
                    for dname, ds in group.items():
                        # add data to the data cache and retrieve
                        # the cache index
                        idx = self._add_to_cache(ds[()], file_path)

                        # find the beginning index of the hdf5 file we are looking for
                        file_idx = next(
                            i
                            for i, v in enumerate(self.data_info)
                            if v["file_path"] == file_path
                        )

                        # the data info should have the same index since we loaded it
                        # in the same way
                        self.data_info[file_idx + idx]["cache_idx"] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [
                {
                    "file_path": di["file_path"],
                    "name": di["name"],
                    "type": di["type"],
                    "shape": di["shape"],
                    "cache_idx": -1,
                }
                if di["file_path"] == removal_keys[0]
                else di
                for di in self.data_info
            ]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data."""
        data_info_type = [di for di in self.data_info if di["type"] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
        dataset. This will make sure that the data is loaded in case it is
        not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]["file_path"]
        if fp not in self.data_cache:
            self._load_data(fp)

        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]["cache_idx"]
        return self.data_cache[fp][cache_idx]
