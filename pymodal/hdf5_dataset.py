import h5py
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils import data


class HDF5Dataset(data.Dataset):
    """PyTorch Dataset wrapper over one or more HDF5 files written by
    :class:`_signal_collection`.

    ``file_path`` can be either a single ``.h5``/``.hdf5`` file or a directory.
    When a directory is given, every HDF5 file found in it (and, if
    ``recursive=True``, in all subdirectories) is included.  All files must
    share the same HDF5 structure: ``measurements/{name}/data`` and optionally
    ``measurements/{name}/label``.

    Files are opened in read-only mode (``"r"``) and each ``__getitem__`` call
    opens/closes its own file handle, making the dataset safe for
    ``DataLoader(num_workers > 0)``.  Pass ``load_data=True`` to pre-load all
    arrays into RAM and avoid repeated I/O.
    """

    def __init__(
        self,
        file_path,
        recursive=False,
        load_data=False,
        transform=None,
    ):
        """Initialise the dataset from a file or a directory of HDF5 files.

        Parameters
        ----------
        file_path : str or Path
            Path to a single HDF5 file **or** a directory containing HDF5
            files produced by :class:`_signal_collection`.
        recursive : bool, optional
            When ``file_path`` is a directory, also search all subdirectories.
            Ignored when ``file_path`` is a file. Default False.
        load_data : bool, optional
            If True, load all arrays into RAM at construction time. Use when
            the full dataset fits in memory to eliminate per-sample I/O.
            Default False (lazy, file-per-access).
        transform : callable, optional
            Applied to every data array in ``__getitem__`` before returning.
            Receives a numpy ndarray; must return a ``torch.Tensor``.
        """
        super().__init__()
        self.data_info = []
        self.ram_cache = {}   # populated only when load_data=True
        self.transform = transform

        p = Path(file_path)
        if p.is_dir():
            search = p.rglob if recursive else p.glob
            h5_files = sorted(search("*.h5")) + sorted(search("*.hdf5"))
            for f in h5_files:
                self._add_data_infos(str(f.resolve()), load_data)
        else:
            self._add_data_infos(str(p.resolve()), load_data)

    def __getitem__(self, index):
        """Return the ``(data, label)`` pair at ``index`` as PyTorch tensors.

        Parameters
        ----------
        index : int

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
        """
        x = self.get_data("data", index)
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        y = self.get_data("label", index)
        y = torch.from_numpy(np.array(y))
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos("data"))

    def _add_data_infos(self, file_path, load_data):
        """Scan the HDF5 file and register metadata for every dataset."""
        with h5py.File(file_path, "r") as h5_file:
            for _, group in h5_file.items():
                for gname, signal_group in group.items():
                    for dname, ds in signal_group.items():
                        entry = {
                            "file_path": file_path,
                            "name": gname,
                            "type": dname,
                            "shape": ds.shape,
                            "hdf5_path": f"measurements/{gname}/{dname}",
                        }
                        if load_data:
                            key = (file_path, gname, dname)
                            self.ram_cache[key] = ds[()]
                        self.data_info.append(entry)

    def get_data_infos(self, type):
        """Return metadata entries for datasets of the given type."""
        return [di for di in self.data_info if di["type"] == type]

    def get_data(self, type, i):
        """Load and return a single dataset by type and index.

        Uses the RAM cache when available (``load_data=True``); otherwise opens
        the HDF5 file on demand so that multiprocessing workers are safe.
        """
        info = self.get_data_infos(type)[i]
        key = (info["file_path"], info["name"], info["type"])
        if key in self.ram_cache:
            return self.ram_cache[key]
        with h5py.File(info["file_path"], "r") as f:
            return f[info["hdf5_path"]][()]

    @staticmethod
    def pad_collate_fn(batch):
        """Collate variable-length samples by zero-padding to the longest in the batch.

        Pass as ``collate_fn`` to ``DataLoader`` when signals in the collection
        may have different sequence lengths (e.g. after ``change_freq_span`` with
        different spans).  Pads the last (sequence) dimension with zeros.

        Parameters
        ----------
        batch : list of (torch.Tensor, torch.Tensor)
            List of ``(x, y)`` pairs from ``__getitem__``.

        Returns
        -------
        tuple of (torch.Tensor, torch.Tensor)
            ``(xs_padded, ys)`` where ``xs_padded`` has shape
            ``(batch, channels, max_len)``.
        """
        xs, ys = zip(*batch)
        max_len = max(x.shape[-1] for x in xs)
        padded = torch.stack([F.pad(x, (0, max_len - x.shape[-1])) for x in xs])
        return padded, torch.stack(ys)
