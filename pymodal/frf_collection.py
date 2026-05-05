from pymodal import _signal_collection, frf
from pathlib import Path
import numpy as np
from copy import deepcopy
import h5py
from warnings import catch_warnings, filterwarnings
import matplotlib.pyplot as plt
from typing import Optional


def change_freq_span(var):
    """Worker that applies ``frf.change_freq_span`` to a single element of a
    collection and writes the result back to the HDF5 file.

    Parameters
    ----------
    var : tuple
        ``(collection, i, new_min_freq, new_max_freq)`` where ``i`` is the
        zero-based index of the signal to process.

    Returns
    -------
    frf
        Modified instance with ``measurements`` deleted (data already written
        to HDF5).
    """
    collection, i, new_min_freq, new_max_freq = var
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
            working_instance = working_instance.change_freq_span(
                new_min_freq, new_max_freq
            )
            f[f"measurements/{collection.name[i]}/data"] = working_instance.measurements
    del working_instance.measurements
    return working_instance


def change_freq_resolution(var):
    """Worker that applies ``frf.change_freq_resolution`` to a single element of a
    collection and writes the result back to the HDF5 file.

    Parameters
    ----------
    var : tuple
        ``(collection, i, freq_resolution)`` where ``i`` is the zero-based index
        of the signal to process.

    Returns
    -------
    frf
        Modified instance with ``measurements`` deleted (data already written
        to HDF5).
    """
    collection, i, freq_resolution = var
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
            working_instance = working_instance.change_freq_resolution(freq_resolution)
            f[f"measurements/{collection.name[i]}/data"] = working_instance.measurements
    del working_instance.measurements
    return working_instance


class frf_collection(_signal_collection):
    """HDF5-backed collection of :class:`frf` objects.

    Inherits all storage and selection behaviour from :class:`_signal_collection`
    and adds frequency-domain batch operations and overlay plotting.
    """

    def __init__(
        self,
        exp_list: list[frf],
        labels: Optional[list[int]] = None,
        path: Optional[Path] = None,
    ):
        """Create a collection from a list of :class:`frf` instances.

        Parameters
        ----------
        exp_list : list of frf
            FRF objects to store. All must share the same non-measurement attributes.
        labels : list of int, optional
            Integer label for each FRF (e.g. damage state index).
        path : Path or str, optional
            Path for the backing HDF5 file. Auto-generated if None.
        """
        super().__init__(exp_list=exp_list, labels=labels, path=path)
        del exp_list

    def change_freq_span(self, new_min_freq=None, new_max_freq=None):
        """Apply :meth:`frf.change_freq_span` to every FRF in the collection.

        Parameters
        ----------
        new_min_freq : float, optional
            New minimum frequency. Unchanged if None.
        new_max_freq : float, optional
            New maximum frequency. Unchanged if None.

        Returns
        -------
        self
        """
        vars = []
        for i in range(len(self)):
            vars.append((self, i, new_min_freq, new_max_freq))
        self.file.close()
        del self.file
        del self.measurements
        for var in vars:
            working_instance = change_freq_span(var)
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("name")
        self.file = h5py.File(self.path, "a")
        self.measurements = list(
            [self.file[f"measurements/{name}/data"] for name in self.name]
        )
        if self.labels is not None:
            self.labels = [self.file[f"measurements/{name}/label"] for name in self.name]
        for attribute in attributes_to_match:
            self.file["measurements"].attrs[attribute] = getattr(
                working_instance, attribute
            )
            setattr(self, attribute, getattr(working_instance, attribute))
        del working_instance
        return self

    def change_freq_resolution(self, freq_resolution):
        """Apply :meth:`frf.change_freq_resolution` to every FRF in the collection.

        Parameters
        ----------
        freq_resolution : float
            Target frequency resolution in the same units as ``self.freq_units``.

        Returns
        -------
        self
        """
        vars = []
        for i in range(len(self)):
            vars.append((self, i, freq_resolution))
        self.file.close()
        del self.file
        del self.measurements
        for var in vars:
            working_instance = change_freq_resolution(var)
        attributes_to_match = deepcopy(self.attributes)
        attributes_to_match.remove("measurements")
        attributes_to_match.remove("name")
        self.file = h5py.File(self.path, "a")
        self.measurements = list(
            [self.file[f"measurements/{name}/data"] for name in self.name]
        )
        if self.labels is not None:
            self.labels = [self.file[f"measurements/{name}/label"] for name in self.name]
        for attribute in attributes_to_match:
            self.file["measurements"].attrs[attribute] = getattr(
                working_instance, attribute
            )
            setattr(self, attribute, getattr(working_instance, attribute))
        del working_instance
        return self

    # ── SHM indicator helpers ──────────────────────────────────────────────────

    def _load_as_matrix(self, i: int) -> np.ndarray:
        """Load the i-th FRF from HDF5 as a (n_dof, n_freq) complex array.

        Matches the input convention of all SHM indicator functions in
        :mod:`pymodal.utils` (rows = DOFs, columns = frequency lines).
        """
        data = self.measurements[i][()]  # (n_freq, n_dof, n_inputs)
        return data.reshape(data.shape[0], -1).T  # (n_dof, n_freq)

    # ── Batch indicator methods (return numpy arrays) ─────────────────────────

    def cfdac(self, reference: frf) -> list:
        """CFDAC matrix for every signal vs a reference FRF.

        Parameters
        ----------
        reference : frf
            Pristine / reference FRF with the same DOF layout.

        Returns
        -------
        list of numpy.ndarray, each (n_dof, n_dof)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return [utils.value_CFDAC(ref_H, self._load_as_matrix(i))
                for i in range(len(self))]

    def fdac(self, reference: frf) -> list:
        """FDAC matrix for every signal vs a reference FRF.

        Returns
        -------
        list of numpy.ndarray, each (n_dof, n_dof)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return [utils.value_FDAC(ref_H, self._load_as_matrix(i))
                for i in range(len(self))]

    def rvac(self, reference: frf) -> np.ndarray:
        """RVAC per DOF for every signal vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_signals, n_dof)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([utils.value_RVAC(ref_H, self._load_as_matrix(i))
                         for i in range(len(self))])

    def gac(self, reference: frf) -> np.ndarray:
        """GAC per DOF for every signal vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_signals, n_dof)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([utils.value_GAC(ref_H, self._load_as_matrix(i))
                         for i in range(len(self))])

    def frfrms(self, reference: frf) -> np.ndarray:
        """FRF RMS metric for every signal vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_signals,)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([utils.FRFRMS(ref_H, self._load_as_matrix(i))
                         for i in range(len(self))])

    def frfsf(self, reference: frf) -> np.ndarray:
        """FRF Scale Factor for every signal vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_signals,)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([utils.FRFSF(ref_H, self._load_as_matrix(i))
                         for i in range(len(self))])

    def frfsm(self, reference: frf, std: float = 6.0) -> np.ndarray:
        """FRF Similarity Measure for every signal vs a reference FRF.

        Parameters
        ----------
        reference : frf
        std : float, optional
            Gaussian width parameter in dB, default 6 dB.

        Returns
        -------
        numpy.ndarray, shape (n_signals,)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([utils.FRFSM(ref_H, self._load_as_matrix(i), std)
                         for i in range(len(self))])

    def ods_diff(self, reference: frf) -> np.ndarray:
        """Summed absolute ODS difference for every signal vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_signals,)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([utils.ODS_diff(ref_H, self._load_as_matrix(i))
                         for i in range(len(self))])

    def r2_imag(self, reference: frf) -> np.ndarray:
        """R² of the imaginary part for every signal vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_signals,)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([utils.r2_imag(ref_H, self._load_as_matrix(i))
                         for i in range(len(self))])

    def sci(self, reference: frf) -> np.ndarray:
        """Structural Change Indicator for every signal vs a reference FRF.

        Uses |CFDAC| (real-valued); |CFDAC(ref, ref)| is computed once as the
        pristine baseline.

        Returns
        -------
        numpy.ndarray, shape (n_signals,)
        """
        import warnings
        from pymodal import utils
        ref_H = reference._as_matrix()
        cfdac_ref = np.abs(utils.value_CFDAC(ref_H, ref_H))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nan_to_num(np.array([
                utils.SCI(cfdac_ref,
                          np.abs(utils.value_CFDAC(ref_H, self._load_as_matrix(i))))
                for i in range(len(self))
            ], dtype=float))

    def drq(self, reference: frf) -> np.ndarray:
        """Damage Residual Quantifier for every signal vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_signals,)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([
            utils.DRQ(utils.value_RVAC(ref_H, self._load_as_matrix(i)))
            for i in range(len(self))
        ])

    def aigac(self, reference: frf) -> np.ndarray:
        """AIGAC for every signal vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_signals,)
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([
            utils.AIGAC(utils.value_GAC(ref_H, self._load_as_matrix(i)))
            for i in range(len(self))
        ])

    def m2l(self, reference: frf) -> np.ndarray:
        """Mode-to-Location indicator for every signal vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_signals, n_dof), real-valued.
        """
        from pymodal import utils
        ref_H = reference._as_matrix()
        return np.array([
            utils.M2L(np.abs(utils.value_CFDAC(ref_H, self._load_as_matrix(i))))
            for i in range(len(self))
        ])

    # ── Collection-returning indicator methods ────────────────────────────────
    #
    # Each *_collection method computes the named indicator for every signal,
    # wraps each result as a minimal frf object, and assembles a new
    # frf_collection — with the same labels — that is HDF5-backed and ready
    # for .torch_dataset().
    #
    # Matrix indicators: stored as (n_rows, n_cols, 1) measurements.
    # Vector indicators: stored as (n_dof, 1, 1) measurements.
    # Scalar indicators: stored as (1, 1, 1) measurements.

    def _indicator_frf(self, data: np.ndarray, name: str) -> frf:
        """Wrap a numpy array as a minimal frf for storage in a new collection.

        The frf constructor requires at least 2 domain points, so scalar (0-d)
        values are duplicated to shape (2, 1, 1).
        """
        if data.ndim == 0:
            val = complex(data)
            arr = np.array([[[val]], [[val]]])  # (2, 1, 1) — satisfies 2-point minimum
        elif data.ndim == 1:
            n = data.shape[0]
            arr = data.astype(complex).reshape(n, 1, 1)
        elif data.ndim == 2:
            r, c = data.shape
            arr = data.astype(complex).reshape(r, c, 1)
        else:
            arr = data.astype(complex)
        return frf(
            measurements=arr,
            freq_resolution=1.0,
            measurements_units="",
            method="SIMO",
            name=name,
        )

    def cfdac_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute CFDAC vs reference for every signal; return an IndicatorCollection.

        Each CFDAC matrix (n_dof × n_dof complex) is stored as a (n_dof, n_dof, 1)
        measurement.  Labels are copied from the original collection.  The reference
        FRF is stored inside the returned collection's HDF5 file.

        Parameters
        ----------
        reference : frf
            Pristine / reference FRF.
        path : Path or str, optional
            HDF5 file for the new collection. Auto-generated if None.

        Returns
        -------
        IndicatorCollection
        """
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(utils.value_CFDAC(ref_H, self._load_as_matrix(i)),
                                self.name[i])
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="cfdac", path=path,
        )

    def cfdac_a_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute CFDAC_A vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(utils.value_CFDAC_A(ref_H, self._load_as_matrix(i)),
                                self.name[i])
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="cfdac_a", path=path,
        )

    def fdac_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute FDAC vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(utils.value_FDAC(ref_H, self._load_as_matrix(i)),
                                self.name[i])
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="fdac", path=path,
        )

    def rvac_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute RVAC per DOF vs reference for every signal; return an IndicatorCollection.

        Each RVAC vector (n_dof,) is stored as (n_dof, 1, 1).
        """
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(utils.value_RVAC(ref_H, self._load_as_matrix(i)),
                                self.name[i])
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="rvac", path=path,
        )

    def rvac_2d_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute RVAC_2d per DOF vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(utils.value_RVAC_2d(ref_H, self._load_as_matrix(i)),
                                self.name[i])
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="rvac_2d", path=path,
        )

    def gac_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute GAC per DOF vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(utils.value_GAC(ref_H, self._load_as_matrix(i)),
                                self.name[i])
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="gac", path=path,
        )

    def m2l_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute M2L vs reference for every signal; return an IndicatorCollection.

        Each M2L vector (n_dof,) is stored as (n_dof, 1, 1).
        """
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(
                utils.M2L(np.abs(utils.value_CFDAC(ref_H, self._load_as_matrix(i)))),
                self.name[i],
            )
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="m2l", path=path,
        )

    def frfrms_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute FRFRMS scalar vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(
                np.array(utils.FRFRMS(ref_H, self._load_as_matrix(i))),
                self.name[i],
            )
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="frfrms", path=path,
        )

    def frfsf_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute FRFSF scalar vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(
                np.array(utils.FRFSF(ref_H, self._load_as_matrix(i))),
                self.name[i],
            )
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="frfsf", path=path,
        )

    def frfsm_collection(
        self, reference: frf, path: Optional[Path] = None,
        std: float = 6.0,
    ) -> "IndicatorCollection":
        """Compute FRFSM scalar vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(
                np.array(utils.FRFSM(ref_H, self._load_as_matrix(i), std=std)),
                self.name[i],
            )
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="frfsm", path=path,
        )

    def ods_diff_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute ODS_diff scalar vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(
                np.array(utils.ODS_diff(ref_H, self._load_as_matrix(i))),
                self.name[i],
            )
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="ods_diff", path=path,
        )

    def r2_imag_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute r2_imag scalar vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(
                np.array(utils.r2_imag(ref_H, self._load_as_matrix(i))),
                self.name[i],
            )
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="r2_imag", path=path,
        )

    def sci_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute SCI scalar vs reference for every signal; return an IndicatorCollection."""
        import warnings
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        cfdac_ref = np.abs(utils.value_CFDAC(ref_H, ref_H))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frfs = [
                self._indicator_frf(
                    np.array(np.nan_to_num(utils.SCI(
                        cfdac_ref,
                        np.abs(utils.value_CFDAC(ref_H, self._load_as_matrix(i)))
                    ))),
                    self.name[i],
                )
                for i in range(len(self))
            ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="sci", path=path,
        )

    def unsigned_sci_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute unsigned_SCI scalar vs reference for every signal; return an IndicatorCollection."""
        import warnings
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        cfdac_ref = np.abs(utils.value_CFDAC(ref_H, ref_H))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            frfs = [
                self._indicator_frf(
                    np.array(np.nan_to_num(utils.unsigned_SCI(
                        cfdac_ref,
                        np.abs(utils.value_CFDAC(ref_H, self._load_as_matrix(i)))
                    ))),
                    self.name[i],
                )
                for i in range(len(self))
            ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="unsigned_sci", path=path,
        )

    def drq_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute DRQ scalar vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(
                np.array(utils.DRQ(utils.value_RVAC(ref_H, self._load_as_matrix(i)))),
                self.name[i],
            )
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="drq", path=path,
        )

    def aigac_collection(
        self, reference: frf, path: Optional[Path] = None
    ) -> "IndicatorCollection":
        """Compute AIGAC scalar vs reference for every signal; return an IndicatorCollection."""
        from pymodal.indicator_collection import IndicatorCollection
        from pymodal import utils
        ref_H = reference._as_matrix()
        frfs = [
            self._indicator_frf(
                np.array(utils.AIGAC(utils.value_GAC(ref_H, self._load_as_matrix(i)))),
                self.name[i],
            )
            for i in range(len(self))
        ]
        labels = [l[()] for l in self.labels] if self.labels is not None else None
        return IndicatorCollection(
            frfs, labels=labels, ref_frfs=[reference],
            ref_indices=[0] * len(self), indicator_type="aigac", path=path,
        )

    # ── Waterfall plot ─────────────────────────────────────────────────────────

    def waterfall(
        self,
        ax=None,
        format: str = "mod",
        dof_idx: int = 0,
        colormap=None,
        alpha: float = 0.7,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        fontname: str = "DejaVu Serif",
        fontsize: float = 12,
        title: Optional[str] = None,
    ):
        """3-D waterfall plot stacking one DOF of each FRF in the collection along the y axis.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.Axes3D, optional
            3-D axes to draw on. Created automatically if None.
        format : str, optional
            ``"mod"`` (default), ``"real"``, or ``"imag"``.
        dof_idx : int, optional
            Which DOF column to display for multi-DOF FRFs, default 0.
        colormap : matplotlib colormap, optional
            Colour map for the signal ribbons. Defaults to ``plt.cm.rainbow``.
        alpha : float, optional
            Ribbon opacity, default 0.7.
        xlabel, ylabel, zlabel : str, optional
            Axis labels; sensible defaults are provided.
        fontname : str, optional
            Font family, default ``"DejaVu Serif"``.
        fontsize : float, optional
            Font size for axis labels, default 12.
        title : str, optional
            Plot title.

        Returns
        -------
        ax : mpl_toolkits.mplot3d.Axes3D
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if format not in ("mod", "real", "imag"):
            raise ValueError(f"Unknown format '{format}'. Use 'mod', 'real', or 'imag'.")

        n = len(self)
        freqs = self.freq_array.magnitude  # (n_freq,)

        if colormap is None:
            colormap = plt.cm.rainbow
        colors = colormap(np.linspace(0, 1, n))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        z_max = 0.0
        for i in range(n):
            data = self.measurements[i][()]  # (n_freq, n_dof, n_inputs)
            n_freq = data.shape[0]
            flat = data.reshape(n_freq, -1)  # (n_freq, n_dof)
            col = min(dof_idx, flat.shape[1] - 1)
            if format == "mod":
                z = np.abs(flat[:, col]).astype(float)
            elif format == "real":
                z = flat[:, col].real.astype(float)
            else:
                z = flat[:, col].imag.astype(float)
            z_max = max(z_max, float(np.max(z)))

            xs = np.concatenate([[freqs[0]], freqs, [freqs[-1]]])
            zs = np.concatenate([[0.0], z, [0.0]])
            verts = [(xs[k], float(i), zs[k]) for k in range(len(xs))]
            poly = Poly3DCollection([verts], alpha=alpha)
            poly.set_facecolor(colors[i])
            poly.set_edgecolor(colors[i])
            ax.add_collection3d(poly)

        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_ylim(-0.5, n - 0.5)
        ax.set_zlim(0.0, z_max * 1.125 if z_max > 0 else 1.0)

        ax.set_xlabel("Frequency" if xlabel is None else xlabel, fontsize=fontsize)
        ax.set_ylabel("Signal index" if ylabel is None else ylabel, fontsize=fontsize)
        ax.set_zlabel("Amplitude" if zlabel is None else zlabel, fontsize=fontsize)
        if title is not None:
            ax.set_title(title, fontsize=fontsize)

        return ax

    def plot(
        self,
        format: str = "mod",
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
        """Overlay all FRFs in the collection on a single plot using a rainbow colormap.

        Each FRF is rendered using :meth:`frf.plot`. Y-axis limits are expanded
        progressively to accommodate every curve. Accepts the same parameters as
        :meth:`frf.plot` except ``color``, which is a matplotlib colormap.

        Parameters
        ----------
        format : str, optional
            Plot format passed to :meth:`frf.plot`, default ``"mod"``.
        ax : plt.Axes or list of plt.Axes, optional
            Axes to draw on. Created automatically if None.
        color : matplotlib colormap, optional
            Colormap used to generate one colour per FRF, default ``plt.cm.rainbow``.

        Returns
        -------
        ax : plt.Axes or list of plt.Axes
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
        if isinstance(ax, list):
            old_bottom_ylim = []
            old_top_ylim = []
            for ax_n in ax:
                ylim = ax_n.get_ylim()
                old_bottom_ylim.append(ylim[0])
                old_top_ylim.append(ylim[1])
        else:
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
                if isinstance(ax, list):
                    for j, ax_n in enumerate(ax):
                        new_bottom_ylim, new_top_ylim = ax_n.get_ylim()
                        if new_bottom_ylim > old_bottom_ylim[j]:
                            ax_n.set_ylim(bottom=old_bottom_ylim[j])
                        else:
                            old_bottom_ylim[j] = new_bottom_ylim
                        if new_top_ylim < old_top_ylim[j]:
                            ax_n.set_ylim(top=old_top_ylim[j])
                        else:
                            old_top_ylim[j] = new_top_ylim
                else:
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
    signal = signal.reshape((time.shape[0], -1))
    signal = np.fft.fft(signal, axis=0)
    test_object = frf(signal, freq_end=5)
    signal_1 = signal_1.reshape((time.shape[0], -1))
    signal_1 = np.fft.fft(signal_1, axis=0)
    test_object_1 = frf(signal_1, freq_end=5)
    signal_2 = signal_2.reshape((time.shape[0], -1))
    signal_2 = np.fft.fft(signal_2, axis=0)
    test_object_2 = frf(signal_2, freq_end=5)
    test_collection = frf_collection(
        [test_object, test_object_1, test_object_2], labels=[0, 1, 2]
    )
    print(test_collection.measurements)
    test_collection.plot()
    plt.show()
    print(test_collection.torch_dataset().dataset.get_data_infos("data"))
    print(test_collection.dataset.get_data("data", -1))
    print(test_collection.dataset.get_data("label", -1))
    import torch

    loader = torch.utils.data.DataLoader(test_collection.dataset, num_workers=2)
    print(next(iter(loader)))
    print(next(iter(loader)))
    print(test_collection.change_freq_span(new_max_freq=10).measurements)
    print(test_collection.change_freq_resolution(freq_resolution=0.2).measurements)
    print(test_collection[["Vibrational data", "Vibrational data_2"]].measurements)
    print(test_collection[1:-1].measurements)
    print(test_collection["Vibrational data"].measurements)
    print(test_collection.select_all().measurements)
    test_collection.close()
