"""Time-series collection.

A :class:`timeseries` is a 1-D collection (:class:`pymodal.collection_1d._collection_1d`)
whose single domain axis is time. The constructor accepts either a single
measurement array (yielding a 1-item collection) or a list of arrays (a
multi-item collection), so the same class plays the role formerly held by
``timeseries`` and ``timeseries_collection`` together.

Items are shaped ``(n_samples, n_outputs, n_inputs)`` and channel/DOF
metadata is preserved at every level.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional, Union
from warnings import catch_warnings, filterwarnings

import matplotlib.pyplot as plt
import numpy as np
from pint import UnitRegistry

from pymodal.collection_1d import _collection_1d


ureg = UnitRegistry()


class timeseries(_collection_1d):
    """Collection of time-domain signals.

    Parameters
    ----------
    measurements : numpy.ndarray or list of numpy.ndarray
        Single signal (3-D ``(n_samples, n_outputs, n_inputs)``) or list of
        such arrays sharing identical shape.
    coordinates, orientations : numpy.ndarray, optional
    dof : int, optional
    time_start, time_end, time_span, time_step : float, optional
        Provide enough of these to define the time axis (or pass
        ``time_array`` directly).
    time_array : numpy.ndarray, optional
        Explicit time axis (1-D).
    measurements_units : str, optional
        Default depends on ``method``: ``"newton"`` for ``"excitation"``,
        ``"meter / second ** 2"`` otherwise.
    time_units : str, optional
        Default ``"second"``.
    space_units : str, optional
        Default ``"millimeter"``.
    method : str, optional
        ``"SIMO"`` (default), ``"MISO"``, ``"MIMO"``, ``"excitation"``.
    name, names : str or list of str, optional
    labels : list of float, optional
    references, reference_index_maps : dict, optional
    path : Path or str, optional
    """

    _default_units = "second"

    def __init__(
        self,
        measurements,
        coordinates: Optional[np.ndarray] = None,
        orientations: Optional[np.ndarray] = None,
        dof: Optional[int] = None,
        time_start: Optional[float] = None,
        time_end: Optional[float] = None,
        time_span: Optional[float] = None,
        time_step: Optional[float] = None,
        time_array: Optional[np.ndarray] = None,
        measurements_units: Optional[str] = None,
        time_units: Optional[str] = "second",
        space_units: Optional[str] = "millimeter",
        method: str = "SIMO",
        name: Optional[str] = None,
        names: Optional[list[str]] = None,
        labels: Optional[list[float]] = None,
        references: Optional[dict] = None,
        reference_index_maps: Optional[dict] = None,
        path=None,
    ):
        items = self._normalise_items(measurements)
        if names is None and name is not None:
            names = [name] * len(items)

        if measurements_units is None:
            measurements_units = (
                "newton" if method == "excitation" else "meter / second ** 2"
            )

        if time_array is None and time_span is not None and time_step is None:
            time_step = time_span / (items[0].shape[0] - 1)
        if time_array is None and time_end is not None and time_step is None:
            start = time_start if time_start is not None else 0.0
            time_step = (time_end - start) / (items[0].shape[0] - 1)

        super().__init__(
            items=items,
            domain_array=time_array,
            domain_start=time_start,
            domain_end=time_end,
            domain_resolution=time_step,
            domain_units=time_units,
            coordinates=coordinates,
            orientations=orientations,
            dof=dof,
            method=method,
            space_units=space_units,
            measurements_units=measurements_units,
            names=names,
            labels=labels,
            references=references,
            reference_index_maps=reference_index_maps,
            path=path,
        )

    @staticmethod
    def _normalise_items(measurements) -> list[np.ndarray]:
        if isinstance(measurements, list):
            return [np.asarray(a) for a in measurements]
        return [np.asarray(measurements)]

    # ── time aliases ──────────────────────────────────────────────────────────

    @property
    def time_array(self) -> np.ndarray:
        return self.domain_array

    @property
    def time_units(self) -> str:
        return self.domain_units_str

    @property
    def time_start(self) -> float:
        return self.domain_start

    @property
    def time_end(self) -> float:
        return self.domain_end

    @property
    def time_span(self) -> float:
        return self.domain_span

    @property
    def time_step(self) -> float:
        return self.domain_resolution

    @property
    def sampling_rate(self) -> float:
        """Sampling frequency in Hz (1 / time_step)."""
        return 1.0 / self.domain_resolution if self.domain_resolution else 0.0

    # ── time edition aliases ─────────────────────────────────────────────────

    def change_time_span(
        self,
        new_min_time: Optional[float] = None,
        new_max_time: Optional[float] = None,
    ) -> "timeseries":
        """Crop or extend every signal along the time axis."""
        return self.change_domain_span(new_min=new_min_time, new_max=new_max_time)

    def change_sampling_rate(self, new_sampling_rate: float) -> "timeseries":
        """Resample every signal to ``new_sampling_rate`` (Hz)."""
        return self.change_domain_resolution(new_resolution=1.0 / new_sampling_rate)

    # ── plotting ──────────────────────────────────────────────────────────────

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        index: int = 0,
        color: str = "blue",
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
    ):
        """Plot one item of the collection in the time domain."""
        data = self.measurements[index][()]
        flat = data.reshape(data.shape[0], -1)
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.time_array, flat, color=color)
        ax.set_xlabel(xlabel or f"Time ({self.time_units})")
        ax.set_ylabel(ylabel or f"Amplitude ({self.measurements_units})")
        if title is not None:
            ax.set_title(title)
        elif self.name:
            ax.set_title(self.name[index])
        return ax

    # ── FRF computation ──────────────────────────────────────────────────────

    def to_FRF(
        self,
        excitation: "timeseries",
        FRF_type: str = "H1",
        resp_delay: int = 0,
        path=None,
    ):
        """Compute FRFs for every item and return a :class:`pymodal.frf`.

        The resulting collection registers two references:

        - ``input``  → ``excitation``
        - ``output`` → ``self``

        Parameters
        ----------
        excitation : timeseries
            Excitation collection (``method="excitation"``); same length as
            ``self`` and indexed in matching order.
        FRF_type : str
            One of ``"H1"``, ``"H2"``, ``"Hv"``, ``"vector"``, ``"ODS"``.
        resp_delay : int
            Response delay (samples) passed to the FRF estimator.
        path : Path or str, optional
        """
        from pyFRF import FRF as _PyFRF
        from pymodal.frf import frf as _frf_cls

        if excitation.method != "excitation":
            raise ValueError("excitation.method must be 'excitation'.")
        if len(self) != len(excitation):
            raise ValueError(
                f"timeseries collection length {len(self)} != excitation length "
                f"{len(excitation)}."
            )

        resp_dim = ureg.parse_expression(self.measurements_units).dimensionality
        if resp_dim == ureg.parse_expression("meter").dimensionality:
            resp_type, form = "d", "receptance"
        elif resp_dim == ureg.parse_expression("meter/second").dimensionality:
            resp_type, form = "v", "mobility"
        elif resp_dim == ureg.parse_expression("meter/second**2").dimensionality:
            resp_type, form = "a", "accelerance"
        else:
            resp_type, form = "e", "receptance"

        exc_dim = ureg.parse_expression(excitation.measurements_units).dimensionality
        if exc_dim == ureg.parse_expression("newton").dimensionality:
            exc_type = "f"
        elif exc_dim == ureg.parse_expression("meter").dimensionality:
            exc_type = "d"
        elif exc_dim == ureg.parse_expression("meter/second").dimensionality:
            exc_type = "v"
        elif exc_dim == ureg.parse_expression("meter/second**2").dimensionality:
            exc_type = "a"
        else:
            exc_type = "e"

        fs = int(round(self.sampling_rate))

        frfs: list[np.ndarray] = []
        for i in range(len(self)):
            resp_arr = self.measurements[i][()]   # (n_t, n_out, n_in)
            exc_arr = excitation.measurements[i][()]   # (n_t, 1, n_dof_exc)
            frf_amp_list = []
            if self.method == "SIMO":
                exc = exc_arr[:, 0, 0]
                for j in range(self.n_outputs):
                    resp = resp_arr[:, j, 0]
                    frf_amp_list.append(
                        _PyFRF(
                            sampling_freq=fs, exc=exc, resp=resp,
                            exc_type=exc_type, resp_type=resp_type,
                            window="none", resp_delay=resp_delay, noverlap=0,
                        ).get_FRF(type=FRF_type, form=form)
                    )
                frf_amp = np.array(frf_amp_list).reshape((-1, self.n_outputs, 1))
            elif self.method == "MISO":
                for j in range(self.n_inputs):
                    exc = exc_arr[:, 0, j]
                    resp = resp_arr[:, 0, j]
                    frf_amp_list.append(
                        _PyFRF(
                            sampling_freq=fs, exc=exc, resp=resp,
                            exc_type=exc_type, resp_type=resp_type, window="none",
                        ).get_FRF(type=FRF_type, form=form)
                    )
                frf_amp = np.array(frf_amp_list).reshape((-1, 1, self.n_inputs))
            elif self.method == "MIMO":
                outer = []
                for j in range(self.n_outputs):
                    inner = []
                    for k in range(self.n_inputs):
                        exc = exc_arr[:, 0, k]
                        resp = resp_arr[:, j, k]
                        inner.append(
                            _PyFRF(
                                sampling_freq=fs, exc=exc, resp=resp,
                                exc_type=exc_type, resp_type=resp_type, window="none",
                            ).get_FRF(type=FRF_type, form=form)
                        )
                    outer.append(np.array(inner))
                frf_amp = np.array(outer).reshape((-1, self.n_outputs, self.n_inputs))
            else:
                raise ValueError(f"Unsupported method '{self.method}' for to_FRF.")
            frfs.append(frf_amp)

        labels = [lbl[()] for lbl in self.labels] if self.labels is not None else None

        return _frf_cls(
            measurements=frfs,
            coordinates=self.coordinates,
            orientations=self.orientations,
            dof=self.dof,
            freq_resolution=1.0 / self.time_span if self.time_span else 1.0,
            measurements_units=f"({self.measurements_units}) / ({excitation.measurements_units})",
            space_units=self.space_units,
            method=self.method,
            names=list(self.name),
            labels=labels,
            references={"input": excitation, "output": self},
            reference_index_maps={
                "input": list(range(len(self))),
                "output": list(range(len(self))),
            },
            path=path,
        )

    # ── augmentation ──────────────────────────────────────────────────────────

    def AddGaussianNoise(
        self,
        min_amplitude: float = 0.001,
        max_amplitude: float = 0.015,
        sample: Optional[Union[float, list[str]]] = None,
    ) -> "timeseries":
        """Append Gaussian-noise-augmented copies of selected items.

        Each augmented copy is appended with the suffix ``_augmented``.
        """
        import random
        from audiomentations import Compose, AddGaussianNoise as _AGN

        if sample is None:
            sample = 1.0
        if isinstance(sample, float):
            n = int(np.floor(len(self) * sample))
            sample = random.sample(self.name, n)

        augmenter = Compose([
            _AGN(min_amplitude=min_amplitude, max_amplitude=max_amplitude, p=1.0),
        ])
        sr = self.sampling_rate
        for i, name in enumerate(list(self.name)):
            if name in sample:
                arr = self.measurements[i][()]
                aug = np.empty_like(arr, dtype=float)
                for j in range(arr.shape[1]):
                    for k in range(arr.shape[2]):
                        aug[:, j, k] = augmenter(
                            samples=arr[:, j, k].astype(np.float32), sample_rate=sr
                        )
                label = (
                    self.labels[i][()] if self.labels is not None else None
                )
                self.append(aug, name=f"{name}_augmented", label=label)
        return self
