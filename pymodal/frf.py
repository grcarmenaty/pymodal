"""Frequency response function collection.

A :class:`frf` is a 1-D collection (:class:`pymodal.collection_1d._collection_1d`)
whose single domain axis is frequency. The constructor accepts either a single
measurement array (yielding a 1-item collection) or a list of arrays (a
multi-item collection), so the same class plays the role formerly held by
``frf`` and ``frf_collection`` together.

Channel/DOF metadata is preserved: items have shape ``(n_freq, n_outputs, n_inputs)``.
Reference linking is first-class: methods that compare against a baseline FRF
return new typed collections that carry both the reference and the damaged
collection as embedded references (see :mod:`pymodal.indicators_0d`,
:mod:`pymodal.indicators_1d`, :mod:`pymodal.indicators_2d`).
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


class frf(_collection_1d):
    """Collection of frequency response functions.

    Parameters
    ----------
    measurements : numpy.ndarray or list of numpy.ndarray
        Single FRF (3-D ``(n_freq, n_outputs, n_inputs)``) or list of such
        arrays sharing identical shape.
    coordinates, orientations : numpy.ndarray, optional
        Channel-DOF spatial metadata.
    dof : int, optional
    freq_start, freq_end, freq_span, freq_resolution : float, optional
        Provide enough of these to define the frequency axis (or pass
        ``freq_array`` directly).
    freq_array : numpy.ndarray, optional
        Explicit frequency axis (1-D).
    measurements_units : str, optional
        Default ``"millimeter / second ** 2 / newton"`` (accelerance).
    freq_units : str, optional
        Default ``"hertz"``.
    space_units : str, optional
        Default ``"millimeter"``.
    method : str, optional
        ``"SIMO"`` (default), ``"MISO"``, or ``"MIMO"``.
    name, names : str or list of str, optional
        Single name (1-item) or per-item names (multi-item).
    labels : list of float, optional
    references, reference_index_maps : dict, optional
        For typed downstream collections produced by indicator methods.
    path : Path or str, optional
    """

    _default_units = "hertz"

    def __init__(
        self,
        measurements,
        coordinates: Optional[np.ndarray] = None,
        orientations: Optional[np.ndarray] = None,
        dof: Optional[int] = None,
        freq_start: Optional[float] = None,
        freq_end: Optional[float] = None,
        freq_span: Optional[float] = None,
        freq_resolution: Optional[float] = None,
        freq_array: Optional[np.ndarray] = None,
        measurements_units: Optional[str] = "millimeter / second ** 2 / newton",
        freq_units: Optional[str] = "hertz",
        space_units: Optional[str] = "millimeter",
        method: str = "SIMO",
        name: Optional[str] = None,
        names: Optional[list[str]] = None,
        labels: Optional[list[float]] = None,
        references: Optional[dict] = None,
        reference_index_maps: Optional[dict] = None,
        path=None,
    ):
        if method == "excitation":
            raise ValueError("FRFs cannot use method='excitation'.")

        items = self._normalise_items(measurements)
        if names is None and name is not None:
            names = [name] * len(items)

        if freq_array is None and freq_span is not None and freq_resolution is None:
            freq_resolution = freq_span / (items[0].shape[0] - 1)
        if freq_array is None and freq_end is not None and freq_resolution is None:
            start = freq_start if freq_start is not None else 0.0
            freq_resolution = (freq_end - start) / (items[0].shape[0] - 1)

        super().__init__(
            items=items,
            domain_array=freq_array,
            domain_start=freq_start,
            domain_end=freq_end,
            domain_resolution=freq_resolution,
            domain_units=freq_units,
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
        """Coerce a single array or list-of-arrays into a list of ndarrays."""
        if isinstance(measurements, list):
            return [np.asarray(a) for a in measurements]
        return [np.asarray(measurements)]

    # ── frequency aliases ─────────────────────────────────────────────────────

    @property
    def freq_array(self) -> np.ndarray:
        return self.domain_array

    @property
    def freq_units(self) -> str:
        return self.domain_units_str

    @property
    def freq_start(self) -> float:
        return self.domain_start

    @property
    def freq_end(self) -> float:
        return self.domain_end

    @property
    def freq_span(self) -> float:
        return self.domain_span

    @property
    def freq_resolution(self) -> float:
        return self.domain_resolution

    # ── frequency edition aliases ─────────────────────────────────────────────

    def change_freq_span(
        self,
        new_min_freq: Optional[float] = None,
        new_max_freq: Optional[float] = None,
    ) -> "frf":
        """Crop or extend every FRF along the frequency axis."""
        return self.change_domain_span(new_min=new_min_freq, new_max=new_max_freq)

    def change_freq_resolution(self, freq_resolution: float) -> "frf":
        """Resample every FRF to ``freq_resolution`` along the frequency axis."""
        return self.change_domain_resolution(new_resolution=freq_resolution)

    # ── helpers for SHM indicator math ────────────────────────────────────────

    def _as_matrix(self, i: int = 0) -> np.ndarray:
        """Return item *i* as a ``(n_freq, n_dof)`` complex matrix.

        Reshapes to match the convention used by the SHM-indicator pure
        functions in :mod:`pymodal.metrics`.
        """
        data = self.measurements[i][()]  # (n_freq, n_outputs, n_inputs)
        return data.reshape(data.shape[0], -1)  # (n_freq, n_dof)

    # ── plotting ──────────────────────────────────────────────────────────────

    def plot(
        self,
        format: str = "mod",
        ax: Optional[plt.Axes] = None,
        index: int = 0,
        color: str = "blue",
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        log_y: Optional[bool] = None,
    ):
        """Plot one item of the collection in a chosen format.

        Parameters
        ----------
        format : str
            ``"mod"`` (default), ``"phase"``, ``"real"``, ``"imag"``.
            ``"mod"`` defaults to log-y.
        ax : matplotlib.axes.Axes, optional
        index : int
            Which item to plot (defaults to 0). Use overlay loops for collections.
        color, title, xlabel, ylabel : optional plotting overrides.
        log_y : bool, optional
            Override the format-derived default y-scale.
        """
        if format not in ("mod", "phase", "real", "imag"):
            raise ValueError(f"Unknown format '{format}'.")
        data = self.measurements[index][()]  # (n_freq, n_outputs, n_inputs)
        n_freq = data.shape[0]
        flat = data.reshape(n_freq, -1)
        x = self.freq_array

        if format == "mod":
            y = np.abs(flat)
            log_y = True if log_y is None else log_y
        elif format == "phase":
            y = np.angle(flat)
            log_y = False if log_y is None else log_y
        elif format == "real":
            y = flat.real
            log_y = False if log_y is None else log_y
        else:  # imag
            y = flat.imag
            log_y = False if log_y is None else log_y

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x, y, color=color)
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel or f"Frequency ({self.freq_units})")
        ax.set_ylabel(
            ylabel or f"Amplitude ({self.measurements_units})"
        )
        if title is not None:
            ax.set_title(title)
        elif self.name:
            ax.set_title(self.name[index])
        return ax

    def waterfall(
        self,
        ax=None,
        format: str = "mod",
        dof_idx: int = 0,
        colormap=None,
        alpha: float = 0.7,
    ):
        """3-D waterfall of every item along the y axis (one DOF column shown)."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        if format not in ("mod", "real", "imag"):
            raise ValueError(f"Unknown format '{format}'.")
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        if colormap is None:
            colormap = plt.cm.rainbow
        n = len(self)
        colors = colormap(np.linspace(0, 1, n))
        freqs = self.freq_array
        z_max = 0.0
        for i in range(n):
            data = self.measurements[i][()]
            flat = data.reshape(data.shape[0], -1)
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
        ax.set_xlabel(f"Frequency ({self.freq_units})")
        ax.set_ylabel("Item index")
        ax.set_zlabel(f"Amplitude ({self.measurements_units})")
        return ax

    # ── pair-op factories: indicator methods that return typed collections ───

    def cfdac(self, reference: "frf", path=None):
        """CFDAC (2-D matrix) per item against ``reference``. Returns a
        :class:`pymodal.indicators_2d.cfdac_collection`."""
        from pymodal.indicators_2d import cfdac_collection
        return cfdac_collection.from_pair(reference=reference, damaged=self, path=path)

    def cfdac_a(self, reference: "frf", path=None):
        """CFDAC-A (2-D) per item. Returns :class:`cfdac_a_collection`."""
        from pymodal.indicators_2d import cfdac_a_collection
        return cfdac_a_collection.from_pair(reference=reference, damaged=self, path=path)

    def fdac(self, reference: "frf", path=None):
        """FDAC (2-D real) per item. Returns :class:`fdac_collection`."""
        from pymodal.indicators_2d import fdac_collection
        return fdac_collection.from_pair(reference=reference, damaged=self, path=path)

    def rvac(self, reference: "frf", path=None):
        """RVAC (1-D over DOF) per item. Returns :class:`rvac_collection`."""
        from pymodal.indicators_1d import rvac_collection
        return rvac_collection.from_pair(reference=reference, damaged=self, path=path)

    def rvac_2d(self, reference: "frf", path=None):
        """RVAC'' curvature variant. Returns :class:`rvac_2d_collection`."""
        from pymodal.indicators_1d import rvac_2d_collection
        return rvac_2d_collection.from_pair(reference=reference, damaged=self, path=path)

    def gac(self, reference: "frf", path=None):
        """GAC (1-D over DOF) per item. Returns :class:`gac_collection`."""
        from pymodal.indicators_1d import gac_collection
        return gac_collection.from_pair(reference=reference, damaged=self, path=path)

    def m2l(self, reference: "frf", path=None):
        """M2L (1-D over DOF) per item. Returns :class:`m2l_collection`."""
        from pymodal.indicators_1d import m2l_collection
        return m2l_collection.from_pair(reference=reference, damaged=self, path=path)

    def sci(self, reference: "frf", path=None):
        """SCI (0-D scalar per channel) per item. Returns :class:`sci_collection`."""
        from pymodal.indicators_0d import sci_collection
        return sci_collection.from_pair(reference=reference, damaged=self, path=path)

    def unsigned_sci(self, reference: "frf", path=None):
        """Unsigned-SCI (0-D). Returns :class:`unsigned_sci_collection`."""
        from pymodal.indicators_0d import unsigned_sci_collection
        return unsigned_sci_collection.from_pair(reference=reference, damaged=self, path=path)

    def drq(self, reference: "frf", path=None):
        """DRQ (0-D). Returns :class:`drq_collection`."""
        from pymodal.indicators_0d import drq_collection
        return drq_collection.from_pair(reference=reference, damaged=self, path=path)

    def aigac(self, reference: "frf", path=None):
        """AIGAC (0-D). Returns :class:`aigac_collection`."""
        from pymodal.indicators_0d import aigac_collection
        return aigac_collection.from_pair(reference=reference, damaged=self, path=path)

    def frfrms(self, reference: "frf", path=None):
        """FRFRMS (0-D). Returns :class:`frfrms_collection`."""
        from pymodal.indicators_0d import frfrms_collection
        return frfrms_collection.from_pair(reference=reference, damaged=self, path=path)

    def frfsf(self, reference: "frf", path=None):
        """FRFSF (0-D). Returns :class:`frfsf_collection`."""
        from pymodal.indicators_0d import frfsf_collection
        return frfsf_collection.from_pair(reference=reference, damaged=self, path=path)

    def frfsm(self, reference: "frf", std: float = 6.0, path=None):
        """FRFSM (0-D). Returns :class:`frfsm_collection`."""
        from pymodal.indicators_0d import frfsm_collection
        return frfsm_collection.from_pair(
            reference=reference, damaged=self, std=std, path=path
        )

    def ods_diff(self, reference: "frf", path=None):
        """ODS-diff (0-D). Returns :class:`ods_diff_collection`."""
        from pymodal.indicators_0d import ods_diff_collection
        return ods_diff_collection.from_pair(reference=reference, damaged=self, path=path)

    def r2_imag(self, reference: "frf", path=None):
        """R²-imag (0-D). Returns :class:`r2_imag_collection`."""
        from pymodal.indicators_0d import r2_imag_collection
        return r2_imag_collection.from_pair(reference=reference, damaged=self, path=path)
