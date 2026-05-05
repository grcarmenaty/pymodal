import numpy as np
from typing import Optional
import numpy.typing as npt
from pymodal import _signal
from matplotlib import pyplot as plt
from pint import UnitRegistry
from copy import deepcopy


ureg = UnitRegistry()


class frf(_signal):
    def __init__(
        self,
        measurements: npt.NDArray[np.complex64],
        coordinates: npt.NDArray[np.float64] = None,
        orientations: npt.NDArray[np.float64] = None,
        dof: Optional[float] = None,
        freq_start: Optional[float] = 0,
        freq_end: Optional[float] = None,
        freq_span: Optional[float] = None,
        freq_resolution: Optional[float] = None,
        measurements_units: Optional[str] = "millimeter / second ** 2 / newton",
        freq_units: Optional[str] = "hertz",
        space_units: Optional[str] = "millimeter",
        method: str = "SIMO",
        name: Optional[str] = None,
    ):
        """Class designed to store all vibrational frequency response function (FRF)
        information measured from a three-dimensional body, along with the spatial
        information related to each of the aforementioned FRFs.

        Parameters
        ----------
        measurements : numpy array of complexes
            A numpy array of up to three dimensions where the first one contains the
            measurements as they change along the frequential domain, and the rest are
            related to the system's degrees of freedom and the obtention method.
        coordinates : numpy array of floats, optional
            A two-dimensional array containing the spatial coordinates of the degrees of
            freedom of the measurements contained within the instance of this class,
            repeating as needed if measurements were taken for more than one orientation
            on the same spatial coordinates, by default None.
        orientations : numpy array of floats, optional
            A two dimensional array containing a unit vector representing the direction
            in which the measurement taken at a given coordinate was recorded, by
            default None.
        dof : float, optional
            How many degrees of freedom have been measured and are stored within the
            instance of this class, by default None.
        freq_start : float, optional
            Starting frequency value, by default 0.
        freq_end : float, optional
            Maximum frequency value, by default None.
        freq_span : float, optional
            Difference between the maximum and minimum values of frequency, by
            default None.
        freq_resolution : float, optional
            Frequential resolution, by default None.
        measurements_units : Optional[str], optional
            Units used for the measurements stored within the instance of this class,
            they are assumed to be Newtons, millimeters and seconds, by default
            "millimeter / second ** 2 / newton".
        space_units : Optional[str], optional
            Units used for the spatial coordinates of the degrees of freedom, by
            default "millimeter".
        method : str, optional
            Whether the method used to get the measurements is Multiple Input Single
            Output (MISO), Single Input Multiple Output (SIMO) or Multiple Input
            Multiple Output (MIMO), by default "SIMO".
        name : Optional[str], optional
            An identifying name for the measurements stored in this instance, by
            default None.
        """
        assert method in ["MISO", "SIMO", "MIMO"]
        super(frf, self).__init__(
            measurements=measurements,
            coordinates=coordinates,
            orientations=orientations,
            dof=dof,
            domain_start=freq_start,
            domain_end=freq_end,
            domain_span=freq_span,
            domain_resolution=freq_resolution,
            measurements_units=measurements_units,
            domain_units=freq_units,
            space_units=space_units,
            method=method,
            name=name,
        )
        self.freq_start = self.domain_start
        self.freq_end = self.domain_end
        self.freq_span = self.domain_span
        self.freq_resolution = self.domain_resolution
        self.freq_array = self.domain_array
        self.freq_units = self.domain_units

    def change_freq_span(
        self, new_min_freq: Optional[float] = None, new_max_freq: Optional[float] = None
    ):
        """Either extend or cut the measured data according to new maximum and minimum
        frequency values.

        Parameters
        ----------
        new_min_freq : float, optional
            The new desired minimum frequency. If negative, frequency will be added
            before the initial frequency with all measurements set to 0, and the new
            frequency origin will be set as the new 0, by default None.
        new_max_freq : float, optional
            The new desired maximum frequency. If greater than the previous max
            frequency, frequency will be added after the previous max frequency with all
            measurements set to 0, by default None.

        Returns
        -------
        frf class object
            A hard copy of the class instance with the modifications pertinent to the
            method applied: the new frequency array without the values that fall outside
            the given range, and extended as necessary to comply with the given range,
            with the corresponding measurements values.
        """
        return super().change_domain_span(
            new_min_domain=new_min_freq, new_max_domain=new_max_freq
        )

    def change_freq_resolution(self, new_resolution):
        """Change the frequential resolution and interpolate as needed to return an
        object with values coherent with the desired frequential resolution.

        Parameters
        ----------
        new_resolution : float
            The desired frequential resolution.

        Returns
        -------
        timeseries class object
            A hard copy of the class instance with the modifications pertinent to the
            method applied: all values coherent with the desired new frequential
            resolution.
        """
        return super().change_domain_resolution(new_resolution=new_resolution)

    # ── SHM indicator helpers ──────────────────────────────────────────────────

    def _as_matrix(self) -> np.ndarray:
        """Return measurements as a (n_dof, n_freq) complex numpy array.

        Collapses the output and input dimensions into a single DOF axis so that
        the array matches the convention expected by all SHM indicator functions
        in :mod:`pymodal.utils` (rows = DOFs, columns = frequency lines).
        """
        mag = self.measurements.magnitude  # (n_freq, n_dof, n_inputs)
        return mag.reshape(mag.shape[0], -1).T  # (n_dof, n_freq)

    def cfdac(self, reference: "frf") -> np.ndarray:
        """Complex Frequency Domain Assurance Criterion matrix vs a reference FRF.

        Parameters
        ----------
        reference : frf
            Pristine / reference FRF with the same DOF layout.

        Returns
        -------
        numpy.ndarray, shape (n_dof, n_dof)
            Complex CFDAC matrix.
        """
        from pymodal import utils
        return utils.value_CFDAC(reference._as_matrix(), self._as_matrix())

    def cfdac_a(self, reference: "frf") -> np.ndarray:
        """CFDAC alternative formulation matrix vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_dof, n_dof)
        """
        from pymodal import utils
        return utils.value_CFDAC_A(reference._as_matrix(), self._as_matrix())

    def fdac(self, reference: "frf") -> np.ndarray:
        """Frequency Domain Assurance Criterion matrix vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_dof, n_dof), real-valued.
        """
        from pymodal import utils
        return utils.value_FDAC(reference._as_matrix(), self._as_matrix())

    def rvac(self, reference: "frf") -> np.ndarray:
        """Response Vector Assurance Criterion per DOF vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_dof,), real-valued in [0, 1].
        """
        from pymodal import utils
        return utils.value_RVAC(reference._as_matrix(), self._as_matrix())

    def rvac_2d(self, reference: "frf") -> np.ndarray:
        """Curvature-based RVAC (second finite difference) per DOF vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_dof,), real-valued.
        """
        from pymodal import utils
        return utils.value_RVAC_2d(reference._as_matrix(), self._as_matrix())

    def gac(self, reference: "frf") -> np.ndarray:
        """Global Assurance Criterion per DOF vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_dof,), real-valued.
        """
        from pymodal import utils
        return utils.value_GAC(reference._as_matrix(), self._as_matrix())

    def frfrms(self, reference: "frf") -> float:
        """FRF RMS metric (log-scale) vs a reference FRF.

        Returns
        -------
        float
        """
        from pymodal import utils
        return utils.FRFRMS(reference._as_matrix(), self._as_matrix())

    def frfsf(self, reference: "frf") -> float:
        """FRF Scale Factor vs a reference FRF.

        Returns
        -------
        float
        """
        from pymodal import utils
        return utils.FRFSF(reference._as_matrix(), self._as_matrix())

    def frfsm(self, reference: "frf", std: float = 6.0) -> float:
        """FRF Similarity Measure vs a reference FRF.

        Parameters
        ----------
        reference : frf
        std : float, optional
            Gaussian width parameter in dB (default 6 dB).

        Returns
        -------
        float
        """
        from pymodal import utils
        return utils.FRFSM(reference._as_matrix(), self._as_matrix(), std)

    def ods_diff(self, reference: "frf") -> float:
        """Summed absolute Operating Deflection Shape difference vs a reference FRF.

        Returns
        -------
        float
        """
        from pymodal import utils
        return utils.ODS_diff(reference._as_matrix(), self._as_matrix())

    def r2_imag(self, reference: "frf") -> float:
        """Coefficient of determination (R²) of the imaginary part vs a reference FRF.

        Returns
        -------
        float
        """
        from pymodal import utils
        return utils.r2_imag(reference._as_matrix(), self._as_matrix())

    def sci(self, reference: "frf") -> float:
        """Structural Change Indicator vs a reference FRF.

        Uses |CFDAC| (real-valued) for the Pearson correlation in SCI.
        CFDAC(ref, ref) is the pristine baseline; CFDAC(ref, self) is the altered state.

        Returns
        -------
        float — 0 when identical to reference, grows with structural change.
        """
        import warnings
        from pymodal import utils
        ref_H = reference._as_matrix()
        cfdac_ref = np.abs(utils.value_CFDAC(ref_H, ref_H))
        cfdac_alt = np.abs(utils.value_CFDAC(ref_H, self._as_matrix()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = utils.SCI(cfdac_ref, cfdac_alt)
        return float(np.nan_to_num(result))

    def unsigned_sci(self, reference: "frf") -> float:
        """Unsigned Structural Change Indicator vs a reference FRF.

        Returns
        -------
        float
        """
        import warnings
        from pymodal import utils
        ref_H = reference._as_matrix()
        cfdac_ref = np.abs(utils.value_CFDAC(ref_H, ref_H))
        cfdac_alt = np.abs(utils.value_CFDAC(ref_H, self._as_matrix()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = utils.unsigned_SCI(cfdac_ref, cfdac_alt)
        return float(np.nan_to_num(result))

    def drq(self, reference: "frf") -> float:
        """Damage Residual Quantifier vs a reference FRF.

        Returns
        -------
        float
        """
        from pymodal import utils
        return float(utils.DRQ(utils.value_RVAC(reference._as_matrix(), self._as_matrix())))

    def aigac(self, reference: "frf") -> float:
        """Average Index from the Global Assurance Criterion vs a reference FRF.

        Returns
        -------
        float
        """
        from pymodal import utils
        return float(utils.AIGAC(utils.value_GAC(reference._as_matrix(), self._as_matrix())))

    def m2l(self, reference: "frf") -> np.ndarray:
        """Mode-to-Location indicator vector vs a reference FRF.

        Returns
        -------
        numpy.ndarray, shape (n_dof,), real-valued.
        """
        from pymodal import utils
        cfdac_mat = np.abs(utils.value_CFDAC(reference._as_matrix(), self._as_matrix()))
        return utils.M2L(cfdac_mat)

    def waterfall(
        self,
        ax=None,
        format: str = "mod",
        colormap=None,
        alpha: float = 0.7,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        fontname: str = "DejaVu Serif",
        fontsize: float = 12,
        title: Optional[str] = None,
    ):
        """3-D waterfall plot stacking each DOF of this FRF along the y axis.

        Parameters
        ----------
        ax : mpl_toolkits.mplot3d.Axes3D, optional
            3-D axes to draw on. Created automatically if None.
        format : str, optional
            ``"mod"`` (default), ``"real"``, or ``"imag"``.
        colormap : matplotlib colormap, optional
            Colour map for the DOF ribbons. Defaults to ``plt.cm.rainbow``.
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

        mag = self.measurements.magnitude  # (n_freq, n_dof, n_inputs)
        n_freq = mag.shape[0]
        data = mag.reshape(n_freq, -1)  # (n_freq, n_dof)
        freqs = self.freq_array.magnitude

        if format == "mod":
            z_data = np.abs(data)
        elif format == "real":
            z_data = data.real
        else:
            z_data = data.imag

        n_dof = data.shape[1]
        if colormap is None:
            colormap = plt.cm.rainbow
        colors = colormap(np.linspace(0, 1, n_dof))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")

        for i in range(n_dof):
            z = z_data[:, i]
            xs = np.concatenate([[freqs[0]], freqs, [freqs[-1]]])
            zs = np.concatenate([[0.0], z.astype(float), [0.0]])
            verts = [(xs[k], float(i), zs[k]) for k in range(len(xs))]
            poly = Poly3DCollection([verts], alpha=alpha)
            poly.set_facecolor(colors[i])
            poly.set_edgecolor(colors[i])
            ax.add_collection3d(poly)

        ax.set_xlim(freqs[0], freqs[-1])
        ax.set_ylim(-0.5, n_dof - 0.5)
        z_max = float(np.max(z_data))
        ax.set_zlim(0.0, z_max * 1.125 if z_max > 0 else 1.0)

        ax.set_xlabel(
            f"Frequency ({ureg.hertz:~P})" if xlabel is None else xlabel,
            fontsize=fontsize,
        )
        ax.set_ylabel("DOF index" if ylabel is None else ylabel, fontsize=fontsize)
        ax.set_zlabel(
            f"Amplitude ({self.measurements_units.u:~P})" if zlabel is None else zlabel,
            fontsize=fontsize,
        )
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
        color: str = "blue",
        linestyle: str = "-",
        ylabel: str = None,
        xlabel: str = None,
        decimals_y: int = 2,
        decimals_x: int = 2,
        bottom_ylim: float = None,
        top_ylim: float = None,
        grid: bool = True,
    ):
        """Plot the FRF in one of several formats.

        Parameters
        ----------
        format : str, optional
            One of ``"mod"``, ``"phase"``, ``"mod-phase"``, ``"real"``,
            ``"imag"``, ``"real-imag"``. Default is ``"mod"``.
            ``"mod"`` uses a logarithmic y scale. ``"phase"`` and the phase
            panel of ``"mod-phase"`` use π-based y tick labels.
            Dual-panel formats (``"mod-phase"``, ``"real-imag"``) require
            ``ax`` to be a list of two Axes or None.
        ax : plt.Axes or list of plt.Axes, optional
            Axes to draw on. Created automatically if None.
        fontname : str, optional
            Font family for all text, default "DejaVu Serif".
        fontsize : float, optional
            Font size for tick labels and axis labels, default 12.
        title : str, optional
            Plot title; defaults to ``self.name``.
        title_size : float, optional
            Font size for the title, default 12.
        major_y_locator : int, optional
            Number of major divisions on the y axis, default 4.
        minor_y_locator : int, optional
            Number of minor divisions per major division on the y axis, default 4.
        major_x_locator : int, optional
            Number of major divisions on the x axis, default 4.
        minor_x_locator : int, optional
            Number of minor divisions per major division on the x axis, default 4.
        color : str, optional
            Line colour, default "blue".
        linestyle : str, optional
            Matplotlib line style string, default "-".
        ylabel : str, optional
            Y-axis label; derived from format and units if None.
        xlabel : str, optional
            X-axis label; defaults to frequency with SI unit symbol.
        decimals_y : int, optional
            Decimal places shown on y tick labels, default 2.
        decimals_x : int, optional
            Decimal places shown on x tick labels, default 2.
        bottom_ylim : float, optional
            Lower y-axis limit.
        top_ylim : float, optional
            Upper y-axis limit.
        grid : bool, optional
            Whether to draw a grid, default True.

        Returns
        -------
        ax : plt.Axes or list of plt.Axes
            Single axes for single-panel formats, list of two for dual-panel formats.
        img : list
            List of Line2D objects from matplotlib.
        """
        assert format in ["mod-phase", "mod", "phase", "real", "imag", "real-imag"]
        xlabel = f"Frequency ({ureg.hertz:~P})" if xlabel is None else xlabel
        measurements_backup = deepcopy(self.measurements)
        if format in ["mod", "real", "imag"]:
            if ax is None:
                fig, ax = plt.subplots()
            ax.xaxis.set_units(ureg.hertz)
            if format == "mod":
                self.measurements = abs(measurements_backup)
            elif format == "real":
                self.measurements = measurements_backup.real
                ylabel = (
                    f"Real part of\namplitude ({self.measurements_units.u:~P})"
                    if ylabel is None
                    else ylabel
                )
            elif format == "imag":
                self.measurements = measurements_backup.imag
                ylabel = (
                    f"Imaginary part of\namplitude ({self.measurements_units.u:~P})"
                    if ylabel is None
                    else ylabel
                )
            ax, img = super().plot(
                ax=ax,
                fontname=fontname,
                fontsize=fontsize,
                title=title,
                title_size=title_size,
                major_y_locator=major_y_locator,
                minor_y_locator=minor_y_locator,
                major_x_locator=major_x_locator,
                minor_x_locator=minor_x_locator,
                color=color,
                linestyle=linestyle,
                ylabel=ylabel,
                xlabel=xlabel,
                decimals_y=decimals_y,
                decimals_x=decimals_x,
                bottom_ylim=bottom_ylim,
                top_ylim=top_ylim,
                grid=grid,
                log=format == "mod",
            )
        if format == "phase":
            self.measurements = np.angle(measurements_backup.m)
            if ax is None:
                fig, ax = plt.subplots()
            ax, img = super().plot(
                ax=ax,
                fontname=fontname,
                fontsize=fontsize,
                title=title,
                title_size=title_size,
                major_y_locator=major_y_locator,
                minor_y_locator=minor_y_locator,
                major_x_locator=major_x_locator,
                minor_x_locator=minor_x_locator,
                color=color,
                linestyle=linestyle,
                ylabel=ylabel,
                xlabel=xlabel,
                decimals_y=decimals_y,
                decimals_x=decimals_x,
                bottom_ylim=bottom_ylim,
                top_ylim=top_ylim,
                grid=grid,
                log=False,
            )
            # y=np.reshape(self.measurements, (len(self), -1))
            if bottom_ylim is None or top_ylim is None:
                # top = np.nanmax(y.m)
                # bottom = np.nanmin(y.m)
                # span = np.abs(top - bottom)
                # bottom_ylim = np.pi * np.ceil(abs(bottom)/np.pi)
                # top_ylim = top + 0.125 * span if top_ylim is None else top_ylim
                bottom_ylim = -np.pi - np.pi / 8
                top_ylim = np.pi + np.pi / 8
            ax.set_ylim(top=top_ylim, bottom=bottom_ylim)
            y_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
            ax.set_yticks([tick for tick in y_ticks])
            y_ticks_labels = [
                r"$\mathregular{-\pi}$",
                r"$\mathregular{-\dfrac{\pi}{2}}$",
                r"$\mathregular{0}$",
                r"$\mathregular{\dfrac{\pi}{2}}$",
                r"$\mathregular{\pi}$",
            ]
            ax.set_yticklabels([label for label in y_ticks_labels])
            y_span = top_ylim - bottom_ylim
            y_minor_step = y_span / (major_y_locator * minor_y_locator + 2)
            y_minor_ticks = np.arange(
                bottom_ylim, top_ylim + y_minor_step / 2, y_minor_step
            )
            ax.set_yticks([tick for tick in y_minor_ticks], minor=True)
            ylabel = f"Phase ({ureg.radian:~P})" if ylabel is None else ylabel
            ax.set_ylabel(ylabel, fontname=fontname, fontsize=fontsize)
        if format == "mod-phase":
            self.measurements = abs(measurements_backup)
            if ax is None:
                fig, ax = plt.subplots(2, 1)
            else:
                assert len(ax) == 2
            ax1, img = super().plot(
                ax=ax[0],
                fontname=fontname,
                fontsize=fontsize,
                title=title,
                title_size=title_size,
                major_y_locator=major_y_locator,
                minor_y_locator=minor_y_locator,
                major_x_locator=major_x_locator,
                minor_x_locator=minor_x_locator,
                color=color,
                linestyle=linestyle,
                ylabel=ylabel,
                xlabel="",
                decimals_y=decimals_y,
                decimals_x=decimals_x,
                bottom_ylim=bottom_ylim,
                top_ylim=top_ylim,
                grid=grid,
                log=True,
            )
            ax1.set_xticklabels([])
            self.measurements = np.angle(measurements_backup.m)
            ax2, img = super().plot(
                ax=ax[1],
                fontname=fontname,
                fontsize=fontsize,
                title="",
                title_size=title_size,
                major_y_locator=major_y_locator,
                minor_y_locator=minor_y_locator,
                major_x_locator=major_x_locator,
                minor_x_locator=minor_x_locator,
                color=color,
                linestyle=linestyle,
                ylabel=ylabel,
                xlabel=xlabel,
                decimals_y=decimals_y,
                decimals_x=decimals_x,
                bottom_ylim=bottom_ylim,
                top_ylim=top_ylim,
                grid=grid,
                log=False,
            )
            # y=np.reshape(self.measurements, (len(self), -1))
            if bottom_ylim is None or top_ylim is None:
                # top = np.nanmax(y.m)
                # bottom = np.nanmin(y.m)
                # span = np.abs(top - bottom)
                # bottom_ylim = np.pi * np.ceil(abs(bottom)/np.pi)
                # top_ylim = top + 0.125 * span if top_ylim is None else top_ylim
                bottom_ylim = -np.pi - np.pi / 8
                top_ylim = np.pi + np.pi / 8
            ax2.set_ylim(top=top_ylim, bottom=bottom_ylim)
            y_ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
            ax2.set_yticks([tick for tick in y_ticks])
            y_ticks_labels = [
                r"$\mathregular{-\pi}$",
                r"$\mathregular{-\dfrac{\pi}{2}}$",
                r"$\mathregular{0}$",
                r"$\mathregular{\dfrac{\pi}{2}}$",
                r"$\mathregular{\pi}$",
            ]
            ax2.set_yticklabels([label for label in y_ticks_labels])
            y_span = top_ylim - bottom_ylim
            y_minor_step = y_span / (major_y_locator * minor_y_locator + 2)
            y_minor_ticks = np.arange(
                bottom_ylim, top_ylim + y_minor_step / 2, y_minor_step
            )
            ax2.set_yticks([tick for tick in y_minor_ticks], minor=True)
            ylabel = f"Phase ({ureg.radian:~P})" if ylabel is None else ylabel
            ax2.set_ylabel(ylabel, fontname=fontname, fontsize=fontsize)
            ax = [ax1, ax2]
        if format == "real-imag":
            self.measurements = measurements_backup.real
            if ax is None:
                fig, ax = plt.subplots(2, 1)
            else:
                assert len(ax) == 2
            ax1, img = super().plot(
                ax=ax[0],
                fontname=fontname,
                fontsize=fontsize,
                title=title,
                title_size=title_size,
                major_y_locator=major_y_locator,
                minor_y_locator=minor_y_locator,
                major_x_locator=major_x_locator,
                minor_x_locator=minor_x_locator,
                color=color,
                linestyle=linestyle,
                ylabel=f"Real part of\namplitude ({self.measurements_units.u:~P})",
                xlabel="",
                decimals_y=decimals_y,
                decimals_x=decimals_x,
                bottom_ylim=bottom_ylim,
                top_ylim=top_ylim,
                grid=grid,
                log=False,
            )
            ax1.set_xticklabels([])
            self.measurements = measurements_backup.imag
            ax2, img = super().plot(
                ax=ax[1],
                fontname=fontname,
                fontsize=fontsize,
                title="",
                title_size=title_size,
                major_y_locator=major_y_locator,
                minor_y_locator=minor_y_locator,
                major_x_locator=major_x_locator,
                minor_x_locator=minor_x_locator,
                color=color,
                linestyle=linestyle,
                ylabel=f"Imaginary part of\namplitude ({self.measurements_units.u:~P})",
                xlabel=xlabel,
                decimals_y=decimals_y,
                decimals_x=decimals_x,
                bottom_ylim=bottom_ylim,
                top_ylim=top_ylim,
                grid=grid,
                log=False,
            )
            ax = [ax1, ax2]
        self.measurements = measurements_backup
        del measurements_backup
        return ax, img


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    time = np.arange(0, 30 + 0.05, 0.1)
    signal = np.sin(1 * time)
    signal = np.vstack((signal, np.sin(2 * time)))
    signal = np.vstack((signal, np.sin(3 * time)))
    signal = np.vstack((signal, np.sin(4 * time)))
    signal = np.vstack((signal, np.sin(5 * time)))
    signal = signal.reshape((time.shape[0], -1))
    signal = np.fft.fft(signal, axis=0)
    test_object = frf(signal, freq_end=5)
    print(len(test_object))
    test_object.plot("mod-phase")
    plt.show()
    test_object.plot("real")
    plt.show()
    test_object.plot("imag")
    plt.show()
    test_object.plot("phase")
    plt.show()
    print(test_object.change_freq_span(new_max_freq=10).measurements.shape)
    test_object.change_freq_span(new_max_freq=10).plot()
    plt.show()
    print(test_object.change_freq_resolution(new_resolution=0.2).measurements.shape)
    test_object.change_freq_resolution(new_resolution=0.2).plot("real-imag")
    plt.show()
    print(test_object[0:2].measurements.shape)
