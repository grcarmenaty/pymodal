import numpy as np
from typing import Optional
import numpy.typing as npt
from pymodal import _signal
from pyFRF import FRF
from pint import UnitRegistry
from matplotlib import pyplot as plt
from warnings import catch_warnings, filterwarnings

ureg = UnitRegistry()


class timeseries(_signal):
    def __init__(
        self,
        measurements: npt.NDArray[np.complex64],
        coordinates: npt.NDArray[np.float64] = None,
        orientations: npt.NDArray[np.float64] = None,
        dof: Optional[float] = None,
        time_start: Optional[float] = 0,
        time_end: Optional[float] = None,
        time_span: Optional[float] = None,
        time_step: Optional[float] = None,
        measurements_units: Optional[str] = None,
        time_units: Optional[str] = None,
        space_units: Optional[str] = None,
        method: str = "SIMO",
        name: Optional[str] = None,
    ):
        """Store vibrational time-domain measurements from a three-dimensional body.

        Parameters
        ----------
        measurements : numpy array of floats
            Up to three-dimensional array where the first dimension contains the
            measurements as they evolve through time, and the remaining dimensions
            correspond to degrees of freedom and the measurement method.
        coordinates : numpy array of floats, optional
            Spatial coordinates of the measurement DOF, by default None.
        orientations : numpy array of floats, optional
            Unit vectors representing the measurement direction at each coordinate,
            by default None.
        dof : float, optional
            Number of measured degrees of freedom, by default None.
        time_start : float, optional
            Starting time value in seconds, by default 0.
        time_end : float, optional
            Maximum time value in seconds, by default None.
        time_span : float, optional
            Total duration in seconds, by default None.
        time_step : float, optional
            Time between consecutive samples in seconds (Δt = 1 / fs).
            ``sampling_rate`` (in Hz) is derived as 1 / time_step.
            By default None.
        measurements_units : string, optional
            Units for the stored measurements. Defaults to "newton" for excitation
            signals and "millimeter / second ** 2" for responses.
        time_units : string, optional
            Units for the time axis, by default None (seconds).
        space_units : string, optional
            Units for spatial coordinates, by default "millimeter".
        method : string, optional
            Measurement configuration: "SIMO", "MISO", "MIMO", or "excitation".
            By default "SIMO".
        name : string, optional
            Identifying label for this signal.
        """
        super().__init__(
            measurements=measurements,
            coordinates=coordinates,
            orientations=orientations,
            dof=dof,
            domain_start=time_start,
            domain_end=time_end,
            domain_span=time_span,
            domain_resolution=time_step,
            measurements_units=measurements_units,
            domain_units=time_units,
            space_units=space_units,
            method=method,
            name=name,
        )
        self.time_start = self.domain_start
        self.time_end = self.domain_end
        self.time_span = self.domain_span
        self.time_step = self.domain_resolution
        # sampling_rate is the sampling frequency in Hz (= 1 / time_step).
        self.sampling_rate = (1.0 / self.time_step) if self.time_step is not None else None
        self.time_units = self.domain_units
        self.time_array = self.domain_array

    def change_time_span(
        self, new_min_time: Optional[float] = None, new_max_time: Optional[float] = None
    ):
        """Crop or extend the signal to new time limits.

        Parameters
        ----------
        new_min_time : float, optional
            New start time. Negative values prepend zeros. By default None.
        new_max_time : float, optional
            New end time. Values beyond the current end append zeros. By default None.

        Returns
        -------
        timeseries
            Deep copy with the modified time axis and measurement array.
        """
        return super().change_domain_span(
            new_min_domain=new_min_time, new_max_domain=new_max_time
        )

    def change_sampling_rate(self, new_sampling_rate: float):
        """Resample the signal to a new sampling frequency.

        Parameters
        ----------
        new_sampling_rate : float
            Desired sampling frequency in Hz (fs = 1 / Δt).

        Returns
        -------
        timeseries
            Deep copy with data interpolated to the new sampling frequency.
        """
        return super().change_domain_resolution(new_resolution=1.0 / new_sampling_rate)

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
        """Plot the time-domain signal with time on the x axis.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to draw on. A new figure is created if None.
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
            Y-axis label; defaults to "Amplitude (<units>)".
        xlabel : str, optional
            X-axis label; defaults to "Time (s)".
        decimals_y : int, optional
            Decimal places shown on y tick labels, default 2.
        decimals_x : int, optional
            Decimal places shown on x tick labels, default 2.
        bottom_ylim : float, optional
            Lower y-axis limit; auto-computed with 12.5 % margin if None.
        top_ylim : float, optional
            Upper y-axis limit; auto-computed with 12.5 % margin if None.
        grid : bool, optional
            Whether to draw a grid, default True.

        Returns
        -------
        ax : plt.Axes
        img : list of Line2D
        """
        xlabel = f"Time ({ureg.second:~P})" if xlabel is None else xlabel
        if ax is None:
            fig, ax = plt.subplots()
        ax.xaxis.set_units(ureg.second)
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
        return ax, img

    def to_FRF(
        self,
        excitation: "_signal",
        FRF_type: str = "H1",
        resp_delay: int = 0,
    ):
        """Compute the FRF between this response and the given excitation.

        Parameters
        ----------
        excitation : timeseries
            Excitation signal (``method="excitation"``).
        FRF_type : str, optional
            Estimator: "H1", "H2", "Hv", "vector", or "ODS". Default "H1".
        resp_delay : int, optional
            Response time delay with respect to the excitation, in samples. Default 0.

        Returns
        -------
        frf
            Computed frequency response function.
        """
        assert excitation.method == "excitation"
        assert self.space_units == excitation.space_units
        # Derive response type and FRF form from measurement units.
        if self.measurements.check("[length]"):
            resp_type = "d"
            form = "receptance"
        elif self.measurements.check("[length] / [time]"):
            resp_type = "v"
            form = "mobility"
        elif self.measurements.check("[length] / [time]**2"):
            resp_type = "a"
            form = "accelerance"
        elif self.measurements.check(""):
            resp_type = "e"
            form = "receptance"
        else:
            raise ValueError(
                f"Unrecognized response units: {self.measurements_units}."
            )
        # Derive excitation type from excitation units.
        if excitation.measurements.check("[force]"):
            exc_type = "f"
        elif excitation.measurements.check("[length]"):
            exc_type = "d"
        elif excitation.measurements.check("[length] / [time]"):
            exc_type = "v"
        elif excitation.measurements.check("[length] / [time]**2"):
            exc_type = "a"
        elif excitation.measurements.check(""):
            exc_type = "e"
        else:
            raise ValueError(
                f"Unrecognized excitation units: {excitation.measurements_units}."
            )
        fs = int(round(self.sampling_rate.magnitude))
        with catch_warnings():
            filterwarnings(
                "ignore",
                message="The unit of the quantity is stripped when downcasting"
                " to ndarray.",
            )
            if self.method == "excitation":
                raise ValueError("Use this method only with responses.")
            elif self.method == "SIMO":
                assert excitation.dof == 1
                exc = excitation.measurements[:, 0, 0].magnitude
                frf_amp = []
                for i in range(self.dof):
                    resp = self.measurements[:, i, 0].magnitude
                    frf_amp.append(
                        FRF(
                            sampling_freq=fs,
                            exc=exc,
                            resp=resp,
                            exc_type=exc_type,
                            resp_type=resp_type,
                            window="none",
                            resp_delay=resp_delay,
                            noverlap=0,
                        ).get_FRF(type=FRF_type, form=form)
                    )
                frf_amp = np.array(frf_amp).reshape((-1, self.dof, 1))
            elif self.method == "MISO":
                frf_amp = []
                for i in range(self.dof):
                    exc = excitation.measurements[:, 0, i].magnitude
                    resp = self.measurements[:, 0, i].magnitude
                    frf_amp.append(
                        FRF(
                            sampling_freq=fs,
                            exc=exc,
                            resp=resp,
                            exc_type=exc_type,
                            resp_type=resp_type,
                            window="none",
                        ).get_FRF(type=FRF_type, form=form)
                    )
                frf_amp = np.array(frf_amp).reshape((-1, 1, self.dof))
            elif self.method == "MIMO":
                outer_frf_amp = []
                for i in range(self.dof):
                    inner_frf = []
                    for j in range(self.dof):
                        exc = excitation.measurements[:, 0, i].magnitude
                        resp = self.measurements[:, i, j].magnitude
                        inner_frf.append(
                            FRF(
                                sampling_freq=fs,
                                exc=exc,
                                resp=resp,
                                exc_type=exc_type,
                                resp_type=resp_type,
                                window="none",
                            ).get_FRF(type=FRF_type, form=form)
                        )
                    outer_frf_amp.append(np.array(inner_frf))
                frf_amp = np.array(outer_frf_amp).reshape((-1, self.dof, self.dof))
        excitation_units = ureg.parse_expression(str(excitation.measurements_units))
        measurements_units = ureg.parse_expression(str(self.measurements_units))
        from pymodal import frf

        return frf(
            measurements=frf_amp,
            coordinates=self.coordinates,
            orientations=self.orientations,
            dof=self.dof,
            freq_resolution=1 / self.time_span,
            measurements_units=measurements_units / excitation_units,
            space_units=self.space_units,
            method=self.method,
        )


if __name__ == "__main__":

    time = np.arange(0, 30 + 0.05, 0.1)
    signal = np.sin(1 * time)
    signal = np.vstack((signal, np.sin(2 * time)))
    signal = np.vstack((signal, np.sin(3 * time)))
    signal = np.vstack((signal, np.sin(4 * time)))
    signal = np.vstack((signal, np.sin(5 * time)))
    signal = signal.reshape((time.shape[0], -1))
    test_object = timeseries(signal, time_end=30)
    test_object.plot()
    plt.show()
    print(test_object.measurements.shape)
    excitation_test = timeseries(np.sin(1 * time), time_end=30, method="excitation")
    frf_test = test_object.to_FRF(excitation_test)
    frf_test.plot()
    plt.show()
    print(frf_test.measurements.shape)
    assert np.allclose(time, test_object.time_array.magnitude)
    print(test_object.change_time_span(new_max_time=20).measurements.shape)
    print(test_object.change_sampling_rate(new_sampling_rate=5.0).measurements.shape)
    print(test_object.measurements.dimensionality)
    print(test_object[0:2].measurements.shape)
    print(test_object.measurements.shape)
