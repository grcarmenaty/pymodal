import numpy as np
from typing import Optional
import numpy.typing as npt
from pymodal import _signal, lineplot
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
        label: Optional[str] = None,
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
        label : Optional[str], optional
            An identifying label for the measurements stored in this instance, by
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
            label=label,
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
