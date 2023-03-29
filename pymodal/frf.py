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
            space_units=space_units,
            method=method,
            label=label,
        )
        self.freq_start = self.domain_start
        self.freq_end = self.domain_end
        self.freq_span = self.domain_span
        self.freq_resolution = self.domain_resolution
        self.freq_array = self.domain_array
        self.freq_array = self.freq_array * ureg.hertz

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
        format: str = "mod-phase",
        ax: plt.Axes = None,
        fontname: str = "serif",
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
        decimals_y: int = 0,
        decimals_x: int = 2,
        bottom_ylim: float = None,
        top_ylim: float = None,
        grid: bool = True,
    ):
        xlabel = f"Frequency ({ureg.hertz:~P})" if xlabel is None else xlabel
        if ax is None:
            fig, ax = plt.subplots()
        ax.xaxis.set_units(ureg.hertz)
        measurements_backup = deepcopy(self.measurements)
        self.measurements = abs(measurements_backup)
        img, ax = super().plot(ax=ax,
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
            log=True,
        )
        self.measurements = measurements_backup
        del measurements_backup
        return img, ax

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from pymodal import timeseries
    
    time = np.arange(0, 30 + 0.05, 0.1)
    signal = np.sin(1 * time)
    signal = np.vstack((signal, np.sin(2 * time)))
    signal = np.vstack((signal, np.sin(3 * time)))
    signal = np.vstack((signal, np.sin(4 * time)))
    signal = np.vstack((signal, np.sin(5 * time)))
    signal = signal.reshape((time.shape[0], -1))
    signal = np.fft.fft(signal, axis=0)
    test_object = frf(signal, freq_end=5)
    test_object.plot()
    plt.show()
    print(test_object.change_freq_span(new_max_freq=10).measurements.shape)
    test_object.change_freq_span(new_max_freq=10).plot()
    plt.show()
    print(test_object.change_freq_resolution(new_resolution=0.2).measurements.shape)
    test_object.change_freq_resolution(new_resolution=0.2).plot()
    plt.show()
    print(test_object[0:2].measurements.shape)
