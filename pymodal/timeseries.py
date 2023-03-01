import numpy as np
from typing import Optional
import numpy.typing as npt
from pymodal import _signal, frf
from pyFRF import FRF
from pint import UnitRegistry


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
        sampling_rate: Optional[float] = None,
        measurements_units: Optional[str] = None,
        space_units: Optional[str] = None,
        method: str = "SIMO",
        label: Optional[str] = None,
    ):
        """_summary_

        Parameters
        ----------
        measurements : numpy array of floats
            A numpy array of up to three dimensions where the first one contains the
            measurements as they change along the temporal domain, and the rest are
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
        time_start : float, optional
            Starting value of the temporal domain, by default 0.
        time_end : float, optional
            Maximum value of the temporal domain, by default None.
        time_span : float, optional
            Total duration of the temporal domain, by default None.
        sampling_rate : float, optional
            Temporal domain quantity between two consecutive measurement points, by
            default None.
        measurements_units : string, optional
            Units used for the measurements stored within the instance of this class,
            they are assumed to be Newtons, millimeters and seconds by default None.
        space_units : string, optional
            Units used for the spatial coordinates of the degrees of freedom, they are
            assumed to be millimeters by default None
        method : string, optional
            Whether the method used to get the measurements is Multiple Input Single
            Output (MISO), Single Input Multiple Output (SIMO), Multiple Input Multiple
            Output (MIMO), or a recording of the excitation inputs, by default "SIMO"
        label : string, optional
            An identifying label for the measurements stored in this instance of the
            signal class.
        """
        super().__init__(
            measurements=measurements,
            coordinates=coordinates,
            orientations=orientations,
            dof=dof,
            domain_start=time_start,
            domain_end=time_end,
            domain_span=time_span,
            domain_resolution=sampling_rate,
            measurements_units=measurements_units,
            space_units=space_units,
            method=method,
            label=label
        )
        self.time_start = self.domain_start
        self.time_end = self.domain_end
        self.time_span = self.domain_span
        self.sampling_rate = self.domain_resolution
        self.time_array = self.domain_array

    def change_time_span(
        self, new_min_time: Optional[float] = None, new_max_time: Optional[float] = None
    ):
        return super().change_domain_span(
            new_min_domain=new_min_time, new_max_domain=new_max_time
        )

    def change_sampling_rate(self, new_sampling_rate):
        return super().change_domain_resolution(new_resolution=new_sampling_rate)

    def to_FRF(self, excitation, type="H1"):
        assert excitation.system_type == "excitation"
        assert self.space_units == excitation.space_units
        # Get response type and desired FRF form from units.
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
        # Get excitation type from excitation units.
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
        if self.system_type == "excitation":
            raise ValueError("Use this method only with responses.")
        elif self.system_type == "SIMO":
            # If there's a single input and multiple outputs, then for every output,
            # an FRF will be calculated with the provided single input excitation.
            assert excitation.dof == 1
            # assert excitations coordinates-orientation pair are in
            # self coordinates-orientations pairs list
            exc = excitation.measurements[:, 0, 0].magnitude
            frf_amp = []
            for i in range(self.dof):
                resp = self.measurements[:, i, 0].magnitude
                frf_amp.append(
                    FRF(
                        sampling_freq=self.sampling_rate,
                        exc=exc,
                        resp=resp,
                        exc_type=exc_type,
                        resp_type=resp_type,
                        exc_window="None",
                        resp_window="None",
                    ).get_FRF(type=type, form=form)
                )
            frf_amp = np.array(frf_amp).reshape((-1, self.dof, 1))
        elif self.system_type == "MISO":
            # If there's a single output and multiple inputs, then for every input,
            # an FRF will be calculated with the provided single output measurement.
            # assert self coordinates-orientation pair are in excitation
            # coordinates-orientations pairs list
            frf_amp = []
            for i in range(self.dof):
                exc = excitation.measurements[:, 0, i].magnitude
                resp = self.measurements[:, 0, i].magnitude
                frf_amp.append(
                    FRF(
                        sampling_freq=self.sampling_rate,
                        exc=exc,
                        resp=resp,
                        exc_type=exc_type,
                        resp_type=resp_type,
                        exc_window="None",
                        resp_window="None",
                    ).get_FRF(type=type, form=form)
                )
            frf_amp = np.array(frf_amp).reshape((-1, 1, self.dof))
        elif self.system_type == "MIMO":
            # If the system has multiple inputs and outputs, compute an FRF for each
            # output and it's corresponding input.
            # assert excitations coordinates-orientation pairs are in
            # self coordinates-orientations pairs list, in the same order.
            outer_frf_amp = []
            for i in range(self.dof):
                inner_frf = []
                for j in range(self.dof):
                    exc = excitation.measurements[:, 0, i].magnitude
                    resp = self.measurements[:, i, j].magnitude
                    inner_frf.append(
                        FRF(
                            sampling_freq=self.sampling_rate,
                            exc=exc,
                            resp=resp,
                            exc_type=exc_type,
                            resp_type=resp_type,
                            exc_window="None",
                            resp_window="None",
                        ).get_FRF(type=type, form=form)
                    )
                outer_frf_amp.append(np.array(inner_frf))
            frf_amp = np.array(outer_frf_amp).reshape((-1, self.dof, self.dof))
        return frf(
            measurements=frf_amp,
            coordinates=self.coordinates,
            orientations=self.orientations,
            dof=self.dof,
            freq_start=0,
            freq_end=1 / (2 * self.sampling_rate),
            freq_span=1 / (2 * self.sampling_rate),
            freq_resolution=1 / self.time_span,
            measurements_units=self.measurements_units / excitation.measurements_units,
            space_units=self.space_units,
            method=self.system_type,
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
    print(test_object.measurements.shape)
    excitation_test = timeseries(
        np.sin(1 * time), time_end=30, method="excitation"
    )
    print(test_object.to_FRF(excitation_test).measurements.shape)
    assert np.allclose(time, test_object.time_array)
    print(test_object.change_time_span(new_max_time=20).measurements.shape)
    print(test_object.change_sampling_rate(new_sampling_rate=0.2).measurements.shape)
    print(test_object.measurements.dimensionality)
    print(test_object[0:2].measurements.shape)
    print(test_object.measurements.shape)
