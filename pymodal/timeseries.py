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
        data: npt.NDArray[np.complex64],
        coordinates: npt.NDArray[np.float64] = None,
        orientations: npt.NDArray[np.float64] = None,
        dof: Optional[float] = None,
        time_start: Optional[float] = 0,
        time_end: Optional[float] = None,
        time_span: Optional[float] = None,
        sampling_rate: Optional[float] = None,
        units: Optional[str] = None,
        system_type: str = "SIMO",
    ):
        super().__init__(
            measurements=data,
            coordinates=coordinates,
            orientations=orientations,
            dof=dof,
            domain_start=time_start,
            domain_end=time_end,
            domain_span=time_span,
            domain_resolution=sampling_rate,
            units=units,
            system_type=system_type,
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
            assert excitation.dof == 1
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
            data=frf_amp,
            coordinates=self.coordinates,
            orientations=self.orientations,
            dof=self.dof,
            freq_start=0,
            freq_end=1/(2*self.sampling_rate),
            freq_span=1/(2*self.sampling_rate),
            freq_resolution=1/self.time_span,
            units=self.units/excitation.units,
            system_type=self.system_type,
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
    excitation_test = timeseries(np.sin(1*time), time_end=30, system_type="excitation")
    print(test_object.to_FRF(excitation_test).measurements.shape)
    assert np.allclose(time, test_object.time_array)
    print(test_object.change_time_span(new_max_time=20).measurements.shape)
    print(test_object.change_sampling_rate(new_sampling_rate=0.2).measurements.shape)
    print(test_object.measurements.dimensionality)
    print(test_object[0:2].measurements.shape)
    print(test_object.measurements.shape)
