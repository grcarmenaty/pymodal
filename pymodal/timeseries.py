import numpy as np
from typing import Optional
import numpy.typing as npt
from pymodal import _signal


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
        time_resolution: Optional[float] = None,
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
            domain_resolution=time_resolution,
            units=units,
            system_type=system_type,
        )
        self.time_start = self.domain_start
        self.time_end = self.domain_end
        self.time_span = self.domain_span
        self.time_resolution = self.domain_resolution
        self.time_array = self.domain_array

    def change_time_span(
        self, new_min_time: Optional[float] = None, new_max_time: Optional[float] = None
    ):
        return super().change_domain_span(
            new_min_domain=new_min_time, new_max_domain=new_max_time
        )

    def change_time_resolution(self, new_resolution):
        return super().change_domain_resolution(new_resolution=new_resolution)


if __name__ == "__main__":
    time = np.arange(0, 30 + 0.05, 0.1)
    signal = np.sin(1 * time)
    signal = np.vstack((signal, np.sin(2 * time)))
    signal = np.vstack((signal, np.sin(3 * time)))
    signal = np.vstack((signal, np.sin(4 * time)))
    signal = np.vstack((signal, np.sin(5 * time)))
    signal = signal.reshape((time.shape[0], -1))
    test_object = timeseries(signal, time_end=30)
    assert np.allclose(time, test_object.time_array)
    test_object.change_time_span(new_max_time=20)
    test_object.change_time_resolution(new_resolution=0.2)
