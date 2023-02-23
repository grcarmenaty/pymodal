import numpy as np
from typing import Optional
import numpy.typing as npt
from .signal import _signal


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
        super(timeseries, self).__init__(
            measurements=data,
            coordinates=coordinates,
            orientations=orientations,
            dof=dof,
            domain_start=time_start,
            domain_end=time_end,
            domain_span=time_span,
            domain_resolution=time_resolution,
            units=units,
            system_type=system_type
        )
        self.time_start = self.domain_start
        self.time_end = self.domain_end
        self.time_span = self.domain_span
        self.time_resolution = self.domain_resolution
        self.time_array = self.domain_array
        self.change_time_span = self.change_domain_span
        self.change_time_resolution = self.change_domain_resolution


if __name__ == "__main__":
    pass
