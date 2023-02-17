import numpy as np
from warnings import warn
from typing import Optional
import numpy.typing as npt
import pymodal
from .signal import _signal

class frf(_signal):
    def __init__(
        self,
        data: npt.NDArray[np.complex64],
        coordinates: npt.NDArray[np.float64] = None,
        orientations: npt.NDArray[np.float64] = None,
        dof: Optional[float] = None,
        freq_start: Optional[float] = 0,
        freq_end: Optional[float] = None,
        freq_span: Optional[float] = None,
        freq_resolution: Optional[float] = None,
        units: Optional[str] = None,
        system_type: str = "SIMO",
    ):
        super(frf, self).__init__(
            measurements=data,
            coordinates=coordinates,
            orientations=orientations,
            dof=dof,
            domain_start=freq_start,
            domain_end=freq_end,
            domain_span=freq_span,
            domain_resolution=freq_resolution,
            units=units,
            system_type=system_type
        )
        self.freq_start = self.domain_start
        self.freq_end = self.domain_end
        self.freq_span = self.domain_span
        self.freq_resolution = self.domain_resolution
        self.freq_array = self.domain_array
        self.change_freq_span = self.change_domain_span
        self.change_freq_resolution = self.change_domain_resolution