import numpy as np
from typing import Optional
import numpy.typing as npt
from pymodal import _signal


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
        measurements_units: Optional[str] = None,
        space_units: Optional[str] = None,
        method: str = "SIMO",
        label: Optional[str] = None,
    ):
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
            label=label
        )
        self.freq_start = self.domain_start
        self.freq_end = self.domain_end
        self.freq_span = self.domain_span
        self.freq_resolution = self.domain_resolution
        self.freq_array = self.domain_array

    def change_freq_span(
        self, new_min_freq: Optional[float] = None, new_max_freq: Optional[float] = None
    ):
        return super().change_domain_span(
            new_min_domain=new_min_freq, new_max_domain=new_max_freq
        )

    def change_freq_resolution(self, new_resolution):
        return super().change_domain_resolution(new_resolution=new_resolution)


if __name__ == "__main__":
    freq = np.arange(0, 30 + 0.05, 0.1)
    signal = np.sin(1 * freq) + np.sin(1 * freq) * 1j
    signal = np.vstack((signal, np.sin(2 * freq) + np.sin(2 * freq) * 1j))
    signal = np.vstack((signal, np.sin(3 * freq) + np.sin(3 * freq) * 1j))
    signal = np.vstack((signal, np.sin(4 * freq) + np.sin(4 * freq) * 1j))
    signal = np.vstack((signal, np.sin(5 * freq) + np.sin(5 * freq) * 1j))
    signal = signal.reshape((freq.shape[0], -1))
    test_object = frf(signal, freq_end=30)
    assert np.allclose(freq, test_object.freq_array)
    print(test_object.change_freq_span(new_max_freq=20).measurements.shape)
    print(test_object.change_freq_resolution(new_resolution=0.2).measurements.shape)
    print(test_object[0:2].measurements.shape)
