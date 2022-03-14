import pymodal.papergraph # noqa F401
from .utils import (  # noqa F401
    save_array,
    load_array,
    load_FRF,
    plot_FRF,
    value_CFDAC,
    value_CFDAC_A,
    value_FDAC,
    value_RVAC,
    value_RVAC_2d,
    value_GAC,
    FRFRMS,
    FRFSF,
    FRFSM,
    ODS_diff,
    r2_imag,
    # compress,
    SCI,
    DRQ,
    AIGAC,
    unsigned_SCI,
    M2L_func,
    M2L,
    plot_CFDAC,
    damping_coefficient,
    synthetic_FRF,
    modal_superposition,
)
import pymodal.mapdl
from .FRF_class import FRF
from .signal_class import signal
