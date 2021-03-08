import numpy as np
import pymodal
import pathlib
from ansys.mapdl.core import launch_mapdl
import matplotlib.pyplot as plt
import shutil
import time
import faulthandler
faulthandler.enable()

path = pathlib.Path(__file__).parent

def unfinished_test_modal_analysis():

    """

    This test is unfinished ans requires a working installation of ANSYS
    """
    working_dir = path / "mapdl-ansys"
    working_dir.mkdir(exist_ok=True)
    mapdl = launch_mapdl(run_location=working_dir, override=True, nproc=8)
    pymodal.mapdl.cantilever_beam(
        mapdl,
        elastic_modulus=185000000000.0,
        Poisson=0.15625,
        density=7917,
        damage_location=[0.9, 0.99],
        b=0.035,
        h=0.007,
        l=1.8,
        damage_level=0,
        ndiv=2000
    )
    master_coords = np.vstack((
        np.arange(0, 1.89, 0.09), np.zeros(21), np.zeros(21)
    )).T
    modal_analysis = pymodal.mapdl.modal_analysis(
        mapdl, frequency_range=[0, 100], master_coords=master_coords,
        dof="uxuyrotz"
    )
    plt.figure()
    for i in range(modal_analysis["modal_frequencies"].shape[0]):
        plt.plot(master_coords[:, 0], modal_analysis["mode_shapes"][:, 1, i],
        label=str(modal_analysis["modal_frequencies"][i]), marker="s")
        plt.legend()
    harmonic_analysis = pymodal.mapdl.harmonic_analysis(
        mapdl, excitation_coordinates=[0.36, 0, 0],
        response_coordinates=[[0.45, 0, 0]], response_directions="Y",
        excitation_vector=[0,100,0], frequency_range=[0, 100],
        damping=[12, 5e-6], N=200, magnitude='acc',
        mode_superposition=True
    )
    mapdl.exit()
    plt.figure()
    harmonic_analysis.plot(color="r")
    mapdl2 = launch_mapdl(run_location=working_dir, override=True, nproc=8)
    pymodal.mapdl.cantilever_beam(
        mapdl2,
        elastic_modulus=185000000000.0,
        Poisson=0.15625,
        density=7917,
        damage_location=[0.9, 0.99],
        b=0.035,
        h=0.007,
        l=1.8,
        damage_level=0.8,
        ndiv=2000
    )
    harmonic_analysis = pymodal.mapdl.harmonic_analysis(
        mapdl2, excitation_coordinates=[0.36, 0, 0],
        response_coordinates=[[0.45, 0, 0]], response_directions="Y",
        excitation_vector=[0,100,0], frequency_range=[0, 100],
        damping=[12, 5e-6], N=200, magnitude='acc',
        mode_superposition=True
    )
    harmonic_analysis.plot()
    mapdl.exit()
    plt.show()

if __name__ == "__main__":
    unfinished_test_modal_analysis()