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


def unfinished_test_los_alamos_building():
    working_dir = path / "mapdl-ansys"
    working_dir.mkdir(exist_ok=True)
    mapdl = launch_mapdl(run_location=working_dir, override=True, nproc=8)
    pymodal.mapdl.los_alamos_building(
        mapdl=mapdl,
        floor_elastic_moduli=[
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
        ],
        column_elastic_moduli=[
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
            72000000000.0,
        ],
        Poisson=0,
        floor_densities=[
            2810,
            2810,
            2810,
            2810,
        ],
        column_densities=[
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
            2810,
        ],
        e_size=0.01,
        column_thicknesses=[
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
            0.006,
        ],
        column_widths=[
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
            0.025,
        ],
        floor_width=0.305,
        floor_heights=[
            0.152,
            0.152,
            0.152,
        ],
        floor_depth=0.305,
        floor_thicknesses=[
            0.025,
            0.025,
            0.025,
            0.025,
        ],
        contact_strengths=[
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
            1e12,
        ],
        mass_coordinates=[
            [0, 0.025/2, 0.025/2],
            [0, 0.305-0.025/2, 0.025/2],
            [0, 0.025/2, 0.152+0.025/2],
            [0, 0.305-0.025/2, 0.152+0.025/2],
            [0, 0.025/2, 2*0.152+0.025/2],
            [0, 0.305-0.025/2, 2*0.152+0.025/2],
            [0, 0.025/2, 3*0.152+0.025/2],
            [0, 0.305-0.025/2, 3*0.152+0.025/2],
            [0.305, 0.025/2, 0.025/2],
            [0.305, 0.305-0.025/2, 0.025/2],
            [0.305, 0.025/2, 0.152+0.025/2],
            [0.305, 0.305-0.025/2, 0.152+0.025/2],
            [0.305, 0.025/2, 2*0.152+0.025/2],
            [0.305, 0.305-0.025/2, 2*0.152+0.025/2],
            [0.305, 0.025/2, 3*0.152+0.025/2],
            [0.305, 0.305-0.025/2, 3*0.152+0.025/2],
        ],
        mass_values=[
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
        ],
    )
    mapdl.eplot()
    pymodal.mapdl.displacement_bc(mapdl, (0, 0.305), (0.305/3-0.05, 0.305/3+0.05), (0, 0.01),
                                  x=False)
    pymodal.mapdl.displacement_bc(mapdl, (0, 0.305), (2*0.305/3-0.05, 2*0.305/3+0.05),
                                  (0, 0.1), x=False)
    modal_analysis = pymodal.mapdl.modal_analysis(
        mapdl, frequency_range=[0, 100], dof="uxuyuz"
    )
    print(modal_analysis)
    harmonic_analysis = pymodal.mapdl.harmonic_analysis(
        mapdl,
        excitation_coordinates=[0.305, 0.305/2, 0.025/2],
        response_coordinates=[
            [0.305, 0.305/2, 0.025/2],
            [0, 0.025/2, 0.025/2],
            [0, 0.305-0.025/2, 0.025/2],
            [0, 0.025/2, 0.152+0.025/2],
            [0, 0.305-0.025/2, 0.152+0.025/2],
            [0, 0.025/2, 2*0.152+0.025/2],
            [0, 0.305-0.025/2, 2*0.152+0.025/2],
            [0, 0.025/2, 3*0.152+0.025/2],
            [0, 0.305-0.025/2, 3*0.152+0.025/2],
        ],
        response_directions="X",
        excitation_vector=[100, 0, 0],
        frequency_range=[0, 100],
        damping=[12, 5e-6],
        N=400,
        magnitude='acc',
        mode_superposition=True
    )
    print(harmonic_analysis)
    mapdl.exit()
    harmonic_analysis.plot()
    plt.show()


if __name__ == "__main__":
    # unfinished_test_modal_analysis()
    unfinished_test_los_alamos_building()