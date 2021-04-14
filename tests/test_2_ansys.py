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


def unfinished_test_plate():
    working_dir = path / "mapdl-ansys"
    working_dir.mkdir(exist_ok=True)
    mapdl = launch_mapdl(run_location=working_dir, override=True, nproc=8)
    elastic_modulus = 72586933547.72092
    density = 2715.9797479
    alpha = 2.37252317
    beta = 1.687709903e-7
    # normal_stiffness = 4138667648558.9054
    # tangential_stiffness = 584391971941.3767
    pymodal.mapdl.free_plate_solid(
        mapdl=mapdl,
        elastic_modulus=elastic_modulus,
        Poisson=0.3,
        density=density,
        thickness=0.00498,
        a=0.3,
        b=0.3,
        e_size=0.005
    )
    # This kind of works with displacement bc, but elastic support is not
    # working.
    # pymodal.mapdl.elastic_support(
    #     mapdl,
    #     x_lim=[-1, 0.4],
    #     y_lim=[-1, 0.026],
    #     z_lim=[-1, 0.001],
    #     normal_stiffness=normal_stiffness,
    #     tangential_stiffness=tangential_stiffness
    # )
    # pymodal.mapdl.elastic_support(
    #     mapdl,
    #     x_lim=[-1, 0.4],
    #     y_lim=[-1, 0.026],
    #     z_lim=[0.004, 0.006],
    #     normal_stiffness=normal_stiffness,
    #     tangential_stiffness=tangential_stiffness
    # )
    pymodal.mapdl.displacement_bc(
        mapdl,
        x_lim=[-1, 0.4],
        y_lim=[-1, 0.026],
        z_lim=[-1, 0.001],
    )
    pymodal.mapdl.displacement_bc(
        mapdl,
        x_lim=[-1, 0.4],
        y_lim=[-1, 0.026],
        z_lim=[0.004, 0.006],
    )
    exp_mesh_81 = []
    mesh_spacing_x = np.arange(0.27, 0, -0.03)
    mesh_spacing_y = np.arange(0.27, 0, -0.03)
    for i in range(9):
        for j in range(9):
            exp_mesh_81.append(
                [mesh_spacing_x[j], mesh_spacing_y[i], 0.00498]
            )
    exp_mesh_81 = np.asarray(exp_mesh_81)
    pymodal.mapdl.modal_analysis(
        mapdl, frequency_range=[0, 3200], dof="uxuyuz"
    )
    mapdl.result.plot_nodal_displacement(
        9, show_displacement=True, displacement_factor=0.005
    )
    num_ref = pymodal.mapdl.harmonic_analysis(
        mapdl=mapdl,
        excitation_coordinates=exp_mesh_81[29],
        response_coordinates=exp_mesh_81,
        response_directions='Z',
        excitation_vector=[0, 0, 1],
        frequency_range=(0, 3200),
        damping=(alpha, beta),
        N=6400,
        magnitude='acc',
        mode_superposition=True
    )
    mapdl.finish()
    mapdl.exit()
    experimental_dir = working_dir.parent / 'data' / 'FRF'
    exp_ref = pymodal.load_FRF(experimental_dir / 'experimental_frf.zip')
    exp_ref[4].normalize().plot(color="b")
    num_ref.normalize().plot(color="r")
    plt.savefig("plot.png")


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
        ],
        floor_Poisson=[
            0.3,
            0.3,
            0.3,
            0.3,
        ],
        column_Poisson=[
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
        ],
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
            0.025051,
            0.025010,
            0.024987,
            0.025033,
            0.025040,
            0.025045,
            0.025034,
            0.025013,
            0.025043,
            0.025043,
            0.024979,
            0.024972,
        ],
        floor_width=0.305,
        floor_heights=[
            0.15185,
            0.15124,
            0.15190,
        ],
        floor_depth=0.305,
        floor_thicknesses=[
            0.025419,
            0.025447,
            0.025457,
            0.025453,
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
        ],
        mass_coordinates=[
            [0, 0.025/2, 0.025/2],
            [0, 0.305-0.025/2, 0.025/2],
            [0, 0.025/2, 0.152+0.025+0.025/2],
            [0, 0.305-0.025/2, 0.152+0.025+0.025/2],
            [0, 0.025/2, 2*0.152+2*0.025+0.025/2],
            [0, 0.305-0.025/2, 2*0.152+2*0.025+0.025/2],
            [0, 0.025/2, 3*0.152+3*0.025+0.025/2],
            [0, 0.305-0.025/2, 3*0.152+3*0.025+0.025/2],
            [0.305, 0.025/2, 0.025/2],
            [0.305, 0.305-0.025/2, 0.025/2],
            [0.305, 0.025/2, 0.152+0.025+0.025/2],
            [0.305, 0.305-0.025/2, 0.152+0.025+0.025/2],
            [0.305, 0.025/2, 2*0.152+2*0.025+0.025/2],
            [0.305, 0.305-0.025/2, 2*0.152+2*0.025+0.025/2],
            [0.305, 0.025/2, 3*0.152+3*0.025+0.025/2],
            [0.305, 0.305-0.025/2, 3*0.152+3*0.025+0.025/2],
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
    # mapdl.eplot()
    pymodal.mapdl.displacement_bc(mapdl, (0, 0.305), (0.305/3-0.05, 0.305/3+0.05), (0, 0.01),
                                  x=False)
    pymodal.mapdl.displacement_bc(mapdl, (0, 0.305), (2*0.305/3-0.05, 2*0.305/3+0.05),
                                  (0, 0.1), x=False)
    modal_analysis = pymodal.mapdl.modal_analysis(
        mapdl, frequency_range=[0, 100], dof="uxuyuz"
    )
    # result = mapdl.result
    # result.plot_nodal_displacement(5, show_displacement=True, displacement_factor=0.4)
    print(modal_analysis)
    harmonic_analysis = pymodal.mapdl.harmonic_analysis(
        mapdl,
        excitation_coordinates=[0.305, 0.305/2, 0.025/2],
        response_coordinates=[
            [0.305, 0.305/2, 0.025/2],
            [0, 0.025/2, 0.025/2],
            [0, 0.305-0.025/2, 0.025/2],
            [0, 0.025/2, 0.152+0.025+0.025/2],
            [0, 0.305-0.025/2, 0.152+0.025+0.025/2],
            [0, 0.025/2, 2*0.152+2*0.025+0.025/2],
            [0, 0.305-0.025/2, 2*0.152+2*0.025+0.025/2],
            [0, 0.025/2, 3*0.152+3*0.025+0.025/2],
            [0, 0.305-0.025/2, 3*0.152+3*0.025+0.025/2],
        ],
        response_directions="X",
        excitation_vector=[100, 0, 0],
        frequency_range=[0, 100],
        damping=[4, 8e-6],
        N=400,
        magnitude='acc',
        mode_superposition=True
    )
    print(harmonic_analysis)
    mapdl.exit()
    harmonic_analysis.plot()
    plt.show()
    plt.close()


if __name__ == "__main__":
    # unfinished_test_modal_analysis()
    unfinished_test_los_alamos_building()
    # unfinished_test_plate()