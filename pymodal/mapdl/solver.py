import numpy as np
import pathlib
from scipy.spatial.distance import cdist
import pymodal
import time
from ansys.mapdl.core.math import MapdlMath


def modal_analysis(mapdl, frequency_range, master_coords=None,
                   dof="uxuyuzrotxrotyrotz", mode_limit=999):
    mapdl.finish()
    mapdl.run("/SOL")
    mapdl.antype(2)
    mapdl.eqslv("SPAR")
    mapdl.mxpand(mode_limit, "", "", "YES")
    mapdl.lumpm(0)
    mapdl.pstres(0)
    mapdl.modopt("LANB", mode_limit, frequency_range[0], frequency_range[1])
    mapdl.solve()
    mapdl.finish()
    mapdl.post1()
    node_coordinates = mapdl.mesh.nodes
    node_coordinates = np.core.records.fromarrays(node_coordinates.transpose(),
                                                  names="x, y, z")
    mapdl.set("FIRST")
    modal_frequencies = []
    for __ in range(mapdl.post_processing.nsets):
        modal_frequencies.append(mapdl.post_processing.freq)
        mapdl.set("NEXT")
    modal_frequencies = np.array(modal_frequencies)
    if master_coords is not None:
        node_list = pymodal.mapdl.get_node_list(mapdl)
        mm = MapdlMath(mapdl)
        master_nodes = []
        for i in range(master_coords.shape[0]):
            closest_node = cdist(np.asarray([master_coords[i]]),
                                node_list[:, 1:4])
            master_nodes.append(int(node_list[
                int(np.argmin(closest_node)), 0
            ]))
        master_nodes = np.array(master_nodes)
        dof_selector = []
        if "ux" in dof:
            dof_selector.append(0)
        if "uy" in dof:
            dof_selector.append(1)
        if "uz" in dof:
            dof_selector.append(2)
        if "rotx" in dof:
            dof_selector.append(3)
        if "roty" in dof:
            dof_selector.append(4)
        if "rotz" in dof:
            dof_selector.append(5)
        dof_corrector = (len(dof_selector) - 1)*np.arange(
            master_nodes.shape[0]
        ) - 1
        dof_corrector[0] = 0
        row_col = []
        for i in range(len(dof_selector)):
            row_col.extend(master_nodes + i - 1 + dof_corrector)
        row_col = np.sort(np.array(row_col)).astype(int)
        k = mm.stiff().asarray()
        m = mm.mass().asarray()
        m = m.todense()
        m = m[:, row_col]
        m = m[row_col, :]
        k = k[:, row_col].todense()
        k = k[row_col, :]
        try:
            mapdl.set("FIRST")
            mode_shapes = []
            for i in range(modal_frequencies.shape[0]):
                mode_shapes.append(
                    np.vstack(
                        (mapdl.post_processing.nodal_displacement("X"),
                         mapdl.post_processing.nodal_displacement("Y"),
                         mapdl.post_processing.nodal_displacement("Z"),
                         mapdl.post_processing.nodal_rotation("X"),
                         mapdl.post_processing.nodal_rotation("Y"),
                         mapdl.post_processing.nodal_rotation("Z"),),
                    )[:, master_nodes.astype(int) - 1].T + 
                    np.hstack((master_coords, np.zeros(master_coords.shape)))
                )
                mapdl.set("NEXT")
            mode_shapes = np.dstack(mode_shapes)
            mode_shapes = mode_shapes[:, dof_selector, :]
        except Exception as __:
            mode_shapes = "Modal shapes unavailable"
            print(__)
        result_dict = {"modal_frequencies": modal_frequencies,
                       "mode_shapes": mode_shapes,
                       "mass_matrix": m,
                       "stiffness_matrix": k}
    else:
        result_dict = {"modal_frequencies": modal_frequencies}
    return result_dict


def get_stiffness(mapdl, coords, force_vector):
    node_list = pymodal.mapdl.get_node_list(mapdl)
    excitation_node = cdist(np.asarray([coords]),
                            node_list[:, 1:4])
    excitation_node = node_list[int(np.argmin(excitation_node)), 0]
    node_id_list = []
    mapdl.f(excitation_node, "FX",
          force_vector[0])
    mapdl.f(excitation_node, "FY",
          force_vector[1])
    mapdl.f(excitation_node, "FZ",
          force_vector[2])
    mapdl.run("/SOL")
    mapdl.antype(0)
    mapdl.solve()
    mapdl.finish()
    path = pathlib.Path(mapdl.inquire("DIRECTORY"))
    jobname = mapdl.inquire("JOBNAME")
    result = mapdl.result
    node_coordinates = result.geometry["nodes"][:, 0:3]
    node_coordinates = np.core.records.fromarrays(node_coordinates.transpose(),
                                                  names="x, y, z")
    node_result = np.hstack((
        result.nodal_solution(0)[0].reshape(-1, 1),
        result.nodal_solution(0)[1][:, 0:3]
    ))
    displacement = np.linalg.norm(
        node_result[node_result[:, 0] == excitation_node, 1:4]
    )
    force_magnitude = np.linalg.norm(np.asarray(force_vector))
    k = force_magnitude / displacement
    return k


def harmonic_analysis(
    mapdl,
    excitation_coordinates,
    response_coordinates,
    response_directions,
    excitation_vector,
    frequency_range,
    damping,
    N,
    magnitude,
    mode_superposition
):

    mapdl.prep7()
    node_list = pymodal.mapdl.get_node_list(mapdl)
    excitation_node = cdist(np.asarray([excitation_coordinates]),
                            node_list[:, 1:4])
    excitation_node = node_list[int(np.argmin(excitation_node)), 0]
    node_id_list = []
    for row in response_coordinates:
        closest_node = cdist(np.asarray([row]), node_list[:, 1:4])
        node_id_list.append(int(node_list[int(np.argmin(closest_node)), 0]))
    mapdl.f(excitation_node, "FX",
          excitation_vector[0])
    mapdl.f(excitation_node, "FY",
          excitation_vector[1])
    mapdl.f(excitation_node, "FZ",
          excitation_vector[2])
    if mode_superposition:
        modal_analysis = pymodal.mapdl.modal_analysis(mapdl, frequency_range)
    mapdl.run("/SOL")
    mapdl.antype(3)
    if mode_superposition:
        mapdl.hropt("MSUP")
    mapdl.harfrq(frequency_range[0], frequency_range[1])
    mapdl.nsubst(N)
    mapdl.kbc(0)
    mapdl.alphad(damping[0])
    mapdl.betad(damping[1])
    mapdl.solve()
    mapdl.finish()
    if mode_superposition:
        mapdl.post26()
        mapdl.file(mapdl.inquire("JOBNAME"), "RFRQ")
        mapdl.run("/UI,COLL,1  ")
        mapdl.numvar(200)
        mapdl.nsol(191, 1, "UX")
        mapdl.store("MERGE")
        mapdl.filldata(191, "", "", "", 1, 1)
        mapdl.realvar(191, 191)
        mapdl.run("*DEL,MAX_PARAM")
        mapdl.run(f"*DIM,FREQ1,ARRAY,{N}")
        mapdl.vget("FREQ1(1)", 1)
        freq_vector = np.insert(mapdl.parameters["FREQ1"], 0, 0)
        nodal_solution = []
        for i, node_id in enumerate(node_id_list):
            mapdl.run("/POST26")
            if "X" in response_directions:
                mapdl.nsol(2, node_id, "U", "X")
                mapdl.store("MERGE")
                mapdl.run(f"*DIM,DISP_REALX{i},ARRAY,{N}")
                mapdl.run(f"*DIM,DISP_IMAGX{i},ARRAY,{N}")
                mapdl.vget(f"DISP_REALX{i}(1)",2,"",0)
                mapdl.vget(f"DISP_IMAGX{i}(1)",2,"",1)
                nodal_solution.append(
                    np.insert(
                        mapdl.parameters[f"DISP_REALX{i}"],
                        0,
                        mapdl.parameters[f"DISP_REALX{i}"][0]
                    ) + np.insert(
                        mapdl.parameters[f"DISP_IMAGX{i}"],
                        0,
                        mapdl.parameters[f"DISP_IMAGX{i}"][0]
                    )*1j
                )
            if "Y" in response_directions:
                mapdl.nsol(2, node_id, "U", "Y")
                mapdl.store("MERGE")
                mapdl.run(f"*DIM,DISP_REALY{i},ARRAY,{N}")
                mapdl.run(f"*DIM,DISP_IMAGY{i},ARRAY,{N}")
                mapdl.vget(f"DISP_REALY{i}(1)",2,"",0)
                mapdl.vget(f"DISP_IMAGY{i}(1)",2,"",1)
                nodal_solution.append(
                    np.insert(
                        mapdl.parameters[f"DISP_REALY{i}"],
                        0,
                        mapdl.parameters[f"DISP_REALY{i}"][0]
                    ) + np.insert(
                        mapdl.parameters[f"DISP_IMAGY{i}"],
                        0,
                        mapdl.parameters[f"DISP_IMAGY{i}"][0]
                    )*1j
                )
            if "Z" in response_directions:
                mapdl.nsol(2, node_id, "U", "Z")
                mapdl.store("MERGE")
                mapdl.run(f"*DIM,DISP_REALZ{i},ARRAY,{N}")
                mapdl.run(f"*DIM,DISP_IMAGZ{i},ARRAY,{N}")
                mapdl.vget(f"DISP_REALZ{i}(1)",2,"",0)
                mapdl.vget(f"DISP_IMAGZ{i}(1)",2,"",1)
                nodal_solution.append(
                    np.insert(
                        mapdl.parameters[f"DISP_REALZ{i}"],
                        0,
                        mapdl.parameters[f"DISP_REALZ{i}"][0]
                    ) + np.insert(
                        mapdl.parameters[f"DISP_IMAGZ{i}"],
                        0,
                        mapdl.parameters[f"DISP_IMAGZ{i}"][0]
                    )*1j
                )
        nodal_solution = np.array(nodal_solution)
    else:
        raise Exception("Harmonic analysis without mode superposition not"
                        "implemented yet.")
    if magnitude != "disp":
        freq_exponent = 2 if magnitude == "acc" else 1
        omega = np.tile(
            freq_vector,
            (len(node_id_list) * len(response_directions), 1)
        )
        omega = omega * (2*np.pi*1j) ** freq_exponent
        nodal_solution = nodal_solution * omega
    return pymodal.FRF(
        frf=nodal_solution.conj().transpose(),
        min_freq=frequency_range[0],
        max_freq=frequency_range[1],
        modal_frequencies=modal_analysis["modal_frequencies"]
    )
