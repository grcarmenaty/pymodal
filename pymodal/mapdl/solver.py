import ansys.mapdl
import numpy as np
import pathlib
from scipy.spatial.distance import cdist
import pymodal
import time


def modal_analysis(mapdl, frequency_range, master_coords=None, dof='xyz',
                   mode_limit=999):
    mapdl.finish()
    mapdl.run('/SOL')
    mapdl.antype(2)
    mapdl.eqslv('SPAR')
    mapdl.mxpand(mode_limit, '', '', 'YES')
    mapdl.lumpm(0)
    mapdl.pstres(0)
    mapdl.modopt('LANB', mode_limit, frequency_range[0], frequency_range[1])
    mapdl.solve()
    mapdl.finish()
    path = pathlib.Path(mapdl.inquire('DIRECTORY'))
    jobname = mapdl.inquire('JOBNAME')
    result = mapdl.result
    mm = mapdl.math
    node_coordinates = result.geometry['nodes'][:, 0:3]
    node_coordinates = np.core.records.fromarrays(node_coordinates.transpose(),
                                                  names='x, y, z')
    node_order = np.argsort(node_coordinates, order=['x', 'y', 'z'])
    node_list = np.hstack((result.geometry['nnum'][node_order].reshape(-1, 1),
                           result.geometry['nodes'][node_order, 0:3]))
    modal_frequencies = result.time_values 
    result_dict = {'modal_frequencies': modal_frequencies}
    if master_coords is not None:
        master_nodes = []
        row_col_list = []
        k = mm.stiff(str(path / f'{jobname}.full')).asarray()
        m = mm.mass(str(path / f'{jobname}.full')).asarray()
        # After the change in pyansys, I'm unsure node_list is a correct
        # substitute for the mass and stiffness reference dof_ref.
        for i in range(master_coords.shape[0]):
            closest_node = cdist(np.asarray([master_coords[i]]),
                                node_list[:, 1:4])
            master_nodes.append(int(node_list[
                int(np.argmin(closest_node)), 0
            ]))
            if 'x' in dof:
                row_col_list.append(
                    list(node_list[:, 0]).index(master_nodes[i])
                )
            if 'y' in dof:
                row_col_list.append(
                    list(node_list[:, 0]).index(master_nodes[i]) + 1
                )
            if 'z' in dof:
                row_col_list.append(
                    list(node_list[:, 0]).index(master_nodes[i]) + 2
                )
        master_nodes = np.array(master_nodes)
        row_col = np.array(row_col_list)
        m = m[:, row_col].todense()
        m = m[row_col, :]
        k = k[:, row_col].todense()
        k = k[row_col, :]
        result_dict = {'modal_frequencies': modal_frequencies,
                       'mass_matrix': m,
                       'stiffness_matrix': k}
        try:
            dof_selector = []
            if 'x' in dof:
                dof_selector.append(0)
            if 'y' in dof:
                dof_selector.append(1)
            if 'z' in dof:
                dof_selector.append(2)
            mode_shapes = []
            for i in range(modal_frequencies.shape[0]):
                mode_shapes.append(
                    result.nodal_solution(i)[1][master_nodes - 1]
                )
            mode_shapes = np.dstack(mode_shapes)
            mode_shapes = mode_shapes[:, dof_selector, :]
            result_dict = {'modal_frequencies': modal_frequencies,
                           'mode_shapes': mode_shapes,
                           'mass_matrix': m,
                           'stiffness_matrix': k}
        except Exception as __:
            mode_shapes = 'Modal shapes unavailable'
    return result_dict


def get_stiffness(mapdl, coords, force_vector):
    node_list = pymodal.mapdl.get_node_list(mapdl, stage='/SOL')
    excitation_node = cdist(np.asarray([coords]),
                            node_list[:, 1:4])
    excitation_node = node_list[int(np.argmin(excitation_node)), 0]
    node_id_list = []
    mapdl.f(excitation_node, 'FX',
          force_vector[0])
    mapdl.f(excitation_node, 'FY',
          force_vector[1])
    mapdl.f(excitation_node, 'FZ',
          force_vector[2])
    mapdl.run('/SOL')
    mapdl.antype(0)
    mapdl.solve()
    mapdl.finish()
    path = pathlib.Path(mapdl.inquire('DIRECTORY'))
    jobname = mapdl.inquire('JOBNAME')
    result = mapdl.result
    node_coordinates = result.geometry['nodes'][:, 0:3]
    node_coordinates = np.core.records.fromarrays(node_coordinates.transpose(),
                                                  names='x, y, z')
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

    node_list = pymodal.mapdl.get_node_list(mapdl, stage='/SOL')
    excitation_node = cdist(np.asarray([excitation_coordinates]),
                            node_list[:, 1:4])
    excitation_node = node_list[int(np.argmin(excitation_node)), 0]
    node_id_list = []
    for row in response_coordinates:
        closest_node = cdist(np.asarray([row]), node_list[:, 1:4])
        node_id_list.append(int(node_list[int(np.argmin(closest_node)), 0]))
    mapdl.f(excitation_node, 'FX',
          excitation_vector[0])
    mapdl.f(excitation_node, 'FY',
          excitation_vector[1])
    mapdl.f(excitation_node, 'FZ',
          excitation_vector[2])
    if mode_superposition:
        pymodal.mapdl.modal_analysis(mapdl, frequency_range)
    mapdl.run('/SOL')
    mapdl.antype(3)
    if mode_superposition:
        mapdl.hropt('MSUP')
    mapdl.harfrq(frequency_range[0], frequency_range[1])
    mapdl.nsubst(N)
    mapdl.kbc(0)
    mapdl.alphad(damping[0])
    mapdl.betad(damping[1])
    mapdl.solve()
    mapdl.finish()
    path = pathlib.Path(mapdl.inquire('DIRECTORY')).absolute()
    if mode_superposition:
        mapdl.run('/POST26')
        mapdl.file(mapdl.inquire('JOBNAME'), 'RFRQ')
        mapdl.run('/UI,COLL,1  ')
        mapdl.numvar(200)
        mapdl.nsol(191, 1, 'UX')
        mapdl.store('MERGE')
        mapdl.filldata(191, '', '', '', 1, 1)
        mapdl.realvar(191, 191)
        mapdl.run('*DEL,MAX_PARAM')
        mapdl.run(f'*DIM,FREQ1,ARRAY,{N}')
        mapdl.vget('FREQ1(1)', 1)
        with mapdl.non_interactive:
            mapdl.run('*CFOPEN,freq,txt, ,')
            mapdl.run('*VWRITE,FREQ1(1)')
            mapdl.run('%12.5E')
            mapdl.run('*CFCLOSE')
        for i, node_id in enumerate(node_id_list):
            mapdl.run('/POST26')
            if 'X' in response_directions:
                mapdl.nsol(2, node_id, 'U', 'X')
                mapdl.store('MERGE')
                mapdl.run(f'*DIM,DISP_REALX{i},ARRAY,{N}')
                mapdl.run(f'*DIM,DISP_IMAGX{i},ARRAY,{N}')
                mapdl.vget(f'DISP_REALX{i}(1)',2,'',0)
                mapdl.vget(f'DISP_IMAGX{i}(1)',2,'',1)
                with mapdl.non_interactive:
                    mapdl.run(f'*CFOPEN,frfx{i},txt, ,')
                    mapdl.run(f'*VWRITE,DISP_REALX{i}(1),DISP_IMAGX{i}(1)')
                    mapdl.run('%12.5E %12.5E')
                    mapdl.run('*CFCLOSE')  
            if 'Y' in response_directions:
                mapdl.nsol(2, node_id, 'U', 'Y')
                mapdl.store('MERGE')
                mapdl.run(f'*DIM,DISP_REALY{i},ARRAY,{N}')
                mapdl.run(f'*DIM,DISP_IMAGY{i},ARRAY,{N}')
                mapdl.vget(f'DISP_REALY{i}(1)',2,'',0)
                mapdl.vget(f'DISP_IMAGY{i}(1)',2,'',1)
                with mapdl.non_interactive:
                    mapdl.run(f'*CFOPEN,frfy{i},txt, ,')
                    mapdl.run(f'*VWRITE,DISP_REALY{i}(1),DISP_IMAGY{i}(1)')
                    mapdl.run('%12.5E %12.5E')
                    mapdl.run('*CFCLOSE')
            if 'Z' in response_directions:
                mapdl.nsol(2, node_id, 'U', 'Z')
                mapdl.store('MERGE')
                mapdl.run(f'*DIM,DISP_REALZ{i},ARRAY,{N}')
                mapdl.run(f'*DIM,DISP_IMAGZ{i},ARRAY,{N}')
                mapdl.vget(f'DISP_REALZ{i}(1)',2,'',0)
                mapdl.vget(f'DISP_IMAGZ{i}(1)',2,'',1)
                with mapdl.non_interactive:
                    mapdl.run(f'*CFOPEN,frfz{i},txt, ,')
                    mapdl.run(f'*VWRITE,DISP_REALZ{i}(1),DISP_IMAGZ{i}(1)')
                    mapdl.run('%12.5E %12.5E')
                    mapdl.run('*CFCLOSE')
        nodal_solution = []
        for i in range(len(node_id_list)):
            displacement = [0+0j]
            if 'X' in response_directions:
                with open(path/f'frfx{i}.txt','r+') as f: 
                    for line in f:
                        line = line.replace('  ', '+')
                        line = line.replace(' ', '')
                        line = line.replace('\n', 'j')
                        displacement.append(complex(line))
                displacement = np.asarray(displacement)
                nodal_solution.append(displacement)
            displacement = [0+0j]
            if 'Y' in response_directions:
                with open(path/f'frfy{i}.txt','r+') as f:
                    for line in f:
                        line = line.replace('  ', '+')
                        line = line.replace(' ', '')
                        line = line.replace('\n', 'j')
                        displacement.append(complex(line))
                displacement = np.asarray(displacement)
                nodal_solution.append(displacement)
            displacement = [0+0j]
            if 'Z' in response_directions:
                with open(path/f'frfz{i}.txt','r+') as f: 
                    for line in f:
                        line = line.replace('  ', '+')
                        line = line.replace(' ', '')
                        line = line.replace('\n', 'j')
                        displacement.append(complex(line))
                displacement = np.asarray(displacement)
                nodal_solution.append(displacement)
        nodal_solution = np.stack(nodal_solution, axis=0)
        with open(path/'freq.txt','r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f' {frequency_range[0]}'.rstrip('\r\n') + '\n' + content)
        freq_vector = np.genfromtxt(path/'freq.txt')
    else:
        result = mapdl.result
        freq_vector = np.hstack((
            np.array([frequency_range[0], frequency_range[0]]),
            result.time_values
        ))
        nodal_solution = []
        for node_id in node_id_list:
            displacement = [0, 0]
            if 'X' in response_directions:
                for i in range(2*N):
                    displacement.append(
                        result.nodal_solution(i)[1][node_id-1, 0]
                    )
                nodal_solution.append(np.asarray(displacement))
            displacement = [0, 0]
            if 'Y' in response_directions:
                for i in range(2*N):
                    displacement.append(
                        result.nodal_solution(i)[1][node_id-1, 1]
                    )
                nodal_solution.append(np.asarray(displacement))
            displacement = [0, 0]
            if 'Z' in response_directions:
                for i in range(2*N):
                    displacement.append(
                        result.nodal_solution(i)[1][node_id-1, 2]
                    )
                nodal_solution.append(np.asarray(displacement))
        nodal_solution = np.stack(nodal_solution, axis=0)
        nodal_solution = nodal_solution[:, 0::2] + nodal_solution[:, 1::2]*1j
        freq_vector = freq_vector[0::2]
    if magnitude != 'disp':
        freq_exponent = 2 if magnitude == 'acc' else 1
        omega = np.tile(
            freq_vector,
            (len(node_id_list) * len(response_directions), 1)
        )
        omega = omega * (2*np.pi*1j) ** freq_exponent
        nodal_solution = nodal_solution * omega
    return pymodal.FRF(
        frf=nodal_solution.conj().transpose(),
        min_freq=frequency_range[0],
        max_freq=frequency_range[1]
    )
