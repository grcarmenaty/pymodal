import pyansys
import numpy as np
import pathlib
from scipy.spatial.distance import cdist
import pymodal
import time


def modal_analysis(ansys, frequency_range, master_coords=None, dof='xyz',
                   mode_limit=999):
    ansys.finish()
    ansys.run('/SOL')
    ansys.antype(2)
    ansys.eqslv('SPAR')
    ansys.mxpand(mode_limit, '', '', 'YES')
    ansys.lumpm(0)
    ansys.pstres(0)
    ansys.modopt('LANB', mode_limit, frequency_range[0], frequency_range[1])
    ansys.solve()
    ansys.finish()
    path = pathlib.Path(ansys.inquire('DIRECTORY'))
    jobname = ansys.inquire('JOBNAME')
    result = pyansys.read_binary(str(path/ f'{jobname}.rst'))
    full = pyansys.read_binary(str(path / f'{jobname}.full'))
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
        dof_ref, k, m = full.load_km(sort=True)
        for i in range(master_coords.shape[0]):
            closest_node = cdist(np.asarray([master_coords[i]]),
                                node_list[:, 1:4])
            master_nodes.append(int(node_list[
                int(np.argmin(closest_node)), 0
            ]))
            if 'x' in dof:
                row_col_list.append(
                    list(dof_ref[:, 0]).index(master_nodes[i])
                )
            if 'y' in dof:
                row_col_list.append(
                    list(dof_ref[:, 0]).index(master_nodes[i]) + 1
                )
            if 'z' in dof:
                row_col_list.append(
                    list(dof_ref[:, 0]).index(master_nodes[i]) + 2
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


def get_stiffness(ansys, coords, force_vector):
    node_list = pymodal.ansys.get_node_list(ansys, stage='/SOL')
    excitation_node = cdist(np.asarray([coords]),
                            node_list[:, 1:4])
    excitation_node = node_list[int(np.argmin(excitation_node)), 0]
    node_id_list = []
    ansys.f(excitation_node, 'FX',
          force_vector[0])
    ansys.f(excitation_node, 'FY',
          force_vector[1])
    ansys.f(excitation_node, 'FZ',
          force_vector[2])
    ansys.run('/SOL')
    ansys.antype(0)
    ansys.solve()
    ansys.finish()
    path = pathlib.Path(ansys.inquire('DIRECTORY'))
    jobname = ansys.inquire('JOBNAME')
    result = pyansys.read_binary(str(path/ f'{jobname}.rst'))
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
    ansys,
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

    node_list = pymodal.ansys.get_node_list(ansys, stage='/SOL')
    excitation_node = cdist(np.asarray([excitation_coordinates]),
                            node_list[:, 1:4])
    excitation_node = node_list[int(np.argmin(excitation_node)), 0]
    node_id_list = []
    for row in response_coordinates:
        closest_node = cdist(np.asarray([row]), node_list[:, 1:4])
        node_id_list.append(int(node_list[int(np.argmin(closest_node)), 0]))
    ansys.f(excitation_node, 'FX',
          excitation_vector[0])
    ansys.f(excitation_node, 'FY',
          excitation_vector[1])
    ansys.f(excitation_node, 'FZ',
          excitation_vector[2])
    if mode_superposition:
        pymodal.ansys.modal_analysis(ansys, frequency_range)
    ansys.run('/SOL')
    ansys.antype(3)
    if mode_superposition:
        ansys.hropt('MSUP')
    ansys.harfrq(frequency_range[0], frequency_range[1])
    ansys.nsubst(N)
    ansys.kbc(0)
    ansys.alphad(damping[0])
    ansys.betad(damping[1])
    ansys.solve()
    ansys.finish()
    path = pathlib.Path(ansys.inquire('DIRECTORY')).absolute()
    if mode_superposition:
        ansys.run('/POST26')
        ansys.file(ansys.inquire('JOBNAME'), 'RFRQ')
        ansys.run('/UI,COLL,1  ')
        ansys.numvar(200)
        ansys.nsol(191, 1, 'UX')
        ansys.store('MERGE')
        ansys.filldata(191, '', '', '', 1, 1)
        ansys.realvar(191, 191)
        ansys.run('*DEL,MAX_PARAM')
        ansys.run(f'*DIM,FREQ1,ARRAY,{N}')
        ansys.vget('FREQ1(1)', 1)
        with ansys.non_interactive:
            ansys.run('*CFOPEN,freq,txt, ,')
            ansys.run('*VWRITE,FREQ1(1)')
            ansys.run('%12.5E')
            ansys.run('*CFCLOSE')
        for i, node_id in enumerate(node_id_list):
            ansys.run('/POST26')
            if 'X' in response_directions:
                ansys.nsol(2, node_id, 'U', 'X')
                ansys.store('MERGE')
                ansys.run(f'*DIM,DISP_REALX{i},ARRAY,{N}')
                ansys.run(f'*DIM,DISP_IMAGX{i},ARRAY,{N}')
                ansys.vget(f'DISP_REALX{i}(1)',2,'',0)
                ansys.vget(f'DISP_IMAGX{i}(1)',2,'',1)
                with ansys.non_interactive:
                    ansys.run(f'*CFOPEN,frfx{i},txt, ,')
                    ansys.run(f'*VWRITE,DISP_REALX{i}(1),DISP_IMAGX{i}(1)')
                    ansys.run('%12.5E %12.5E')
                    ansys.run('*CFCLOSE')  
            if 'Y' in response_directions:
                ansys.nsol(2, node_id, 'U', 'Y')
                ansys.store('MERGE')
                ansys.run(f'*DIM,DISP_REALY{i},ARRAY,{N}')
                ansys.run(f'*DIM,DISP_IMAGY{i},ARRAY,{N}')
                ansys.vget(f'DISP_REALY{i}(1)',2,'',0)
                ansys.vget(f'DISP_IMAGY{i}(1)',2,'',1)
                with ansys.non_interactive:
                    ansys.run(f'*CFOPEN,frfy{i},txt, ,')
                    ansys.run(f'*VWRITE,DISP_REALY{i}(1),DISP_IMAGY{i}(1)')
                    ansys.run('%12.5E %12.5E')
                    ansys.run('*CFCLOSE')
            if 'Z' in response_directions:
                ansys.nsol(2, node_id, 'U', 'Z')
                ansys.store('MERGE')
                ansys.run(f'*DIM,DISP_REALZ{i},ARRAY,{N}')
                ansys.run(f'*DIM,DISP_IMAGZ{i},ARRAY,{N}')
                ansys.vget(f'DISP_REALZ{i}(1)',2,'',0)
                ansys.vget(f'DISP_IMAGZ{i}(1)',2,'',1)
                with ansys.non_interactive:
                    ansys.run(f'*CFOPEN,frfz{i},txt, ,')
                    ansys.run(f'*VWRITE,DISP_REALZ{i}(1),DISP_IMAGZ{i}(1)')
                    ansys.run('%12.5E %12.5E')
                    ansys.run('*CFCLOSE')
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
        result = pyansys.read_binary(
            str(path/(ansys.inquire('JOBNAME') + '.rst'))
        )
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
