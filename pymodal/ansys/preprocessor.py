import pyansys
import numpy as np
import pathlib
from scipy.spatial.distance import cdist
import pymodal


def _get_max_param_id(ansys, param):

    ansys.finish()
    ansys.run('/PREP7')
    ansys.run('*DEL,MAX_PARAM')
    ansys.get('MAX_PARAM', param, 0, 'NUM', 'MAX')
    ansys.finish()
    ansys.load_parameters()
    try:
        return ansys.parameters['MAX_PARAM']
    except Exception as __:
        return 0


def set_linear_elastic(ansys, elastic_modulus, Poisson_ratio, density):

    mat_id = _get_max_param_id(ansys, 'MAT') + 1
    ansys.run('/PREP7')
    ansys.mp('EX', mat_id, elastic_modulus)
    ansys.mp('PRXY', mat_id, Poisson_ratio)
    ansys.mp('DENS', mat_id, density)
    ansys.finish()
    return mat_id


def create_line(ansys, start_coordinates, end_coordinates):

    return_data = {}
    return_data['start_kp'] = _get_max_param_id(ansys, 'KP') + 1
    return_data['end_kp'] = return_data['start_kp'] + 1
    ansys.run('/PREP7')
    ansys.k(
        return_data['start_kp'],
        start_coordinates[0],
        start_coordinates[1],
        start_coordinates[2]
    )
    ansys.k(
        return_data['end_kp'],
        end_coordinates[0],
        end_coordinates[1],
        end_coordinates[2]
    )
    ansys.l(return_data['start_kp'], return_data['end_kp'])
    ansys.get('CURRENT_L', 'LINE', 0, 'NUM', 'MAX')
    ansys.load_parameters()
    return_data['line_id'] = ansys.parameters['CURRENT_L']
    ansys.run('CURRENT_L=')
    ansys.finish()
    return return_data


def create_area(ansys, coords):

    coords = np.array(coords)
    if coords.shape[0] > 18:
        raise Exception("Areas can only have up to 18 KP.")
    return_data = {}
    return_data['start_kp'] = _get_max_param_id(ansys, 'KP') + 1
    ansys.run('/PREP7')
    kp_id = return_data['start_kp']
    kp_list = []
    for coord in coords:
        ansys.k(kp_id, coord[0], coord[1], coord[2])
        kp_list.append(kp_id)
        kp_id = kp_id + 1
    return_data['end_kp'] = kp_list[-1]
    for _ in range(len(kp_list), 18):
        kp_list.append('')
    ansys.a(kp_list[0], kp_list[1], kp_list[2], kp_list[3], kp_list[4],
            kp_list[5], kp_list[6], kp_list[7], kp_list[8], kp_list[9],
            kp_list[10], kp_list[11], kp_list[12], kp_list[13], kp_list[14],
            kp_list[15], kp_list[16], kp_list[17])
    ansys.get('CURRENT_A', 'AREA', 0, 'NUM', 'MAX')
    ansys.load_parameters()
    return_data['area_id'] = ansys.parameters['CURRENT_A']
    ansys.run('CURRENT_A=')
    ansys.finish()
    return return_data


def create_prism(ansys, x_origin, y_origin, width, depth, height):

    return_data = {}
    ansys.run('/PREP7')
    ansys.blc4(x_origin, y_origin, width, depth, height)
    ansys.get('CURRENT_V', 'VOLU', 0, 'NUM', 'MAX')
    ansys.load_parameters()
    return_data['volume_id'] = ansys.parameters['CURRENT_V']
    ansys.run('CURRENT_V=')
    ansys.finish()
    return return_data


def create_cylinder(ansys, x_center, y_center, radius, height):

    return_data = {}
    ansys.run('/PREP7')
    ansys.cyl4(x_center, y_center, radius, '', '', '', height)
    ansys.get('CURRENT_V', 'VOLU', 0, 'NUM', 'MAX')
    ansys.load_parameters()
    return_data['volume_id'] = ansys.parameters['CURRENT_V']
    ansys.run('CURRENT_V=')
    ansys.finish()
    return return_data


def create_extruded_volume(ansys, coords, thickness):

    return_data = {}
    area_id = create_area(ansys, coords)
    ansys.run('/PREP7')
    ansys.voffst(area_id['area_id'], thickness)
    ansys.get('CURRENT_V', 'VOLU', 0, 'NUM', 'MAX')
    ansys.load_parameters()
    return_data['volume_id'] = ansys.parameters['CURRENT_V']
    ansys.run('CURRENT_V=')
    ansys.finish()
    return return_data


def set_beam3(ansys, area, inertia, height):

    return_data = {}
    return_data['etype_id'] = _get_max_param_id(ansys, 'ETYPE') + 1
    return_data['sectype_id'] = _get_max_param_id(ansys, 'RCON') + 1
    ansys.run('/PREP7')
    ansys.et(return_data['etype_id'], 'BEAM3')
    ansys.r(return_data['sectype_id'], area, inertia, height)
    ansys.finish()
    return return_data


def set_solid186(ansys):

    return_data = {}
    return_data['etype_id'] = _get_max_param_id(ansys, 'ETYPE') + 1
    ansys.run('/PREP7')
    ansys.et(return_data['etype_id'], 'SOLID186')
    ansys.finish()
    return return_data


def set_shell181(ansys, thickness, material, angle=0, int_points=3,
                 offset='MID'):

    return_data = {}
    return_data['etype_id'] = _get_max_param_id(ansys, 'ETYPE') + 1
    return_data['sec_id'] = _get_max_param_id(ansys, 'GENS') + 1
    ansys.run('/PREP7')
    ansys.et(return_data['etype_id'], 'SHELL181')
    ansys.sectype(return_data['sec_id'], 'SHELL')
    ansys.secdata(thickness, material, angle, int_points)
    ansys.secoffset(offset)
    ansys.finish()
    return return_data


def _get_closest_node(ansys, coordinates):

    node_list = get_node_list(ansys)
    node_distance = cdist(np.array([coordinates]),
                          node_list[:, 1:4])
    return node_list[int(np.argmin(node_distance)), 0]


def get_node_list(ansys, stage='/SOL'):

    ansys.finish()
    ansys.run('/SOL')
    ansys.antype(2)
    ansys.modopt('LANB', 1)
    ansys.solve()
    ansys.finish()
    path = pathlib.Path(ansys.inquire('DIRECTORY'))
    result = pyansys.read_binary(str(path/(ansys.inquire('JOBNAME') + '.rst')))
    node_coordinates = result.geometry['nodes'][:, 0:3]
    node_coordinates = np.core.records.fromarrays(node_coordinates.transpose(), 
                                                  names='x, y, z')
    node_order = np.argsort(node_coordinates, order=['x', 'y', 'z'])
    node_list = np.hstack(
        (result.geometry['nnum'][node_order].reshape(-1, 1),
         result.geometry['nodes'][node_order, 0:3])
    )
    ansys.run(stage)
    return node_list


def select_nodes(ansys, x_lim, y_lim, z_lim):

    ansys.run('/SOL')
    # ansys.run('*DEL,MAX_X')
    # ansys.get('MAX_X', 'NODE', 0, 'MXLOC', 'X')
    ansys.run('*DEL,MAX_Y')
    ansys.get('MAX_Y', 'NODE', 0, 'MXLOC', 'Y')
    ansys.run('*DEL,MAX_Z')
    ansys.get('MAX_Z', 'NODE', 0, 'MXLOC', 'Z')
    # ansys.run('*DEL,MIN_X')
    # ansys.get('MIN_X', 'NODE', 0, 'MNLOC', 'X')
    ansys.run('*DEL,MIN_Y')
    ansys.get('MIN_Y', 'NODE', 0, 'MNLOC', 'Y')
    ansys.run('*DEL,MIN_Z')
    ansys.get('MIN_Z', 'NODE', 0, 'MNLOC', 'Z')
    ansys.load_parameters()
    # max_x = ansys.parameters['MAX_X']
    max_y = ansys.parameters['MAX_Y']
    max_z = ansys.parameters['MAX_Z']
    # min_x = ansys.parameters['MIN_X']
    min_y = ansys.parameters['MIN_Y']
    min_z = ansys.parameters['MIN_Z']
    ansys.nsel('S', 'LOC', 'X', x_lim[0], x_lim[1])
    ansys.nsel('U', 'LOC', 'Y', min_y - 1, y_lim[0])
    ansys.nsel('U', 'LOC', 'Y', y_lim[1], max_y + 1)
    ansys.nsel('U', 'LOC', 'Z', min_z - 1, z_lim[0])
    ansys.nsel('U', 'LOC', 'Z', z_lim[1], max_z + 1)
    ansys.finish()


def displacement_bc(ansys, x_lim, y_lim, z_lim, x=True, y=True, z=True,
                    rotx=True, roty=True, rotz=True):

    select_nodes(ansys, x_lim, y_lim, z_lim)
    ansys.run('/SOL')
    if x:
        ansys.d('ALL', 'UX', 0)
    if y:
        ansys.d('ALL', 'UY', 0)
    if z:
        ansys.d('ALL', 'UZ', 0)
    if rotx:
        ansys.d('ALL', 'ROTX', 0)
    if roty:
        ansys.d('ALL', 'ROTY', 0)
    if rotz:
        ansys.d('ALL', 'ROTZ', 0)
    ansys.allsel()
    ansys.finish()


def elastic_support(ansys, x_lim, y_lim, z_lim, normal_stiffness,
                    tangential_stiffness=None, pinball_radius=0):

    ansys.run(f'ARG1={normal_stiffness}')
    tangential_stiffness = (normal_stiffness if tangential_stiffness is None
                            else tangential_stiffness)
    ansys.run(f'ARG2={tangential_stiffness}')
    ansys.run(f'ARG3={pinball_radius}')
    select_nodes(ansys, x_lim, y_lim, z_lim)
    ansys.cm('Elastic_Here', 'NODE')
    ansys.allsel()
    # Source for the following code snippet: https://www.simutechgroup.com/tips-and-tricks/fea-articles/143-a-normal-and-tangential-elastic-foundation-in-workbench-mechanical
    ansys.run('/PREP7')
    with ansys.non_interactive:
        ansys.run('*if,ARG1,LE,0,then')
        ansys.run('*MSG,ERROR')
        ansys.run(('ARG1 for Normal Stiffness on XYZ Elastic Foundation must'
                   ' be positive'))
        ansys.run('/EOF')
        ansys.run('*return,-1')
        ansys.run('*endif')
        ansys.run('*if,ARG2,LE,0,then')
        ansys.run('ARG2=ARG1')
        ansys.run('/COM,######## ARG2 was made equal to ARG1 ########')
        ansys.run('*endif')
        ansys.run('fini')
        ansys.run('/prep7')
        ansys.run('*get,nodemax,NODE,,NUM,MAX        ')
        ansys.cmsel('s', 'Elastic_Here')
        ansys.esln()
        ansys.esel('u', 'ename', '', 151, 154)
        ansys.esel('u', 'ename', '', 169, 180)
        ansys.esel('u', 'ename', '', 188, 189)
        ansys.run('*get,maxtype,ETYP,,NUM,MAX        ')
        ansys.run('*get, maxmat,MAT,,NUM,MAX         ')
        ansys.run('*get,maxreal,RCON,,NUM,MAX        ')
        ansys.run('*if,maxtype,gt,maxmat,then')
        ansys.run('maxmat=maxtype')
        ansys.run('*else')
        ansys.run('maxtype=maxmat')
        ansys.run('*endif')
        ansys.run('*if,maxreal,gt,maxtype,then')
        ansys.run('maxtype=maxreal')
        ansys.run('maxmat=maxreal')
        ansys.run('*else')
        ansys.run('maxreal=maxtype')
        ansys.run('*endif')
        ansys.et('maxtype+1', 'CONTA174', '', 1, '', 0, 3)
        ansys.keyopt('maxtype+1', 9, 1)
        ansys.keyopt('maxtype+1', 12, 5)
        ansys.et('maxtype+2', 'TARGE170', '', 1)  #Constraints by user
        ansys.r('maxreal+1', 0, 0, '-ARG1', '', '', '-abs(ARG3)')
        ansys.rmodif('maxreal+1', 12, '-ARG2')
        ansys.type('maxtype+1')
        ansys.real('maxreal+1')
        ansys.mat('maxmat+1')
        ansys.esurf()
        ansys.run('*get,current_nodemin,node,,num,min')
        ansys.esln('r', 1)
        ansys.esel('r', 'ename', '', 174)
        ansys.esel('r', 'real', '', 'maxreal+1')
        ansys.ngen(2, '(nodemax-current_nodemin)+1', 'ALL', '', '', 0, 0, 0)
        ansys.egen(2, '(nodemax-current_nodemin)+1', 'ALL', '', '', 0, 1, 0)
        ansys.esel('r', 'type', '', 'maxtype+2')
        ansys.ensym(0, '', 0, 'ALL')
        ansys.nsle()
        ansys.d('all', 'all')
        ansys.allsel()
    ansys.finish()


def bonded_surface_contact(ansys, area_1_id, area_2_id):

    mat_id = set_linear_elastic(ansys, 1e12, 0.3, 0)
    rc_max = _get_max_param_id(ansys, 'RCON')
    et_max = _get_max_param_id(ansys, 'ETYPE')
    ansys.run('/PREP7')
    ansys.mp("MU", mat_id, "")
    ansys.mp("EMIS", mat_id, 7.88860905221e-031)
    ansys.cm("_NODECM", "NODE")
    ansys.cm("_ELEMCM", "ELEM")
    ansys.cm("_KPCM", "KP")
    ansys.cm("_LINECM", "LINE")
    ansys.cm("_AREACM", "AREA")
    ansys.cm("_VOLUCM", "VOLU")
    ansys.mat(mat_id)
    ansys.r(rc_max + 1)
    ansys.real(rc_max + 1)
    ansys.et(et_max + 1, 170)
    ansys.et(et_max + 2, 174)
    ansys.r(rc_max + 1, "", "", 1.0, 0.1, 0, "")
    ansys.rmore("", "", 1.0E20, 0.0, 1.0, "")
    ansys.rmore(0.0, 0, 1.0, "", 1.0, 0.5)
    ansys.rmore(0, 1.0, 1.0, 0.0, "", 1.0)
    ansys.rmore("", "", "", "", "", 1.0)
    ansys.keyopt(et_max + 2, 4, 0)
    ansys.keyopt(et_max + 2, 5, 0)
    ansys.keyopt(et_max + 2, 7, 0)
    ansys.keyopt(et_max + 2, 8, 0)
    ansys.keyopt(et_max + 2, 9, 0)
    ansys.keyopt(et_max + 2, 10, 0)
    ansys.keyopt(et_max + 2, 11, 0)
    ansys.keyopt(et_max + 2, 12, 5)
    ansys.keyopt(et_max + 2, 14, 0)
    ansys.keyopt(et_max + 2, 18, 0)
    ansys.keyopt(et_max + 2, 2, 0)
    ansys.keyopt(et_max + 1, 5, 0)
    ansys.asel("S", "", "", area_1_id)
    ansys.cm("_TARGET", "AREA")
    ansys.type(et_max + 1)
    ansys.nsla("S", 1)
    ansys.esln("S", 0)
    ansys.esll("U")
    ansys.esel("U", "ENAME", "", 188, 189)
    ansys.nsle("A", "CT2")
    ansys.esurf()
    ansys.cmsel("S", "_ELEMCM")
    ansys.asel("S", "", "", area_2_id)
    ansys.cm("_CONTACT", "AREA")
    ansys.type(et_max + 2)
    ansys.nsla("S", 1)
    ansys.esln("S", 0)
    ansys.nsle("A", "CT2")
    ansys.esurf()
    ansys.r(rc_max + 2)
    ansys.real(rc_max + 2)
    ansys.et(et_max + 3, 170)
    ansys.et(et_max + 4, 174)
    ansys.r(rc_max + 2, "", "", 1.0, 0.1, 0, "")
    ansys.rmore("", "", 1.0E20, 0.0, 1.0, "")
    ansys.rmore(0.0, 0, 1.0, "", 1.0, 0.5)
    ansys.rmore(0, 1.0, 1.0, 0.0, "", 1.0)
    ansys.rmore("", "", "", "", "", 1.0)
    ansys.keyopt(et_max + 4, 4, 0)
    ansys.keyopt(et_max + 4, 5, 0)
    ansys.keyopt(et_max + 4, 7, 0)
    ansys.keyopt(et_max + 4, 8, 0)
    ansys.keyopt(et_max + 4, 9, 0)
    ansys.keyopt(et_max + 4, 10, 0)
    ansys.keyopt(et_max + 4, 11, 0)
    ansys.keyopt(et_max + 4, 12, 5)
    ansys.keyopt(et_max + 4, 14, 0)
    ansys.keyopt(et_max + 4, 18, 0)
    ansys.keyopt(et_max + 4, 2, 0)
    ansys.keyopt(et_max + 3, 1, 0)
    ansys.keyopt(et_max + 3, 3, 0)
    ansys.keyopt(et_max + 3, 5, 0)
    ansys.type(et_max + 3)
    ansys.esel("S", "TYPE", "", et_max + 2)
    ansys.nsle("S")
    ansys.esln("S", 0)
    ansys.esurf()
    ansys.type(et_max + 4)
    ansys.esel("S", "TYPE", "", et_max + 1)
    ansys.nsle("S")
    ansys.esln("S", 0)
    ansys.esurf()
    ansys.allsel()
    ansys.cmsel("A", "_NODECM")
    ansys.run("CMDEL,_NODECM")
    ansys.cmsel("A", "_ELEMCM")
    ansys.run("CMDEL,_ELEMCM")
    ansys.cmsel("S", "_KPCM")
    ansys.run("CMDEL,_KPCM")
    ansys.cmsel("S", "_LINECM")
    ansys.run("CMDEL,_LINECM")
    ansys.cmsel("S", "_AREACM")
    ansys.run("CMDEL,_AREACM")
    ansys.cmsel("S", "_VOLUCM")
    ansys.run("CMDEL,_VOLUCM")
    ansys.run("CMDEL,_TARGET")
    ansys.run("CMDEL,_CONTACT")
    ansys.finish()


def mass_to_node(ansys, node_id, mass_value):

    ansys.run('/PREP7')
    mat_id = _get_max_param_id(ansys, 'ETYPE') + 1
    ansys.et(80, 'MASS21')
    ansys.r(80, mass_value, mass_value, mass_value)
    ansys.type(80)
    ansys.real(80)
    ansys.e(node_id)
    ansys.finish()