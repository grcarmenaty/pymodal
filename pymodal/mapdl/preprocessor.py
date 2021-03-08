import ansys
import numpy as np
import pandas as pd
import pathlib
from scipy.spatial.distance import cdist
import pymodal


def _get_max_param_id(mapdl, param):

    mapdl.finish()
    mapdl.run('/PREP7')
    mapdl.run('*DEL,MAX_PARAM')
    mapdl.get('MAX_PARAM', param, 0, 'NUM', 'MAX')
    mapdl.finish()
    try:
        return mapdl.parameters['MAX_PARAM']
    except Exception as __:
        return 0


def set_linear_elastic(mapdl, elastic_modulus, Poisson_ratio, density):

    mat_id = _get_max_param_id(mapdl, 'MAT') + 1
    mapdl.run('/PREP7')
    mapdl.mp('EX', mat_id, elastic_modulus)
    mapdl.mp('PRXY', mat_id, Poisson_ratio)
    mapdl.mp('DENS', mat_id, density)
    mapdl.finish()
    return mat_id


def create_line(mapdl, start_coordinates, end_coordinates):

    return_data = {}
    return_data['start_kp'] = _get_max_param_id(mapdl, 'KP') + 1
    return_data['end_kp'] = return_data['start_kp'] + 1
    mapdl.run('/PREP7')
    mapdl.k(
        return_data['start_kp'],
        start_coordinates[0],
        start_coordinates[1],
        start_coordinates[2]
    )
    mapdl.k(
        return_data['end_kp'],
        end_coordinates[0],
        end_coordinates[1],
        end_coordinates[2]
    )
    mapdl.l(return_data['start_kp'], return_data['end_kp'])
    mapdl.get('CURRENT_L', 'LINE', 0, 'NUM', 'MAX')
    return_data['line_id'] = mapdl.parameters['CURRENT_L']
    mapdl.run('CURRENT_L=')
    mapdl.finish()
    return return_data


def create_area(mapdl, coords):

    coords = np.array(coords)
    if coords.shape[0] > 18:
        raise Exception("Areas can only have up to 18 KP.")
    return_data = {}
    return_data['start_kp'] = _get_max_param_id(mapdl, 'KP') + 1
    mapdl.run('/PREP7')
    kp_id = return_data['start_kp']
    kp_list = []
    for coord in coords:
        mapdl.k(kp_id, coord[0], coord[1], coord[2])
        kp_list.append(kp_id)
        kp_id = kp_id + 1
    return_data['end_kp'] = kp_list[-1]
    for _ in range(len(kp_list), 18):
        kp_list.append('')
    mapdl.a(kp_list[0], kp_list[1], kp_list[2], kp_list[3], kp_list[4],
            kp_list[5], kp_list[6], kp_list[7], kp_list[8], kp_list[9],
            kp_list[10], kp_list[11], kp_list[12], kp_list[13], kp_list[14],
            kp_list[15], kp_list[16], kp_list[17])
    mapdl.get('CURRENT_A', 'AREA', 0, 'NUM', 'MAX')
    return_data['area_id'] = mapdl.parameters['CURRENT_A']
    mapdl.run('CURRENT_A=')
    mapdl.finish()
    return return_data


def create_prism(mapdl, x_origin, y_origin, width, depth, height):

    return_data = {}
    mapdl.run('/PREP7')
    mapdl.blc4(x_origin, y_origin, width, depth, height)
    mapdl.get('CURRENT_V', 'VOLU', 0, 'NUM', 'MAX')
    return_data['volume_id'] = mapdl.parameters['CURRENT_V']
    mapdl.run('CURRENT_V=')
    mapdl.finish()
    return return_data


def create_cylinder(mapdl, x_center, y_center, radius, height):

    return_data = {}
    mapdl.run('/PREP7')
    mapdl.cyl4(x_center, y_center, radius, '', '', '', height)
    mapdl.get('CURRENT_V', 'VOLU', 0, 'NUM', 'MAX')
    return_data['volume_id'] = mapdl.parameters['CURRENT_V']
    mapdl.run('CURRENT_V=')
    mapdl.finish()
    return return_data


def create_extruded_volume(mapdl, coords, thickness):

    return_data = {}
    area_id = create_area(mapdl, coords)
    mapdl.run('/PREP7')
    mapdl.voffst(area_id['area_id'], thickness)
    mapdl.get('CURRENT_V', 'VOLU', 0, 'NUM', 'MAX')
    return_data['volume_id'] = mapdl.parameters['CURRENT_V']
    mapdl.run('CURRENT_V=')
    mapdl.finish()
    return return_data


def set_beam3(mapdl, area, inertia, height):

    return_data = {}
    return_data['etype_id'] = _get_max_param_id(mapdl, 'ETYPE') + 1
    return_data['sectype_id'] = _get_max_param_id(mapdl, 'RCON') + 1
    mapdl.run('/PREP7')
    mapdl.et(return_data['etype_id'], 'BEAM3')
    mapdl.r(return_data['sectype_id'], area, inertia, height)
    mapdl.finish()
    return return_data


def set_solid186(mapdl):

    return_data = {}
    return_data['etype_id'] = _get_max_param_id(mapdl, 'ETYPE') + 1
    mapdl.run('/PREP7')
    mapdl.et(return_data['etype_id'], 'SOLID186')
    mapdl.finish()
    return return_data


def set_shell181(mapdl, thickness, material, angle=0, int_points=3,
                 offset='MID'):

    return_data = {}
    return_data['etype_id'] = _get_max_param_id(mapdl, 'ETYPE') + 1
    return_data['sec_id'] = _get_max_param_id(mapdl, 'GENS') + 1
    mapdl.run('/PREP7')
    mapdl.et(return_data['etype_id'], 'SHELL181')
    mapdl.sectype(return_data['sec_id'], 'SHELL')
    mapdl.secdata(thickness, material, angle, int_points)
    mapdl.secoffset(offset)
    mapdl.finish()
    return return_data


def _get_closest_node(mapdl, coordinates):

    node_list = get_node_list(mapdl)
    node_distance = cdist(np.array([coordinates]),
                          node_list[:, 1:4])
    return node_list[int(np.argmin(node_distance)), 0]


def get_node_list(mapdl, tol=6):

    node_coordinates = mapdl.mesh.nodes
    node_coordinates = np.hstack((mapdl.mesh.nnum.reshape(-1, 1),
                                  node_coordinates)).round(tol)
    node_coordinates = pd.DataFrame(node_coordinates,
                                    columns=['num', 'x', 'y', 'z'])
    node_coordinates.sort_values(['x', 'y', 'z', 'num'], inplace=True)
    node_coordinates.to_csv('nodes.csv')
    node_list = node_coordinates.to_numpy()
    return node_list


def select_nodes(mapdl, x_lim, y_lim, z_lim):

    mapdl.run('/SOL')
    # mapdl.run('*DEL,MAX_X')
    # mapdl.get('MAX_X', 'NODE', 0, 'MXLOC', 'X')
    mapdl.run('*DEL,MAX_Y')
    mapdl.get('MAX_Y', 'NODE', 0, 'MXLOC', 'Y')
    mapdl.run('*DEL,MAX_Z')
    mapdl.get('MAX_Z', 'NODE', 0, 'MXLOC', 'Z')
    # mapdl.run('*DEL,MIN_X')
    # mapdl.get('MIN_X', 'NODE', 0, 'MNLOC', 'X')
    mapdl.run('*DEL,MIN_Y')
    mapdl.get('MIN_Y', 'NODE', 0, 'MNLOC', 'Y')
    mapdl.run('*DEL,MIN_Z')
    mapdl.get('MIN_Z', 'NODE', 0, 'MNLOC', 'Z')
    # max_x = mapdl.parameters['MAX_X']
    max_y = mapdl.parameters['MAX_Y']
    max_z = mapdl.parameters['MAX_Z']
    # min_x = mapdl.parameters['MIN_X']
    min_y = mapdl.parameters['MIN_Y']
    min_z = mapdl.parameters['MIN_Z']
    mapdl.nsel('S', 'LOC', 'X', x_lim[0], x_lim[1])
    mapdl.nsel('U', 'LOC', 'Y', min_y - 1, y_lim[0])
    mapdl.nsel('U', 'LOC', 'Y', y_lim[1], max_y + 1)
    mapdl.nsel('U', 'LOC', 'Z', min_z - 1, z_lim[0])
    mapdl.nsel('U', 'LOC', 'Z', z_lim[1], max_z + 1)
    mapdl.finish()


def displacement_bc(mapdl, x_lim, y_lim, z_lim, x=True, y=True, z=True,
                    rotx=True, roty=True, rotz=True):

    select_nodes(mapdl, x_lim, y_lim, z_lim)
    mapdl.run('/SOL')
    if x:
        mapdl.d('ALL', 'UX', 0)
    if y:
        mapdl.d('ALL', 'UY', 0)
    if z:
        mapdl.d('ALL', 'UZ', 0)
    if rotx:
        mapdl.d('ALL', 'ROTX', 0)
    if roty:
        mapdl.d('ALL', 'ROTY', 0)
    if rotz:
        mapdl.d('ALL', 'ROTZ', 0)
    mapdl.allsel()
    mapdl.finish()


def elastic_support(mapdl, x_lim, y_lim, z_lim, normal_stiffness,
                    tangential_stiffness=None, pinball_radius=0):

    mapdl.run(f'ARG1={normal_stiffness}')
    tangential_stiffness = (normal_stiffness if tangential_stiffness is None
                            else tangential_stiffness)
    mapdl.run(f'ARG2={tangential_stiffness}')
    mapdl.run(f'ARG3={pinball_radius}')
    select_nodes(mapdl, x_lim, y_lim, z_lim)
    mapdl.cm('Elastic_Here', 'NODE')
    mapdl.allsel()
    # Source for the following code snippet: https://www.simutechgroup.com/tips-and-tricks/fea-articles/143-a-normal-and-tangential-elastic-foundation-in-workbench-mechanical
    mapdl.run('/PREP7')
    with mapdl.non_interactive:
        mapdl.run('*if,ARG1,LE,0,then')
        mapdl.run('*MSG,ERROR')
        mapdl.run(('ARG1 for Normal Stiffness on XYZ Elastic Foundation must'
                   ' be positive'))
        mapdl.run('/EOF')
        mapdl.run('*return,-1')
        mapdl.run('*endif')
        mapdl.run('*if,ARG2,LE,0,then')
        mapdl.run('ARG2=ARG1')
        mapdl.run('/COM,######## ARG2 was made equal to ARG1 ########')
        mapdl.run('*endif')
        mapdl.run('fini')
        mapdl.run('/prep7')
        mapdl.run('*get,nodemax,NODE,,NUM,MAX        ')
        mapdl.cmsel('s', 'Elastic_Here')
        mapdl.esln()
        mapdl.esel('u', 'ename', '', 151, 154)
        mapdl.esel('u', 'ename', '', 169, 180)
        mapdl.esel('u', 'ename', '', 188, 189)
        mapdl.run('*get,maxtype,ETYP,,NUM,MAX        ')
        mapdl.run('*get, maxmat,MAT,,NUM,MAX         ')
        mapdl.run('*get,maxreal,RCON,,NUM,MAX        ')
        mapdl.run('*if,maxtype,gt,maxmat,then')
        mapdl.run('maxmat=maxtype')
        mapdl.run('*else')
        mapdl.run('maxtype=maxmat')
        mapdl.run('*endif')
        mapdl.run('*if,maxreal,gt,maxtype,then')
        mapdl.run('maxtype=maxreal')
        mapdl.run('maxmat=maxreal')
        mapdl.run('*else')
        mapdl.run('maxreal=maxtype')
        mapdl.run('*endif')
        mapdl.et('maxtype+1', 'CONTA174', '', 1, '', 0, 3)
        mapdl.keyopt('maxtype+1', 9, 1)
        mapdl.keyopt('maxtype+1', 12, 5)
        mapdl.et('maxtype+2', 'TARGE170', '', 1)  #Constraints by user
        mapdl.r('maxreal+1', 0, 0, '-ARG1', '', '', '-abs(ARG3)')
        mapdl.rmodif('maxreal+1', 12, '-ARG2')
        mapdl.type('maxtype+1')
        mapdl.real('maxreal+1')
        mapdl.mat('maxmat+1')
        mapdl.esurf()
        mapdl.run('*get,current_nodemin,node,,num,min')
        mapdl.esln('r', 1)
        mapdl.esel('r', 'ename', '', 174)
        mapdl.esel('r', 'real', '', 'maxreal+1')
        mapdl.ngen(2, '(nodemax-current_nodemin)+1', 'ALL', '', '', 0, 0, 0)
        mapdl.egen(2, '(nodemax-current_nodemin)+1', 'ALL', '', '', 0, 1, 0)
        mapdl.esel('r', 'type', '', 'maxtype+2')
        mapdl.ensym(0, '', 0, 'ALL')
        mapdl.nsle()
        mapdl.d('all', 'all')
        mapdl.allsel()
    mapdl.finish()


def linear_elastic_surface_contact(mapdl, area_1_id, area_2_id,
                                   elastic_modulus=1e12, Poisson=0.3,
                                   density=0):

    """
    Create the mesh of SOLID186 elements of a rectangular plate with a
    rectangular stringer support united to it with a bonded contact.

    Parameters
    ----------
    mapdl : ansys.mapdl.mapdl_console.MapdlConsole or 
            ansys.mapdl.mapdl_corba.MapdlCorba class object
        An ANSYS MAPDL instance as started by ansys.mapdl.launch_mapdl()
    
    area_1_id : int
        The identifier of the first area, as master, that participates
        in the contact.

    area_1_id : int
        The identifier of the second area, as slave, that participates
        in the contact.

    elastic_modulus : float
        The isotropic linear elastic modulus for the material of the
        contact.

    Poisson : float
        The Poisson coefficient for the material of the contact.
    
    density : float
        The density value for the material of the contact.
    
    Returns
    -------
    out : dictionary
        Dictionary containing IDs created in the contact (TO BE
        IMPLEMENTED).

    Notes
    -----
    Units are not specified, but should be coherent. Any result obtained
    from a simulation using this function will also be coherent with the
    units used for the rest of figures.
    """

    mat_id = set_linear_elastic(mapdl, elastic_modulus, Poisson, density)
    rc_max = _get_max_param_id(mapdl, 'RCON')
    et_max = _get_max_param_id(mapdl, 'ETYPE')
    mapdl.run('/PREP7')
    mapdl.mp("MU", mat_id, "")
    mapdl.mp("EMIS", mat_id, 7.88860905221e-031)
    mapdl.cm("_NODECM", "NODE")
    mapdl.cm("_ELEMCM", "ELEM")
    mapdl.cm("_KPCM", "KP")
    mapdl.cm("_LINECM", "LINE")
    mapdl.cm("_AREACM", "AREA")
    mapdl.cm("_VOLUCM", "VOLU")
    mapdl.mat(mat_id)
    mapdl.r(rc_max + 1)
    mapdl.real(rc_max + 1)
    mapdl.et(et_max + 1, 170)
    mapdl.et(et_max + 2, 174)
    mapdl.r(rc_max + 1, "", "", 1.0, 0.1, 0, "")
    mapdl.rmore("", "", 1.0E20, 0.0, 1.0, "")
    mapdl.rmore(0.0, 0, 1.0, "", 1.0, 0.5)
    mapdl.rmore(0, 1.0, 1.0, 0.0, "", 1.0)
    mapdl.rmore("", "", "", "", "", 1.0)
    mapdl.keyopt(et_max + 2, 4, 0)
    mapdl.keyopt(et_max + 2, 5, 0)
    mapdl.keyopt(et_max + 2, 7, 0)
    mapdl.keyopt(et_max + 2, 8, 0)
    mapdl.keyopt(et_max + 2, 9, 0)
    mapdl.keyopt(et_max + 2, 10, 0)
    mapdl.keyopt(et_max + 2, 11, 0)
    mapdl.keyopt(et_max + 2, 12, 5)
    mapdl.keyopt(et_max + 2, 14, 0)
    mapdl.keyopt(et_max + 2, 18, 0)
    mapdl.keyopt(et_max + 2, 2, 0)
    mapdl.keyopt(et_max + 1, 5, 0)
    mapdl.asel("S", "", "", area_1_id)
    mapdl.cm("_TARGET", "AREA")
    mapdl.type(et_max + 1)
    mapdl.nsla("S", 1)
    mapdl.esln("S", 0)
    mapdl.esll("U")
    mapdl.esel("U", "ENAME", "", 188, 189)
    mapdl.nsle("A", "CT2")
    mapdl.esurf()
    mapdl.cmsel("S", "_ELEMCM")
    mapdl.asel("S", "", "", area_2_id)
    mapdl.cm("_CONTACT", "AREA")
    mapdl.type(et_max + 2)
    mapdl.nsla("S", 1)
    mapdl.esln("S", 0)
    mapdl.nsle("A", "CT2")
    mapdl.esurf()
    mapdl.r(rc_max + 2)
    mapdl.real(rc_max + 2)
    mapdl.et(et_max + 3, 170)
    mapdl.et(et_max + 4, 174)
    mapdl.r(rc_max + 2, "", "", 1.0, 0.1, 0, "")
    mapdl.rmore("", "", 1.0E20, 0.0, 1.0, "")
    mapdl.rmore(0.0, 0, 1.0, "", 1.0, 0.5)
    mapdl.rmore(0, 1.0, 1.0, 0.0, "", 1.0)
    mapdl.rmore("", "", "", "", "", 1.0)
    mapdl.keyopt(et_max + 4, 4, 0)
    mapdl.keyopt(et_max + 4, 5, 0)
    mapdl.keyopt(et_max + 4, 7, 0)
    mapdl.keyopt(et_max + 4, 8, 0)
    mapdl.keyopt(et_max + 4, 9, 0)
    mapdl.keyopt(et_max + 4, 10, 0)
    mapdl.keyopt(et_max + 4, 11, 0)
    mapdl.keyopt(et_max + 4, 12, 5)
    mapdl.keyopt(et_max + 4, 14, 0)
    mapdl.keyopt(et_max + 4, 18, 0)
    mapdl.keyopt(et_max + 4, 2, 0)
    mapdl.keyopt(et_max + 3, 1, 0)
    mapdl.keyopt(et_max + 3, 3, 0)
    mapdl.keyopt(et_max + 3, 5, 0)
    mapdl.type(et_max + 3)
    mapdl.esel("S", "TYPE", "", et_max + 2)
    mapdl.nsle("S")
    mapdl.esln("S", 0)
    mapdl.esurf()
    mapdl.type(et_max + 4)
    mapdl.esel("S", "TYPE", "", et_max + 1)
    mapdl.nsle("S")
    mapdl.esln("S", 0)
    mapdl.esurf()
    mapdl.allsel()
    mapdl.cmsel("A", "_NODECM")
    mapdl.run("CMDEL,_NODECM")
    mapdl.cmsel("A", "_ELEMCM")
    mapdl.run("CMDEL,_ELEMCM")
    mapdl.cmsel("S", "_KPCM")
    mapdl.run("CMDEL,_KPCM")
    mapdl.cmsel("S", "_LINECM")
    mapdl.run("CMDEL,_LINECM")
    mapdl.cmsel("S", "_AREACM")
    mapdl.run("CMDEL,_AREACM")
    mapdl.cmsel("S", "_VOLUCM")
    mapdl.run("CMDEL,_VOLUCM")
    mapdl.run("CMDEL,_TARGET")
    mapdl.run("CMDEL,_CONTACT")
    mapdl.finish()