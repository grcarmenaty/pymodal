import pymodal
import numpy as np


def cantilever_beam(ansys, elastic_modulus, poisson, density, damage_location,
                    b, h, l, damage_level, ndiv):
    damage_element_start = np.arange(0, l, l / ndiv)[
        int(np.floor(damage_location / (l/ndiv)))
    ]
    damage_element_end = damage_element_start + l / ndiv
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, poisson,
                                              density)
    line_start = pymodal.ansys.create_line(ansys, [0, 0, 0],
                                           [damage_element_start, 0, 0])
    line_damage = pymodal.ansys.create_line(
        ansys,
        [damage_element_start, 0, 0],
        [damage_element_end, 0, 0]
    )
    line_end = pymodal.ansys.create_line(ansys, [damage_element_end, 0, 0],
                                         [l, 0, 0])
    element_id_pristine = pymodal.ansys.set_beam3(ansys, b * h,
                                                  (b * h**3) / 12, h)
    ansys.run('/PREP7')
    ansys.esize(l/ndiv, 0)
    ansys.lmesh(line_start['line_id'])
    ansys.lmesh(line_end['line_id'])
    element_id_pristine = pymodal.ansys.set_beam3(
        ansys, b * h, damage_level * ((b * h**3)/12), h
    )
    ansys.run('/PREP7')
    ansys.esize(l/ndiv, 0)
    ansys.type(2)
    ansys.real(2)
    ansys.lmesh(line_damage['line_id'])
    ansys.nummrg('ALL')
    ansys.numcmp('NODE')
    ansys.dk(1, 'ALL')
    ansys.finish()
    return (damage_element_start, damage_element_end)


def free_beam(ansys, elastic_modulus, poisson, density, damage_location, b, h,
              l, damage_level, ndiv):
    damage_element_start = np.arange(0, l, l / ndiv)[
        int(np.floor(damage_location / (l/ndiv)))
    ]
    damage_element_end = damage_element_start + l / ndiv
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, poisson,
                                              density)
    line_start = pymodal.ansys.create_line(ansys, [0, 0, 0],
                                           [damage_element_start, 0, 0])
    line_damage = pymodal.ansys.create_line(
        ansys,
        [damage_element_start, 0, 0],
        [damage_element_end, 0, 0]
    )
    line_end = pymodal.ansys.create_line(ansys, [damage_element_end, 0, 0],
                                         [l, 0, 0])
    element_id_pristine = pymodal.ansys.set_beam3(ansys, b * h,
                                                  (b * h**3) / 12, h)
    ansys.run('/PREP7')
    ansys.esize(l/ndiv, 0)
    ansys.lmesh(line_start['line_id'])
    ansys.lmesh(line_end['line_id'])
    element_id_pristine = pymodal.ansys.set_beam3(
        ansys, b * h, damage_level * ((b * h**3)/12), h
    )
    ansys.run('/PREP7')
    ansys.esize(l/ndiv, 0)
    ansys.type(2)
    ansys.real(2)
    ansys.lmesh(line_damage['line_id'])
    ansys.nummrg('ALL')
    ansys.numcmp('NODE')
    ansys.finish()
    return (damage_element_start, damage_element_end)


def free_plate(ansys, elastic_modulus, poisson, density, thickness, a, b,
               e_size):
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, poisson,
                                              density)
    element_id = pymodal.ansys.set_shell181(ansys, thickness, mat_id)
    kp_list = [[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0]]
    area_id = pymodal.ansys.create_area(ansys, kp_list)
    ansys.run('/PREP7')
    ansys.esize(e_size, 0)
    ansys.amesh(area_id['area_id'])
    ansys.finish()


def free_plate_solid(ansys, elastic_modulus, poisson, density, thickness, a, b,
                     e_size):
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, poisson,
                                              density)
    element_id = pymodal.ansys.set_solid186(ansys)
    volume_id = pymodal.ansys.create_prism(ansys, 0, 0, a, b, thickness)
    ansys.run('/PREP7')
    ansys.esize(e_size, 0)
    ansys.vmesh(volume_id['volume_id'])
    ansys.finish()


def circ_hole_solid(ansys, elastic_modulus, poisson, density, thickness, a, b,
                    e_size, center, radius):
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, poisson,
                                              density)
    element_id = pymodal.ansys.set_solid186(ansys)
    prism_id = pymodal.ansys.create_prism(ansys, 0, 0, a, b, thickness)
    cyl_id = pymodal.ansys.create_cylinder(ansys, center[0], center[1], radius,
                                           thickness)
    ansys.run('/PREP7')
    ansys.vsbv(prism_id['volume_id'], cyl_id['volume_id'])
    ansys.esize(e_size, 0)
    ansys.vsweep(cyl_id['volume_id'] + 1)
    ansys.finish()


def crack_analogy_solid(ansys, elastic_modulus, poisson, density, thickness, a,
                        b, e_size, start, end, width):
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, poisson,
                                              density)
    element_id = pymodal.ansys.set_solid186(ansys)
    prism_id = pymodal.ansys.create_prism(ansys, 0, 0, a, b, thickness)
    start = np.array([start[0], start[1], 0])
    end = np.array([end[0], end[1], 0])
    direction = end - start
    direction = direction / np.linalg.norm(direction, 2)
    perpendicular = np.array([-direction[1], direction[0], 0])
    coords = [
        start + (width / 2)*perpendicular,
        start - (width / 2)*perpendicular,
        end - (width / 2)*perpendicular,
        end + (width / 2)*perpendicular
    ]
    crack_id = pymodal.ansys.create_extruded_volume(ansys, coords, 0.00498)
    ansys.run('/PREP7')
    ansys.vsbv(prism_id['volume_id'], crack_id['volume_id'])
    ansys.esize(e_size, 0)
    ansys.vsweep(crack_id['volume_id'] + 1)
    ansys.finish()


def stringer_support_solid(ansys, elastic_modulus, poisson, density, thickness, a,
                     b, e_size, start, end, width, height):
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, poisson,
                                              density)
    element_id = pymodal.ansys.set_solid186(ansys)
    prism_id = pymodal.ansys.create_prism(ansys, 0, 0, a, b, thickness)
    start = np.array([start[0], start[1]])
    end = np.array([end[0], end[1]])
    direction = end - start
    direction = direction / np.linalg.norm(direction, 2)
    perpendicular = np.array([-direction[1], direction[0]])
    coords = [
        np.append(start + (width / 2)*perpendicular, thickness),
        np.append(start - (width / 2)*perpendicular, thickness),
        np.append(end - (width / 2)*perpendicular, thickness),
        np.append(end + (width / 2)*perpendicular, thickness)
    ]
    stringer_id = pymodal.ansys.create_extruded_volume(ansys, coords, height)
    ansys.run('/PREP7')
    # ansys.vadd(prism_id['volume_id'], stringer_id['volume_id'])
    ansys.esize(e_size, 0)
    ansys.vmesh('ALL')
    ansys.asel('S', 'LOC', 'Z', thickness, thickness)
    ansys.run('/PREP7')
    ansys.run('*DEL,MAX_PARAM')
    ansys.run('*DEL,MIN_PARAM')
    ansys.get('MAX_PARAM', 'AREA', 0, 'NUM', 'MAX')
    ansys.get('MIN_PARAM', 'AREA', 0, 'NUM', 'MIN')
    ansys.load_parameters()
    area_1 = ansys.parameters['MAX_PARAM']
    area_2 = ansys.parameters['MIN_PARAM']
    pymodal.ansys.bonded_surface_contact(ansys, area_1, area_2)
    ansys.finish()