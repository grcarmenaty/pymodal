import pymodal
import numpy as np
from scipy.spatial.distance import cdist


def cantilever_beam(mapdl, elastic_modulus, Poisson, density, damage_location,
                    b, h, l, damage_level, ndiv):
    damage_level = 1 - damage_level
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    line_start = pymodal.mapdl.create_line(mapdl, [0, 0, 0],
                                           [damage_location[0], 0, 0])
    line_damage = pymodal.mapdl.create_line(
        mapdl,
        [damage_location[0], 0, 0],
        [damage_location[1], 0, 0]
    )
    line_end = pymodal.mapdl.create_line(mapdl, [damage_location[1], 0, 0],
                                         [l, 0, 0])
    element_id_pristine = pymodal.mapdl.set_beam3(mapdl, b * h,
                                                  (b * h**3) / 12, h)
    mapdl.prep7()
    mapdl.esize(l/ndiv, 0)
    mapdl.lmesh(line_start['line_id'])
    mapdl.lmesh(line_end['line_id'])
    element_id_damaged = pymodal.mapdl.set_beam3(
        mapdl, b * h, damage_level * ((b * h**3)/12), h
    )
    mapdl.prep7()
    mapdl.esize(l/ndiv, 0)
    mapdl.type(2)
    mapdl.real(2)
    mapdl.lmesh(line_damage['line_id'])
    mapdl.nummrg('ALL')
    mapdl.numcmp('NODE')
    mapdl.dk(1, 'ALL')
    mapdl.finish()
    return {'mat_id': mat_id, 'element_id_pristine': element_id_pristine,
            'element_id_damaged': element_id_damaged, 'line_start': line_start,
            'line_damage': line_damage, 'line_end': line_end}


def free_beam_solid(mapdl, elastic_modulus, Poisson, density, b, h, l,
                    e_size):
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    element_id = pymodal.mapdl.set_solid186(mapdl)
    volume_id = pymodal.mapdl.create_prism(mapdl, 0, 0, b, l, h)
    mapdl.prep7()
    mapdl.esize(e_size, 0)
    mapdl.vmesh(volume_id['volume_id'])
    mapdl.finish()
    return {'mat_id': mat_id, 'element_id': element_id,
            'volume_id': volume_id}


def damaged_beam_solid(mapdl, elastic_modulus, Poisson, density, b, h, l,
                       e_size, damage_location, damage_width, damage_depth):
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    element_id = pymodal.mapdl.set_solid186(mapdl)
    beam_id = pymodal.mapdl.create_prism(mapdl, 0, 0, b, l, h)
    indentation_id = pymodal.mapdl.create_prism(
        mapdl, 0, damage_location-damage_width/2, b, damage_width, damage_depth
    )
    pymodal.mapdl.move_volume(mapdl, indentation_id['volume_id'], 0, 0,
                              h-damage_depth)
    mapdl.prep7()
    mapdl.vsbv(beam_id['volume_id'], indentation_id['volume_id'])
    mapdl.esize(e_size, 0)
    mapdl.vsweep(indentation_id['volume_id']+1)
    mapdl.finish()
    return {'mat_id': mat_id, 'element_id': element_id,
            'volume_id': indentation_id['volume_id']+1}


def mass_beam_solid(mapdl, elastic_modulus, Poisson, density, b, h, l,
                    e_size, mass_location, mass_value):
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    element_id = pymodal.mapdl.set_solid186(mapdl)
    volume_id = pymodal.mapdl.create_prism(mapdl, 0, 0, b, l, h)
    mapdl.prep7()
    mapdl.esize(e_size, 0)
    mapdl.vmesh(volume_id['volume_id'])
    node_list = pymodal.mapdl.get_node_list(mapdl)
    closest_node = cdist(np.array([[b/2, mass_location, h]]), node_list[:, 1:4])
    node_id = int(node_list[int(np.argmin(closest_node)), 0])
    current_etype = pymodal.mapdl.set_mass21(mapdl, mass_value)
    mapdl.prep7()
    mapdl.type(current_etype['etype_id'])
    mapdl.real(current_etype['real_constant_id'])
    mapdl.e(node_id)
    mapdl.finish()
    return {'mat_id': mat_id, 'element_id': element_id,
            'volume_id': volume_id}


def stringer_beam_solid(mapdl, elastic_modulus, Poisson, density, b, h, l,
                    e_size, stringer_start, stringer_length, stringer_height):
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    element_id = pymodal.mapdl.set_solid186(mapdl)
    volume_id = pymodal.mapdl.create_prism(mapdl, 0, 0, b, l, h)
    stringer_id = pymodal.mapdl.create_prism(mapdl, b/2-h/2, stringer_start, h,
                                             stringer_length, stringer_height)
    pymodal.mapdl.move_volume(mapdl, stringer_id['volume_id'], 0, 0, h)
    mapdl.prep7()
    mapdl.esize(e_size, 0) # Set element size
    mapdl.vmesh('ALL') # Mesh the defined volumes
    # Select the areas based on their location coordinates in the Z axis, only
    # the ones contained in the plane parallel to the Z plane at the value of
    # z = thickness. This plane only contains the lower face of the stringer
    # support and the upper face of the plate.
    mapdl.asel('S', 'LOC', 'Z', h, h)
    mapdl.prep7()
    # Get the IDs for both areas
    mapdl.run('*DEL,MAX_PARAM')
    mapdl.run('*DEL,MIN_PARAM')
    mapdl.get('MAX_PARAM', 'AREA', 0, 'NUM', 'MAX')
    mapdl.get('MIN_PARAM', 'AREA', 0, 'NUM', 'MIN')
    area_1 = mapdl.parameters['MAX_PARAM']
    area_2 = mapdl.parameters['MIN_PARAM']
    # Deifne a bonded contact between both areas
    pymodal.mapdl.linear_elastic_surface_contact(mapdl, area_1, area_2,
                                                 elastic_modulus*10e7)
    mapdl.prep7()
    mapdl.vmesh("ALL")
    mapdl.finish()
    return {'mat_id': mat_id, 'element_id': element_id,
            'volume_id': stringer_id['volume_id']}


def free_beam(mapdl, elastic_modulus, Poisson, density, damage_location, b, h,
              l, damage_level, ndiv):
    damage_element_start = np.arange(0, l, l / ndiv)[
        int(np.floor(damage_location / (l/ndiv)))
    ]
    damage_element_end = damage_element_start + l / ndiv
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    line_start = pymodal.mapdl.create_line(mapdl, [0, 0, 0],
                                           [damage_element_start, 0, 0])
    line_damage = pymodal.mapdl.create_line(
        mapdl,
        [damage_element_start, 0, 0],
        [damage_element_end, 0, 0]
    )
    line_end = pymodal.mapdl.create_line(mapdl, [damage_element_end, 0, 0],
                                         [l, 0, 0])
    element_id_pristine = pymodal.mapdl.set_beam3(mapdl, b * h,
                                                  (b * h**3) / 12, h)
    mapdl.prep7()
    mapdl.esize(l/ndiv, 0)
    mapdl.lmesh(line_start['line_id'])
    mapdl.lmesh(line_end['line_id'])
    element_id_pristine = pymodal.mapdl.set_beam3(
        mapdl, b * h, damage_level * ((b * h**3)/12), h
    )
    mapdl.prep7()
    mapdl.esize(l/ndiv, 0)
    mapdl.type(2)
    mapdl.real(2)
    mapdl.lmesh(line_damage['line_id'])
    mapdl.nummrg('ALL')
    mapdl.numcmp('NODE')
    mapdl.finish()
    return (damage_element_start, damage_element_end)


def free_plate_shell(mapdl, elastic_modulus, Poisson, density, thickness, a, b,
                     e_size):
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    element_id = pymodal.mapdl.set_shell181(mapdl, thickness, mat_id)
    kp_list = [[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0]]
    area_id = pymodal.mapdl.create_area(mapdl, kp_list)
    mapdl.prep7()
    mapdl.esize(e_size, 0)
    mapdl.aatt(mat_id, "", element_id["etype_id"], "", "")
    mapdl.amesh(int(area_id['area_id']))
    mapdl.finish()
    return {
        "mat_id": mat_id,
        "element_id": element_id,
        "area_id": area_id
    }


def free_plate_solid(mapdl, elastic_modulus, Poisson, density, thickness, a, b,
                     e_size):
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    element_id = pymodal.mapdl.set_solid186(mapdl)
    volume_id = pymodal.mapdl.create_prism(mapdl, 0, 0, a, b, thickness)
    mapdl.prep7()
    mapdl.esize(e_size, 0)
    mapdl.vmesh(volume_id['volume_id'])
    mapdl.finish()
    return {'mat_id': mat_id, 'element_id': element_id,
            'volume_id': volume_id}


def circ_hole_solid(mapdl, elastic_modulus, Poisson, density, thickness, a, b,
                    e_size, center, radius):

    """
    Create the mesh of SOLID186 elements of a rectangular plate with a
    rectangular stringer support united to it with a bonded contact.

    Parameters
    ----------
    mapdl : pymapdl.mapdl_console.MapdlConsole or 
            pymapdl.mapdl_corba.MapdlCorba class object
        An ANSYS MAPDL instance as started by pymapdl.launch_mapdl()
    
    elastic_modulus : float
        The isotropic linear elastic modulus for the material of the
        plate and the stringer support

    elastic_modulus : float
        The isotropic linear elastic modulus for the material of the
        plate and the stringer support.

    Poisson : float
        The Poisson coefficient for the material of the plate and the
        stringer support.
    
    density : float
        The density value for the material of the plate and the stringer
        support.

    thickness : float
        The thickness of the plate.
    
    a : float
        Size of the side of the plate along the x axis.
    
    b : float
        Size of the side of the plate along the y axis.

    e_size : float
        Element size for the mesh.
    
    center : iterable
        Coordinates of the center of the hole, only the first two
        positions are used.

    radius : float
        Radius of the hole.

    Returns
    -------
    out : dictionary
        Dictionary containing the material ID, the element ID, the plate
        volume ID, the ID of the volume used to substract the hole from
        the plate volume and  the ID of the plate resulting from the
        substraction.

    Notes
    -----
    Units are not specified, but should be coherent. Any result obtained
    from a simulation using this function will also be coherent with the
    units used for the rest of figures.
    """

    # Define a linear elastic material with the specified properties
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    # Define a SOLID186 element type for meshing
    element_id = pymodal.mapdl.set_solid186(mapdl)
    # Create a rectangular prism volume with the specified measurements
    plate_id = pymodal.mapdl.create_prism(mapdl, 0, 0, a, b, thickness)
    # Create a cylinder as tall as the plate is thick, with it's center on the
    # specified coordinates.
    hole_id = pymodal.mapdl.create_cylinder(mapdl, center[0], center[1],
                                            radius, thickness)
    mapdl.prep7()
    # Substract the cylinder from the plate
    mapdl.vsbv(plate_id['volume_id'], hole_id['volume_id'])
    mapdl.esize(e_size, 0) # Define element size
    mapdl.vsweep(hole_id['volume_id'] + 1) # Mesh the volume
    mapdl.finish()
    return {'mat_id': mat_id, 'element_id': element_id,
            'plate_id': plate_id['volume_id'], 'hole_id': hole_id['volume_id'],
            'modified_plate_id': hole_id['volume_id'] + 1}


def crack_analogy_solid(mapdl, elastic_modulus, Poisson, density, thickness, a,
                        b, e_size, start, end, width):
    
    """
    Create the mesh of SOLID186 elements of a rectangular plate with a
    rectangular stringer support united to it with a bonded contact.

    Parameters
    ----------
    mapdl : pymapdl.mapdl_console.MapdlConsole or 
            pymapdl.mapdl_corba.MapdlCorba class object
        An ANSYS MAPDL instance as started by pymapdl.launch_mapdl()
    
    elastic_modulus : float
        The isotropic linear elastic modulus for the material of the
        plate and the stringer support

    elastic_modulus : float
        The isotropic linear elastic modulus for the material of the
        plate and the stringer support.

    Poisson : float
        The Poisson coefficient for the material of the plate and the
        stringer support.
    
    density : float
        The density value for the material of the plate and the stringer
        support.

    thickness : float
        The thickness of the plate.
    
    a : float
        Size of the side of the plate along the x axis.
    
    b : float
        Size of the side of the plate along the y axis.

    e_size : float
        Element size for the mesh.

    start : iterable
        One of two points defining the straight line along which the
        crack analogy is created. Only the first two positions are
        used, the first one for the x coordinate and the second one for
        the y coordinate.

    end : iterable
        One of two points defining the straight line along which the
        crack analogy is created. Only the first two positions are
        used, the first one for the x coordinate and the second one for
        the y coordinate.

    width : float
        The width of the crack analogy, with half of it going to each 
        side of the line defined between start and end.
    
    Returns
    -------
    out : dictionary
        Dictionary containing the material ID, the element ID, the plate
        volume ID, the ID of the volume used to substract the crack
        analogy from the plate volume and the ID of the volume resuting
        from the substraction.

    Notes
    -----
    Units are not specified, but should be coherent. Any result obtained
    from a simulation using this function will also be coherent with the
    units used for the rest of figures.
    """

    # Define a linear elastic material with the specified properties
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    # Define a SOLID186 element type for meshing
    element_id = pymodal.mapdl.set_solid186(mapdl)
    # Create a rectangular prism volume with the specified measurements
    plate_id = pymodal.mapdl.create_prism(mapdl, 0, 0, a, b, thickness)
    # Make sure start and end are arrays
    start = np.array([start[0], start[1]])
    end = np.array([end[0], end[1]])
    # Calculate the director vector for the line normalized to unit norm
    direction = end - start
    direction = direction / np.linalg.norm(direction, 2)
    # Calculate the director vector for a line perpendicular to the one defined
    # above
    perpendicular = np.array([-direction[1], direction[0]])
    # Calculate the coordinates for the four corners of the rectangle defining
    # the crack analogy
    coords = [
        np.append(start + (width / 2)*perpendicular, -thickness*0.1),
        np.append(start - (width / 2)*perpendicular, -thickness*0.1),
        np.append(end - (width / 2)*perpendicular, -thickness*0.1),
        np.append(end + (width / 2)*perpendicular, -thickness*0.1)
    ]
    # Create a volume that, when substracted from the volume defining the
    # plate, produces the crack analogy, based on the previously defined
    # coordinates
    crack_id = pymodal.mapdl.create_extruded_volume(mapdl, coords,
                                                    thickness*1.2)
    mapdl.prep7()
    # Substract the volume defining the crack from the volume defining the
    # plate.
    mapdl.vsbv(plate_id['volume_id'], crack_id['volume_id'])
    mapdl.esize(e_size, 0) # Define element size
    mapdl.vsweep(crack_id['volume_id'] + 1) # Mesh the volume
    mapdl.finish()
    return {'mat_id': mat_id, 'element_id': element_id, 'plate_id': plate_id, 
            'crack': crack_id['volume_id'],
            'modified_plate_id': crack_id['volume_id'] + 1}


def stringer_support_solid(mapdl, elastic_modulus, Poisson, density, thickness,
                           a, b, e_size, start, end, width, height):
    
    """
    Create the mesh of SOLID186 elements of a rectangular plate with a
    rectangular stringer support united to it with a bonded contact.

    Parameters
    ----------
    mapdl : pymapdl.mapdl_console.MapdlConsole or 
            pymapdl.mapdl_corba.MapdlCorba class object
        An ANSYS MAPDL instance as started by pymapdl.launch_mapdl()
    
    elastic_modulus : float
        The isotropic linear elastic modulus for the material of the
        plate and the stringer support

    elastic_modulus : float
        The isotropic linear elastic modulus for the material of the
        plate and the stringer support.

    Poisson : float
        The Poison coefficient for the material of the plate and the
        stringer support.
    
    density : float
        The density value for the material of the plate and the stringer
        support.

    thickness : float
        The thickness of the plate.
    
    a : float
        Size of the side of the plate along the x axis.
    
    b : float
        Size of the side of the plate along the y axis.

    e_size : float
        Element size for the mesh.

    start : iterable
        One of two points defining the straight line along which the
        stringer support is created. Only the first two positions are
        used, the first one for the x coordinate and the second one for
        the y coordinate.

    end : iterable
        One of two points defining the straight line along which the
        stringer support is created. Only the first two positions are
        used, the first one for the x coordinate and the second one for
        the y coordinate.

    width : float
        The width of the stringer, with half of it going to each side of
        the line defined between start and end.
    
    height : float
        The height of the stringer, how much it protrudes from the
        plate.
    
    Returns
    -------
    out : dictionary
        Dictionary containing the material ID, the element ID, the plate
        volume ID and the stringer volume ID.

    Notes
    -----
    Units are not specified, but should be coherent. Any result obtained
    from a simulation using this function will also be coherent with the
    units used for the rest of figures.
    """

    # Define a linear elastic material with the specified properties
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    # Define a SOLID186 element type for meshing
    element_id = pymodal.mapdl.set_solid186(mapdl)
    # Create a rectangular prism volume with the specified measurements
    plate_id = pymodal.mapdl.create_prism(mapdl, 0, 0, a, b, thickness)
    # Make sure start and end are arrays
    start = np.array([start[0], start[1]])
    end = np.array([end[0], end[1]])
    # Calculate the director vector for the line normalized to unit norm
    direction = end - start
    direction = direction / np.linalg.norm(direction, 2)
    # Calculate the director vector for a line perpendicular to the one defined
    # above
    perpendicular = np.array([-direction[1], direction[0]])
    # Calculate the coordinates for the four corners of the rectangle defining
    # the stringer support
    coords = [
        np.append(start + (width / 2)*perpendicular, thickness),
        np.append(start - (width / 2)*perpendicular, thickness),
        np.append(end - (width / 2)*perpendicular, thickness),
        np.append(end + (width / 2)*perpendicular, thickness)
    ]
    # Create the stringer volume based upon the coordinates calculated above
    stringer_id = pymodal.mapdl.create_extruded_volume(mapdl, coords, height)
    mapdl.prep7() # Enter preprocessor
    mapdl.esize(e_size, 0) # Set element size
    mapdl.vmesh('ALL') # Mesh the defined volumes
    # Select the areas based on their location coordinates in the Z axis, only
    # the ones contained in the plane parallel to the Z plane at the value of
    # z = thickness. This plane only contains the lower face of the stringer
    # support and the upper face of the plate.
    mapdl.asel('S', 'LOC', 'Z', thickness, thickness)
    mapdl.prep7()
    # Get the IDs for both areas
    mapdl.run('*DEL,MAX_PARAM')
    mapdl.run('*DEL,MIN_PARAM')
    mapdl.get('MAX_PARAM', 'AREA', 0, 'NUM', 'MAX')
    mapdl.get('MIN_PARAM', 'AREA', 0, 'NUM', 'MIN')
    area_1 = mapdl.parameters['MAX_PARAM']
    area_2 = mapdl.parameters['MIN_PARAM']
    # Deifne a bonded contact between both areas
    pymodal.mapdl.linear_elastic_surface_contact(mapdl, area_1, area_2,
                                                 elastic_modulus*10e7)
    mapdl.finish()
    return {'mat_id': mat_id, 'element_id': element_id, 'plate_id': plate_id, 
            'stringer_id': stringer_id['volume_id']}


def los_alamos_building(mapdl, floor_elastic_moduli, column_elastic_moduli,
                         floor_Poisson, column_Poisson, floor_densities,
                         column_densities, e_size, column_thicknesses,
                         column_widths, floor_width, floor_heights,
                         floor_depth, floor_thicknesses, contact_strengths,
                         mass_coordinates, mass_values, foundation,
                         foundation_points, foundation_moduli,
                         foundation_Poisson, foundation_densities,
                         foundation_width, foundation_height,
                         foundation_depth, foundation_contact_strength):

    # Define a SOLID186 element type for meshing
    element_id = pymodal.mapdl.set_solid186(mapdl)

    floor_id = {}
    floor_mat_id = {}
    
    for i in range(len(list(floor_heights))+1):
        # Define material for current floor
        floor_mat_id[f'floor_{i}'] = pymodal.mapdl.set_linear_elastic(
            mapdl, floor_elastic_moduli[i], floor_Poisson[i],
            floor_densities[i]
        )
        # Create a rectangular prism volume with the specified measurements
        floor_id[f'floor_{i}'] = pymodal.mapdl.create_prism(
            mapdl, 0, 0, floor_width, floor_depth, floor_thicknesses[i]
        )
        # Move the prism up according to sum of floor heights and thicknesses
        # up to current floor
        pymodal.mapdl.move_volume(
            mapdl, floor_id[f'floor_{i}']['volume_id'], 0, 0,
            sum(floor_heights[0:i])+sum(floor_thicknesses[0:i])
        )
        mapdl.prep7()
        mapdl.esize(e_size, 0) # Set element size
        mapdl.aatt(floor_mat_id[f'floor_{i}'], "", element_id["etype_id"], "",
                   "")
        # Mesh current floor
        mapdl.vmesh(floor_id[f'floor_{i}']['volume_id'])

    col_id = {}
    col_mat_id = {}

    for i in range(len(list(floor_heights))):
        for j in range(4):
            col = 4*i + j
            # Define material for current column
            col_mat_id[f'floor_{i}_col_{j}'] = (
                pymodal.mapdl.set_linear_elastic(
                    mapdl, column_elastic_moduli[col], column_Poisson[i],
                    column_densities[col]
                )
            )
            # Create a rectangular prism volume with the specified measurements
            col_id[f'floor_{i}_col_{j}'] = pymodal.mapdl.create_prism(
                mapdl, 0, 0, column_thicknesses[col], column_widths[col],
                floor_heights[i]+floor_thicknesses[i]/2+
                floor_thicknesses[i+1]/2
            )
            # Move the prism to its rightful position
            if j == 0:
                dx = -column_thicknesses[col]
                dy = 0
            elif j == 1:
                dx = floor_width
                dy = 0
            elif j == 2:
                dx = floor_width
                dy = floor_depth - column_widths[col]
            else:
                dx = -column_thicknesses[col]
                dy = floor_depth - column_widths[col]
            pymodal.mapdl.move_volume(
                mapdl, col_id[f'floor_{i}_col_{j}']['volume_id'], dx, dy,
                sum(floor_heights[0:i])+sum(floor_thicknesses[0:i])+
                floor_thicknesses[i]/2
            )
            mapdl.prep7()
            mapdl.esize(e_size, 0) # Set element size
            mapdl.aatt(col_mat_id[f'floor_{i}_col_{j}'], "",
                       element_id["etype_id"], "", "")
            # Mesh current column
            mapdl.vmesh(col_id[f'floor_{i}_col_{j}']['volume_id'])
    col_area_id = []
    floor_area_id = []
    for i in range(len(list(floor_heights))):
        for j in range(8):
            if j == 0 or j == 4:
                col = 0
                x_min = 0
                x_max = 0
                y_min = 0
                y_max = column_widths[col]
            elif j == 1 or j == 5:
                col = 1
                x_min = floor_width
                x_max = floor_width
                y_min = 0
                y_max = column_widths[col]
            elif j == 2 or j == 6:
                col = 2
                x_min = floor_width
                x_max = floor_width
                y_min = floor_depth - column_widths[col]
                y_max = floor_depth
            else:
                col = 3
                x_min = 0
                x_max = 0
                y_min = floor_depth - column_widths[col]
                y_max = floor_depth
            z_min = (sum(floor_heights[0:i]) + sum(floor_thicknesses[0:i]) +
                     floor_thicknesses[i]/2)
            z_max = (z_min + floor_heights[i] + floor_thicknesses[i]/2 +
                     floor_thicknesses[i+1]/2)
            pymodal.mapdl.select_areas(mapdl, (x_min, x_max), (y_min, y_max),
                                       (z_min, z_max))
            mapdl.prep7()
            mapdl.run('*DEL,A_ID')
            mapdl.get('A_ID', 'AREA', 0, 'NUM', 'MAX')
            current_col_id = mapdl.parameters['A_ID']
            col_area_id.append(current_col_id)

            if j == 0 or j == 4 or j == 3 or j == 7:
                x_min = 0
                x_max = 0
                y_min = 0
                y_max = floor_depth
            else:
                x_min = floor_width
                x_max = floor_width
                y_min = 0
                y_max = floor_depth
            if j < 4:
                z_min = sum(floor_heights[0:i]) + sum(floor_thicknesses[0:i])
                z_max = (sum(floor_heights[0:i]) + sum(floor_thicknesses[0:i])
                         + floor_thicknesses[i])
            else:
                z_min = (sum(floor_heights[0:i+1]) +
                         sum(floor_thicknesses[0:i+1]))
                z_max = (sum(floor_heights[0:i+1]) +
                         sum(floor_thicknesses[0:i+1]) + floor_thicknesses[i])
            pymodal.mapdl.select_areas(mapdl, (x_min, x_max), (y_min, y_max),
                                       (z_min, z_max))
            mapdl.prep7()
            mapdl.run('*DEL,A_ID')
            mapdl.get('A_ID', 'AREA', 0, 'NUM', 'MAX')
            current_floor_id = mapdl.parameters['A_ID']
            floor_area_id.append(current_floor_id)
            pymodal.mapdl.linear_elastic_surface_contact(
                mapdl,
                current_floor_id,
                current_col_id,
                contact_strengths[8*i+j]
            )
            mapdl.allsel()
    if foundation:
        pymodal.mapdl.select_areas(mapdl, (-1, 1), (-1, 1), (-1, e_size/2))
        mapdl.prep7()
        mapdl.run('*DEL,A_ID')
        mapdl.get('A_ID', 'AREA', 0, 'NUM', 'MAX')
        base_area_id = mapdl.parameters['A_ID']
        mapdl.allsel()
        foundation_id = {}
        foundation_mat_id = {}
        foundation_base_area_id = {}
        for i, foundation_point in enumerate(foundation_points):
            foundation_mat_id[f'foundation_{i}'] = (
                pymodal.mapdl.set_linear_elastic(
                    mapdl, foundation_moduli[i], foundation_Poisson[i],
                    foundation_densities[i]
                )
            )
            current_x_origin = foundation_point[0] - foundation_width/2
            current_y_origin = foundation_point[1] - foundation_depth/2
            foundation_id[f'foundation_{i}'] = pymodal.mapdl.create_prism(
                mapdl, current_x_origin, current_y_origin,
                foundation_width, foundation_depth, foundation_height
            )
            pymodal.mapdl.move_volume(
                mapdl, foundation_id[f'foundation_{i}']['volume_id'], 0, 0,
                -foundation_height
            )
            mapdl.prep7()
            mapdl.aatt(foundation_mat_id[f'foundation_{i}'], "",
                       element_id["etype_id"], "", "")
            # Mesh current column
            mapdl.vmesh(foundation_id[f'foundation_{i}']['volume_id'])
            pymodal.mapdl.select_areas(mapdl, (-2*floor_width, 2*floor_width), 
                                       (-2*floor_depth, 2*floor_depth),
                                       (-e_size/2, e_size/2))
            mapdl.run('*DEL,A_ID')
            mapdl.get('A_ID', 'AREA', 0, 'NUM', 'MAX')
            current_foundation_top_area_id = mapdl.parameters['A_ID']
            pymodal.mapdl.linear_elastic_surface_contact(
                mapdl,
                current_foundation_top_area_id,
                base_area_id,
                foundation_contact_strength
            )
            pymodal.mapdl.select_areas(mapdl, (-2*floor_width, 2*floor_width), 
                                       (-2*floor_depth, 2*floor_depth),
                                       (-2*foundation_height,
                                        -foundation_height+e_size/2))
            mapdl.prep7()
            mapdl.run('*DEL,A_ID')
            mapdl.get('A_ID', 'AREA', 0, 'NUM', 'MAX')
            foundation_base_area_id[f'foundation_{i}'] = mapdl.parameters[
                'A_ID'
            ]
            mapdl.allsel()
    node_list = pymodal.mapdl.get_node_list(mapdl)
    node_id_list = []
    for row in mass_coordinates:
        closest_node = cdist(np.asarray([row]), node_list[:, 1:4])
        node_id_list.append(int(node_list[int(np.argmin(closest_node)), 0]))
    for i, node in enumerate(node_id_list):
        current_etype = pymodal.mapdl.set_mass21(mapdl, mass_values[i])
        mapdl.prep7()
        mapdl.type(current_etype['etype_id'])
        mapdl.real(current_etype['real_constant_id'])
        mapdl.e(node)
    mapdl.finish()


def add_spring(mapdl, anchor_node, k, destination=None, destination_node=None):
    if destination_node is None:
        if destination is None:
            raise Exception("At least one of destination or destnation node"
                            " must be specifiedd.")
        anchor_node_id = pymodal.mapdl._get_closest_node(mapdl, anchor_node)
        destination_node_id = (
            pymodal.mapdl._get_max_param_id(mapdl, 'NODE') + 1
        )
        mapdl.prep7()
        mapdl.n(destination_node_id, destination[0], destination[1],
                destination[2])
        length = cdist(np.array([anchor_node]), np.array([destination]))[0][0]
    else:
        if destination is not None:
            raise Exception("Destination and destination node cannot be both"
                            " defined at the same time")
        anchor_node_id = pymodal.mapdl._get_closest_node(mapdl, anchor_node)
        destination_node_id = pymodal.mapdl._get_closest_node(mapdl,
                                                              destination_node)
        length = cdist(
            np.array([anchor_node]), np.array([destination_node])
        )[0][0]
    element_type = pymodal.mapdl.set_link180(mapdl, length, 0, 0)
    material = pymodal.mapdl.set_linear_elastic(mapdl, k, 0)
    element_id = (pymodal.mapdl._get_max_param_id(mapdl, 'ELEM') + 1)
    # mat_id = pymodal.mapdl._get_max_param_id(mapdl, 'MAT') + 1
    # etype_id = pymodal.mapdl._get_max_param_id(mapdl, 'ETYPE') + 1
    # sec_id = pymodal.mapdl._get_max_param_id(mapdl, 'LINK') + 1
    mapdl.prep7()
    # mapdl.mp('EX', mat_id, k)
    # mapdl.mp('PRXY', mat_id, 0)
    # mapdl.et(etype_id, 'LINK180')
    # mapdl.sectype(sec_id, "LINK")
    # mapdl.secdata(length)
    # mapdl.seccontrol(0, 0)
    mapdl.type(element_type['etype_id'])
    mapdl.mat(material)
    # print(anchor_node_id)
    # print(destination_node_id)
    # mapdl.open_gui()
    mapdl.en(element_id, anchor_node_id, destination_node_id)
    mapdl.finish()
    return {
        "anchor_node_id": anchor_node_id,
        "destination_node_id": destination_node_id
    }
