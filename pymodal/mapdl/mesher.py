import pymodal
import numpy as np


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
    mapdl.amesh(area_id['area_id'])
    mapdl.finish()


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
        np.append(start + (width / 2)*perpendicular, thickness),
        np.append(start - (width / 2)*perpendicular, thickness),
        np.append(end - (width / 2)*perpendicular, thickness),
        np.append(end + (width / 2)*perpendicular, thickness)
    ]
    # Create a volume that, when substracted from the volume defining the 
    # plate, produces the crack analogy, based on the previously defined
    # coordinates
    crack_id = pymodal.mapdl.create_extruded_volume(mapdl, coords, 0.00498)
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
                         Poisson, floor_densities, column_densities, e_size,
                         column_thicknesses, column_widths, floor_width,
                         floor_heights, floor_depth, floor_thicknesses,
                         contact_strengths, mass_coordinates, mass_values):

    # Define a SOLID186 element type for meshing
    element_id = pymodal.mapdl.set_solid186(mapdl)

    floor_id = {}
    floor_mat_id = {}
    
    for i in range(len(list(floor_heights))+1):
        # Define material for current floor
        floor_mat_id[f'floor_{i}'] = pymodal.mapdl.set_linear_elastic(
            mapdl, floor_elastic_moduli[i], Poisson, floor_densities[i]
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
                    mapdl, column_elastic_moduli[col], Poisson,
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
