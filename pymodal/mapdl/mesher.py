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
    mapdl.run('/PREP7')
    mapdl.esize(l/ndiv, 0)
    mapdl.lmesh(line_start['line_id'])
    mapdl.lmesh(line_end['line_id'])
    element_id_damaged = pymodal.mapdl.set_beam3(
        mapdl, b * h, damage_level * ((b * h**3)/12), h
    )
    mapdl.run('/PREP7')
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
    mapdl.run('/PREP7')
    mapdl.esize(l/ndiv, 0)
    mapdl.lmesh(line_start['line_id'])
    mapdl.lmesh(line_end['line_id'])
    element_id_pristine = pymodal.mapdl.set_beam3(
        mapdl, b * h, damage_level * ((b * h**3)/12), h
    )
    mapdl.run('/PREP7')
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
    mapdl.run('/PREP7')
    mapdl.esize(e_size, 0)
    mapdl.amesh(area_id['area_id'])
    mapdl.finish()


def free_plate_solid(mapdl, elastic_modulus, Poisson, density, thickness, a, b,
                     e_size):
    mat_id = pymodal.mapdl.set_linear_elastic(mapdl, elastic_modulus, Poisson,
                                              density)
    element_id = pymodal.mapdl.set_solid186(mapdl)
    volume_id = pymodal.mapdl.create_prism(mapdl, 0, 0, a, b, thickness)
    mapdl.run('/PREP7')
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
    mapdl.run('/PREP7')
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
    mapdl.run('/PREP7')
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
    mapdl.run('/PREP7') # Enter preprocessor
    mapdl.esize(e_size, 0) # Set element size
    mapdl.vmesh('ALL') # Mesh the defined volumes
    # Select the areas based on their location coordinates in the Z axis, only
    # the ones contained in the plane parallel to the Z plane at the value of
    # z = thickness. This plane only contains the lower face of the stringer
    # support and the upper face of the plate.
    mapdl.asel('S', 'LOC', 'Z', thickness, thickness)
    mapdl.run('/PREP7')
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