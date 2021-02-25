import pymodal
import numpy as np


def cantilever_beam(ansys, elastic_modulus, Poisson, density, damage_location,
                    b, h, l, damage_level, ndiv):
    damage_element_start = np.arange(0, l, l / ndiv)[
        int(np.floor(damage_location / (l/ndiv)))
    ]
    damage_element_end = damage_element_start + l / ndiv
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, Poisson,
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


def free_beam(ansys, elastic_modulus, Poisson, density, damage_location, b, h,
              l, damage_level, ndiv):
    damage_element_start = np.arange(0, l, l / ndiv)[
        int(np.floor(damage_location / (l/ndiv)))
    ]
    damage_element_end = damage_element_start + l / ndiv
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, Poisson,
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


def free_plate_shell(ansys, elastic_modulus, Poisson, density, thickness, a, b,
               e_size):
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, Poisson,
                                              density)
    element_id = pymodal.ansys.set_shell181(ansys, thickness, mat_id)
    kp_list = [[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0]]
    area_id = pymodal.ansys.create_area(ansys, kp_list)
    ansys.run('/PREP7')
    ansys.esize(e_size, 0)
    ansys.amesh(area_id['area_id'])
    ansys.finish()


def free_plate_solid(ansys, elastic_modulus, Poisson, density, thickness, a, b,
                     e_size):
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, Poisson,
                                              density)
    element_id = pymodal.ansys.set_solid186(ansys)
    volume_id = pymodal.ansys.create_prism(ansys, 0, 0, a, b, thickness)
    ansys.run('/PREP7')
    ansys.esize(e_size, 0)
    ansys.vmesh(volume_id['volume_id'])
    ansys.finish()


def circ_hole_solid(ansys, elastic_modulus, Poisson, density, thickness, a, b,
                    e_size, center, radius):

    """
    Create the mesh of SOLID186 elements of a rectangular plate with a
    rectangular stringer support united to it with a bonded contact.

    Parameters
    ----------
    ansys : pyansys.mapdl_console.MapdlConsole or 
            pyansys.mapdl_corba.MapdlCorba class object
        An ANSYS MAPDL instance as started by pyansys.launch_mapdl()
    
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
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, Poisson,
                                              density)
    # Define a SOLID186 element type for meshing
    element_id = pymodal.ansys.set_solid186(ansys)
    # Create a rectangular prism volume with the specified measurements
    plate_id = pymodal.ansys.create_prism(ansys, 0, 0, a, b, thickness)
    # Create a cylinder as tall as the plate is thick, with it's center on the
    # specified coordinates.
    hole_id = pymodal.ansys.create_cylinder(ansys, center[0], center[1],
                                            radius, thickness)
    ansys.run('/PREP7')
    # Substract the cylinder from the plate
    ansys.vsbv(plate_id['volume_id'], hole_id['volume_id'])
    ansys.esize(e_size, 0) # Define element size
    ansys.vsweep(hole_id['volume_id'] + 1) # Mesh the volume
    ansys.finish()
    return {'mat_id': mat_id, 'element_id': element_id,
            'plate_id': plate_id['volume_id'], 'hole_id': hole_id['volume_id'],
            'modified_plate_id': hole_id['volume_id'] + 1}


def crack_analogy_solid(ansys, elastic_modulus, Poisson, density, thickness, a,
                        b, e_size, start, end, width):
    
    """
    Create the mesh of SOLID186 elements of a rectangular plate with a
    rectangular stringer support united to it with a bonded contact.

    Parameters
    ----------
    ansys : pyansys.mapdl_console.MapdlConsole or 
            pyansys.mapdl_corba.MapdlCorba class object
        An ANSYS MAPDL instance as started by pyansys.launch_mapdl()
    
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
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, Poisson,
                                              density)
    # Define a SOLID186 element type for meshing
    element_id = pymodal.ansys.set_solid186(ansys)
    # Create a rectangular prism volume with the specified measurements
    plate_id = pymodal.ansys.create_prism(ansys, 0, 0, a, b, thickness)
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
    crack_id = pymodal.ansys.create_extruded_volume(ansys, coords, 0.00498)
    ansys.run('/PREP7')
    # Substract the volume defining the crack from the volume defining the
    # plate.
    ansys.vsbv(plate_id['volume_id'], crack_id['volume_id'])
    ansys.esize(e_size, 0) # Define element size
    ansys.vsweep(crack_id['volume_id'] + 1) # Mesh the volume
    ansys.finish()
    return {'mat_id': mat_id, 'element_id': element_id, 'plate_id': plate_id, 
            'crack': crack_id['volume_id'],
            'modified_plate_id': crack_id['volume_id'] + 1}


def stringer_support_solid(ansys, elastic_modulus, Poisson, density, thickness,
                           a, b, e_size, start, end, width, height):
    
    """
    Create the mesh of SOLID186 elements of a rectangular plate with a
    rectangular stringer support united to it with a bonded contact.

    Parameters
    ----------
    ansys : pyansys.mapdl_console.MapdlConsole or 
            pyansys.mapdl_corba.MapdlCorba class object
        An ANSYS MAPDL instance as started by pyansys.launch_mapdl()
    
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
    mat_id = pymodal.ansys.set_linear_elastic(ansys, elastic_modulus, Poisson,
                                              density)
    # Define a SOLID186 element type for meshing
    element_id = pymodal.ansys.set_solid186(ansys)
    # Create a rectangular prism volume with the specified measurements
    plate_id = pymodal.ansys.create_prism(ansys, 0, 0, a, b, thickness)
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
    stringer_id = pymodal.ansys.create_extruded_volume(ansys, coords, height)
    ansys.run('/PREP7') # Enter preprocessor
    ansys.esize(e_size, 0) # Set element size
    ansys.vmesh('ALL') # Mesh the defined volumes
    # Select the areas based on their location coordinates in the Z axis, only
    # the ones contained in the plane parallel to the Z plane at the value of
    # z = thickness. This plane only contains the lower face of the stringer
    # support and the upper face of the plate.
    ansys.asel('S', 'LOC', 'Z', thickness, thickness)
    ansys.run('/PREP7')
    # Get the IDs for both areas
    ansys.run('*DEL,MAX_PARAM')
    ansys.run('*DEL,MIN_PARAM')
    ansys.get('MAX_PARAM', 'AREA', 0, 'NUM', 'MAX')
    ansys.get('MIN_PARAM', 'AREA', 0, 'NUM', 'MIN')
    ansys.load_parameters()
    area_1 = ansys.parameters['MAX_PARAM']
    area_2 = ansys.parameters['MIN_PARAM']
    # Deifne a bonded contact between both areas
    pymodal.ansys.linear_elastic_surface_contact(ansys, area_1, area_2,
                                                 elastic_modulus*10e7)
    ansys.finish()
    return {'mat_id': mat_id, 'element_id': element_id, 'plate_id': plate_id, 
            'stringer_id': stringer_id['volume_id']}