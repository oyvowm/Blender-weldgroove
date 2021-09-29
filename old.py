def rotate_brace_element(brace, leg, y, groove_dist, circle_radius, degrees):
    """
    Function that translates the brace element along the curvature of the leg element, moves it back onto the leg element, and then rotates it.
    
    Args:
        brace: the brace element
        leg: the leg element
        y (float): the distance to move the brace element along the y-axis
        groove_dist (float): distance between brace- and leg element
        circle_radius (float): radius of imaginary cylider of which leg is a part
        degrees (float): number of degrees to rotate the brace element 
    """
    circle_radius = -circle_radius
    # translating the brace element
    #brace.location[1] += y
    
    # edge to rotate around
    rotation_edge = [y, leg.scale[1]]
    
    # circle center of the leg element
    circle_center = [0, circle_radius]
    
     # vector from edge to circle center
    vector_edge_to_cc = [circle_center[0] - rotation_edge[0], circle_center[1] - rotation_edge[1]]

    # distance between circle center and edge
    edge_to_cc = np.sqrt((vector_edge_to_cc[0])**2 + (vector_edge_to_cc[1])**2)

    # distance from circle to edge
    distance = np.abs(np.abs(circle_center[1]) - edge_to_cc) - leg.scale[1]

    fraction_to_move = (distance - groove_dist) / edge_to_cc

    # translating the brace as to follow the curvature of the leg element
    brace.location[1] += fraction_to_move * vector_edge_to_cc[0]
    brace.location[2] += fraction_to_move * vector_edge_to_cc[1]

    # change the location of the 3D cursor, then set this as the rotation pivot point
    bpy.context.scene.cursor.location = (0.0, brace.location[1] - 0.5 * brace.scale[1], brace.location[2] - 0.5 * brace.scale[0])
    cursor_loc = bpy.context.scene.cursor.location
    
    # composite transformation matrix, translating the object into the coordinate frame of the cursor and rotating
    mat = (Matrix.Translation(cursor_loc) @
           Matrix.Rotation(math.radians(degrees), 4, 'X') @
           Matrix.Translation(-cursor_loc))
           

    # rotating the object around the cursor point and global x-axis
    brace.matrix_basis = mat @ brace.matrix_basis


    # if one wishes to reset the cursor location
    #bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

    
    # just to ensure that new distance = groove_dist
    #rotation_edge2 = [rotation_edge[0] + fraction_to_move * vector_edge_to_cc[0], rotation_edge[1] + fraction_to_move * vector_edge_to_cc[1]]
    #vector_edge_to_cc2 = [circle_center[0] - rotation_edge2[0], circle_center[1] - rotation_edge2[1]]
    #edge_to_cc2 = np.sqrt((vector_edge_to_cc2[0])**2 + (vector_edge_to_cc2[1])**2)
    #new_distance = np.abs(np.abs(circle_center[1]) - edge_to_cc2) - leg.scale[1]
    #print(f'new distance: {new_distance}')

    #bpy.ops.transform.translate(value=(0, 0.015 + y, 0.23), orient_type='GLOBAL')