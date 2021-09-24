import bøpy
import bmesh
import math
import numpy as np
from mathutils import Vector, Euler, Matrix, Quaternion

def add_brace_element(groove_angle):
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0), scale=(1, 1, 1))

    brace = bpy.context.active_object

    brace.scale[0] = 0.4
    brace.scale[1] = 0.03

    brace.rotation_euler[1] += math.pi / 2

    brace.location[1] = 0.015
    brace.location[0] = -0.15

    bpy.ops.object.editmode_toggle()

    # creates bmesh for accessing vertices
    bm = bmesh.from_edit_mesh(bpy.context.object.data)
    bm.select_mode = {'VERT'}
    bpy.ops.mesh.select_all(action='DESELECT')

    # selects the vertex to move
    EPSILON = 0.00001
    for v in bm.verts:
        vertex = (brace.matrix_world @ v.co)
        if vertex[1] > 0 and vertex[2] <= 0 + EPSILON:
            v.select = True
            break
       
    to_translate = np.tan(np.radians(groove_angle)) * brace.scale[1]
    print('translate', to_translate)
    bpy.ops.transform.translate(value=(0, 0, to_translate), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    
    # translating along z-axis after transforming the chosen vertex
    brace.location[2] = brace.scale[0] / 2 + 0.03
    
    bpy.ops.mesh.select_all(action='SELECT')

    # finally extrudes the brace object
    bpy.ops.mesh.extrude_context_move(TRANSFORM_OT_translate={"value":(0.3, 0, 0)})
    
    bpy.ops.object.editmode_toggle()
    return brace

def add_leg_element():
    # defining leg object
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0.015))
    leg = bpy.context.active_object
    leg.scale[0] = 0.3
    leg.scale[1] = 0.03
    bpy.ops.transform.rotate(value=np.pi/2, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False))
    bpy.ops.object.editmode_toggle()

    # extruding the leg object as to resemble a cylindrical shape
    bpy.ops.mesh.spin(steps=60, angle=np.pi / 3, center=(0, 0, -0.8), axis=(1, 0, 0))

    bpy.ops.object.editmode_toggle()
    return leg

def rotate_brace_element(brace, leg, y, groove_dist, circle_radius, degrees):
    # translating the brace element
    brace.location[1] += y
    
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

    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)

    bpy.ops.object.editmode_toggle()



'''
# just to ensure that new distance = groove_dist
#rotation_edge2 = [rotation_edge[0] + fraction_to_move * vector_edge_to_cc[0], rotation_edge[1] + fraction_to_move * vector_edge_to_cc[1]]
#vector_edge_to_cc2 = [circle_center[0] - rotation_edge2[0], circle_center[1] - rotation_edge2[1]]
#edge_to_cc2 = np.sqrt((vector_edge_to_cc2[0])**2 + (vector_edge_to_cc2[1])**2)
#new_distance = np.abs(np.abs(circle_center[1]) - edge_to_cc2) - leg.scale[1]
#print(f'new distance: {new_distance}')

#bpy.ops.transform.translate(value=(0, 0.015 + y, 0.23), orient_type='GLOBAL')
'''

if __name__ == "__main__":
    if len(bpy.data.objects) > 0:
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
    
    brace = add_brace_element(45)
    leg = add_leg_element()
    
    rotate_brace_element(brace, leg, 0.9, 0.01, -0.8, -5)
    
    
