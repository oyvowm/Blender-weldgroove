import bpy
import bmesh
import math
import numpy as np
from mathutils import Vector, Euler, Matrix, Quaternion



class WeldGroove():
    
    def __init__(self, groove_angle, groove_dist=0.01, brace_height=0.4, element_thickness=0.03, 
                 groove_width=0.3, circle_radius=0.8, brace_rotation=20):
        
        assert groove_angle + brace_rotation > 0, "The given values gives no groove opening"
        
        self.groove_angle = groove_angle
        self.brace_rotation = brace_rotation
        
        self.brace = self.add_brace_element(groove_angle, groove_width, groove_dist, brace_height, element_thickness)
        self.leg = self.add_leg_element(groove_width, element_thickness)
        self.rotation_edge = self.rotate_brace_element(groove_dist, circle_radius, self.brace_rotation, brace_height)
        
    def add_brace_element(self, groove_angle, groove_width, groove_dist, brace_height, brace_thickness):
        """
        Adds a brace element to the scene
        
        Args:
            groove_angle (int): the groove angle of the brace element
        """
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0), scale=(1, 1, 1))

        brace = bpy.context.active_object

        brace.scale[0] = brace_height
        brace.scale[1] = brace_thickness

        brace.rotation_euler[1] += math.pi / 2

        brace.location[1] = brace_thickness / 2
        brace.location[0] = - groove_width / 2

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
        bpy.ops.transform.translate(value=(0, 0, to_translate), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
        
        # translating along z-axis after transforming the chosen vertex
        brace.location[2] = brace_height / 2 + brace_thickness + groove_dist
        
        bpy.ops.mesh.select_all(action='SELECT')

        # finally extrudes the brace object
        bpy.ops.mesh.extrude_context_move(TRANSFORM_OT_translate={"value":(0.3, 0, 0)})
        
        bpy.ops.object.editmode_toggle()
        
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return brace

    def add_leg_element(self, groove_width, leg_thickness):
        """
        Adds a leg element to the scene.
        """
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0.015))
        leg = bpy.context.active_object
        leg.scale[0] = groove_width
        leg.scale[1] = leg_thickness
        bpy.ops.transform.rotate(value=np.pi/2, orient_axis='X', orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(True, False, False))
        bpy.ops.object.editmode_toggle()

        # extruding the leg object as to resemble a cylindrical shape
        bpy.ops.mesh.spin(steps=60, angle=np.pi / 5, center=(0, 0, -0.8), axis=(1, 0, 0))

        bpy.ops.object.editmode_toggle()
        
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return leg

    def rotate_brace_element(self, groove_dist, circle_radius, degrees, brace_height):
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
        rotation_edge = self.brace.location[2] - brace_height / 2
        
        print(rotation_edge, 'Rotation edge')
        # circle center of the leg element
        circle_center = [0, circle_radius]
        
        
        # change the location of the 3D cursor
        bpy.context.scene.cursor.location = (0.0, 0.0, rotation_edge)
        cursor_loc = bpy.context.scene.cursor.location
        
        # composite transformation matrix, translating the object into the coordinate frame of the cursor and rotating
        mat = (Matrix.Translation(cursor_loc) @
               Matrix.Rotation(math.radians(degrees), 4, 'X') @
               Matrix.Translation(-cursor_loc))
               

        # rotating the object around the cursor point and global x-axis
        self.brace.matrix_basis = mat @ self.brace.matrix_basis


        # if one wishes to reset the cursor location
        #bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
        return cursor_loc
    
    def apply_smart_projection(self, object):
        bpy.ops.object.select_all(action='DESELECT')
        object.select_set(True)
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_linked(delimit={'SEAM'})
        #bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()
        bpy.ops.object.editmode_toggle()


if __name__ == "__main__":
    
    # deselects all objects
    bpy.ops.object.select_all(action='DESELECT')

    # deletes any pre-existing brace or leg elements
    for obj in bpy.data.objects:
        if obj.name[:5] == "Plane":
            obj.select_set(True)
    bpy.ops.object.delete()
    
    
    brace = add_brace_element(45)
    leg = add_leg_element()
    
    rotate_brace_element(brace, leg, 0.8, 0.01, 0.8, 45)
    
    
