import bpy
import random
from mathutils import Matrix
from math import radians

class LaserSetup():
    
    def __init__(self, laser_type):
        if laser_type == "cycles":
            self.laser = self.create_cycles_laser()
        
        self.camera = self.add_camera()
        
        self.camera.parent = self.laser
        
    
    
    def create_cycles_laser(self):
        """
        Adds a cycles laser defined using nodes.
        """
        bpy.ops.object.light_add(type='SPOT', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

        laser = bpy.context.active_object
        
        # changes the "spot shape" to 25 degres, should be equivalent to a laser with the same aperture angle 
        laser.data.spot_size = 0.436332
        
        laser.data.use_nodes = True
        nodes = laser.data.node_tree.nodes
        links = laser.data.node_tree.links

        # emission node
        emission = nodes.get("Emission")
        emission.inputs[0].default_value = (1, 0, 0, 1)

        # compare node
        compare = nodes.new(type="ShaderNodeMath")
        compare.location = [-200, 300]
        compare.operation = 'COMPARE'
        compare.inputs[1].default_value = 0
        compare.inputs[2].default_value = 0.001
        links.new(compare.outputs[0], emission.inputs[1])

        # separate xyz node 
        separate_x = nodes.new(type="ShaderNodeSeparateXYZ") 
        separate_x.location = [-400, 300]
        links.new(separate_x.outputs[0], compare.inputs[0])

        # nodes to divide the normal coordinates by their z-value
        vector_divide = nodes.new(type="ShaderNodeVectorMath")
        vector_divide.location = [-600, 100]
        vector_divide.operation = 'DIVIDE'
        links.new(vector_divide.outputs[0], separate_x.inputs[0])

        separate_z = nodes.new(type="ShaderNodeSeparateXYZ")
        separate_z.location = [-800, 300]
        links.new(separate_z.outputs[2], vector_divide.inputs[1])

        texture_coordinates = nodes.new(type="ShaderNodeTexCoord")
        texture_coordinates.location = [-1000, 100]
        links.new(texture_coordinates.outputs[1], separate_z.inputs[0])
        links.new(texture_coordinates.outputs[1], vector_divide.inputs[0])

        # changes the radius value to 0 m
        laser.data.shadow_soft_size = 0

        # change power to 100 W to make it more visible
        laser.data.energy = 100
        
        return laser


    def move_laser_to_groove(self):
        """
        Moves the laser to the groove, using the 3D cursor (assumes that the cursor was moved in brace.py)
        """
        
        # where to place the laser along the width of the groove
        x_location = 0
        
        # moves the laser out of the groove by a random distance
        y_dist = random.uniform(0.06, 0.08)
        z_dist = random.uniform(-0.03, 0.01)
        
        to_cursor = Matrix.Translation(bpy.context.scene.cursor.location) @ Matrix.Translation((x_location, y_dist, z_dist))
        
        self.laser.matrix_world = to_cursor @ self.laser.matrix_world
        
        
    def rotate_laser(self, axis, degrees):
        """
        Rotates the setup as to point towards the groove.
        """
        
        to_rotate = random.uniform(-5, 5)
        rot = Matrix.Rotation(radians(degrees + to_rotate), 4, 'X')
        self.laser.matrix_world = self.laser.matrix_world @ rot
        

    def add_camera(self):
        """
        Adds a camera to the scene.
        """
        
        bpy.ops.object.camera_add(location=(0, 0, 0), scale=(1, 1, 1))

        camera = bpy.context.active_object

        translation = random.uniform(-0.1, -0.06)
        rotation = random.uniform(-20, -40)
        sensor_width = random.uniform(10,15)
        focal_length = random.uniform(15,25)

        # sets the sensor size of the camera
        camera.data.sensor_width = sensor_width

        # sets the focal length of the camera
        camera.data.lens = focal_length

        mat = Matrix.Translation((translation, 0, 0)) @ Matrix.Rotation(radians(rotation), 4, 'Y')

        camera.matrix_world = mat @ camera.matrix_world
        
        return camera



if __name__ == "__main__":
    # deselects all objects
    bpy.ops.object.select_all(action='DESELECT')

    # deletes any pre-existing lights
    for obj in bpy.data.objects:
        if obj.name[:5] == "Light" or obj.name[:4] == "Spot" or obj.name[:6] == "Camera":
            obj.select_set(True)
    bpy.ops.object.delete()
    
   # laser = create_cycles_laser()
    #laser.location[0] = 1
