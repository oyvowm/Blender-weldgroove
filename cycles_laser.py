import bpy
import random
from mathutils import Matrix
from math import radians, cos, sin, atan, pi
from scipy.stats import truncnorm

class LaserSetup():
    
    def __init__(self, laser_type, groove_angle, min_dist=0.19, max_dist=0.29):
        if laser_type == "cycles":
            self.laser = self.create_cycles_laser()
        
        self.groove_angle = radians(groove_angle)
        self.camera = self.add_camera()
        self.camera.parent = self.laser
        self.min_dist = min_dist
        self.max_dist = max_dist
        
    
    
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
        
        # transformation matrix from origin to cursor location
        to_cursor = Matrix.Translation(bpy.context.scene.cursor.location)
        
        # where to place the laser along the width of the groove
        x_location = 0
        
        # limits how acute the angle between the laser and groove faces can be
        acute = self.groove_angle * 0.1
        
        # moves laser by a random angle drawn from a truncated normal distribution
        
        mean = self.groove_angle / 2
        sigma = self.groove_angle / 4
        lower = 0 + acute
        upper = self.groove_angle - acute
        
        normal = truncnorm((lower - mean) / sigma, (upper - mean) / sigma, loc = mean, scale = sigma)
        
        # samples one number from the distribution
        angle = normal.rvs(1)
        
        direction_vector = (cos(angle), sin(angle))
        
        # moves the laser by a random amount along the direction vector
        length = random.uniform(self.min_dist, self.max_dist)
        from_groove = Matrix.Translation((0, length * direction_vector[0], length * direction_vector[1]))
        
        # transformation from cursor location to an arbitrary point outside the weld groove
        self.laser.matrix_world = to_cursor @ from_groove @ self.laser.matrix_world
        
        # ensure that the laser scanner is moved within the acceptable range
        dist_vec = [self.laser.location[1] - bpy.context.scene.cursor.location[1], self.laser.location[2] - bpy.context.scene.cursor.location[2]]
        dist = (dist_vec[0] ** 2 + dist_vec[1] ** 2) ** 0.5 
        assert self.min_dist < dist < self.max_dist, "Laser scanner not within the allowed distance."
        
        return angle

        
        
    def rotate_laser(self, axis, angle, noise=0):
        """
        Rotates the setup
        """

        rot = Matrix.Rotation((-angle + radians(random.uniform(-noise, noise))), 4, axis)
        self.laser.matrix_world = self.laser.matrix_world @ rot
        

    def add_camera(self):
        """
        Adds a camera to the scene.
        """
        
        bpy.ops.object.camera_add(location=(0, 0, 0), scale=(1, 1, 1))

        camera = bpy.context.active_object
        translation = random.uniform(0.06, 0.1)
        
        mid_point = 0.24
        
        angle = atan(translation / mid_point)
        
        print(angle, 'angle')
        
        # intrinsic parameters
        sensor_width = random.uniform(19,22)
        focal_length = random.uniform(20,25)

        # sets the sensor size of the camera
        camera.data.sensor_width = sensor_width

        # sets the focal length of the camera
        camera.data.lens = focal_length

        mat = Matrix.Rotation(pi, 4, 'Z') @ Matrix.Translation((-translation, 0, 0)) @ Matrix.Rotation(-angle, 4, 'Y')

        camera.matrix_world = camera.matrix_world @ mat
        
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
