import bpy
import random
from mathutils import Matrix
from math import radians


def add_camera():
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

    # deletes any pre-existing cameras
    for obj in bpy.data.objects:
        if obj.name[:6] == "Camera":
            obj.select_set(True)
    bpy.ops.object.delete()
    
    camera = add_camera()