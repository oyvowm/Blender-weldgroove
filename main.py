import bpy
import random
from math import pi, radians


groove = bpy.data.texts["brace.py"].as_module()
material = bpy.data.texts["materials.py"].as_module()
laser_setup = bpy.data.texts["cycles_laser.py"].as_module()
utils = bpy.data.texts["utils.py"].as_module()
#camera = bpy.data.texts["camera.py"].as_module()


# deletes pre-existing objects from the scene
if len(bpy.data.objects) > 0:
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    # stops the script from deleting the light and camera objects
    
    #for obj in bpy.context.selected_objects:
    #        if obj.name == "Light" or obj.name == "Camera":
    #            obj.select_set(False)
    bpy.ops.object.delete(use_global=False)
    
weld_groove = groove.WeldGroove(groove_angle=45, groove_dist=0.0, brace_rotation=-19)

#brace = groove.add_brace_element(45, 0.01)
#leg = groove.add_leg_element()
#groove.rotate_brace_element(brace, leg, 0.9, 0.01, 0.8, 20)


mat = material.define_material("Galvanized Steel")

# makes the newly defined material the active one
if weld_groove.brace.data.materials:
    weld_groove.brace.data.materials[0] = mat
else:
    weld_groove.brace.data.materials.append(mat)
if weld_groove.leg.data.materials:
    weld_groove.leg.data.materials[0] = mat
else:
    weld_groove.leg.data.materials.append(mat)

# apply smart uv projection
weld_groove.apply_smart_projection(weld_groove.brace)
weld_groove.apply_smart_projection(weld_groove.leg)

# adds laser scanner + camera setup    
scanner = laser_setup.LaserSetup("cycles", weld_groove.groove_angle + weld_groove.brace_rotation)
angle = scanner.move_laser_to_groove()

#bpy.ops.object.select_all(action='DESELECT')

# rotates the setup to point towards the groove
scanner.rotate_laser('X', (pi / 2) - angle, 4)

# rotates the setup a small amount around its local Y-axis to account for it not always aligning perfectly with the braces' normal vector
scanner.rotate_laser('Y', radians(random.uniform(-4, 4)))


bpy.ops.object.select_all(action='DESELECT')

#bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
