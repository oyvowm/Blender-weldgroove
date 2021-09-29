import bpy

groove = bpy.data.texts["brace.py"].as_module()
material = bpy.data.texts["materials.py"].as_module()
laser_setup = bpy.data.texts["cycles_laser.py"].as_module()
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
    
brace = groove.add_brace_element(45)
leg = groove.add_leg_element()
    
groove.rotate_brace_element(brace, leg, 0.9, 0.01, 0.8, -5)

# 
mat = material.define_material("Galvanized Steel")

# makes the newly defined material the active one
if brace.data.materials:
    brace.data.materials[0] = mat
else:
    brace.data.materials.append(mat)
if leg.data.materials:
    leg.data.materials[0] = mat
else:
    leg.data.materials.append(mat)
    
scanner = laser_setup.LaserSetup("cycles")

scanner.move_laser_to_groove()

bpy.ops.object.select_all(action='DESELECT')

# deletes any pre-existing brace or leg elements
for obj in bpy.data.objects:
    if obj.name[:4] == "Spot":
        obj.select_set(True)

scanner.rotate_laser(-90)

#cycles_laser.move_laser_to_groove(laser)
bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)