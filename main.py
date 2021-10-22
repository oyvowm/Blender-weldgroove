import bpy
import random
import os
import numpy as np
from math import pi, radians


groove = bpy.data.texts["brace.py"].as_module()
material = bpy.data.texts["materials.py"].as_module()
laser_setup = bpy.data.texts["cycles_laser.py"].as_module()
utils = bpy.data.texts["utils.py"].as_module()

# updates scene parameters
utils.luxcore_main_scene()

# adds diffuse material
material.diffuse_material()

# set keyframes to animate
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 3 # 36

if os.path.exists("/home/oyvind/Blender-weldgroove/render/i.npy"):
    iteration = int(np.load("/home/oyvind/Blender-weldgroove/render/i.npy"))
else:
    iteration = 1

stop = 20

if stop - iteration  > 0:
    for i in range(iteration, stop+1):
        
        os.chdir("/home/oyvind/Desktop/blender-2.92.0-linux64/")
        print(os.getcwd())

        # deletes pre-existing objects from the scene
        for obj in bpy.data.objects:
            if obj.name[:4] != "Weld" and obj.name[-3:] != "003" and obj.name[-3:] != "002":
                bpy.data.objects.remove(obj)
        
        # create the weld groove
        groove_angle = np.random.randint(30,46)
        #groove_dist = np.random.choice([0.00, 0.003, 0.01])
        brace_rotation = np.random.randint(-20, 40)
        
        # Using several welds in one file led to crashes, so groove dist and accompanying weld is defined manually
        groove_dist = 0.003
        
        while groove_angle + brace_rotation < 20:
            print("Current values gives no groove opening, sampling new values...")
            groove_angle = np.random.randint(30, 46)
            brace_rotation = np.random.randint(-20, 40)
        
        
        weld_groove = groove.WeldGroove(groove_angle=groove_angle, groove_dist=groove_dist, groove_width=0.4, brace_rotation=brace_rotation)

        ### MATERIAL ###

        # uses the factory method associated with the desired material and render engine
        mat = material.define_material.luxcore_brushed_iron()

        # makes the newly defined material the active one
        if weld_groove.brace.data.materials:
            weld_groove.brace.data.materials[0] = mat.material
        else:
            weld_groove.brace.data.materials.append(mat.material)
        if weld_groove.leg.data.materials:
            weld_groove.leg.data.materials[0] = mat.material
        else:
            weld_groove.leg.data.materials.append(mat.material)

        # apply smart uv projection
        weld_groove.apply_smart_projection(weld_groove.brace)
        weld_groove.apply_smart_projection(weld_groove.leg)

        ### LASER SCANNER SETUP ###

        # adds laser scanner + camera setup    
        scanner = laser_setup.LaserSetup("luxcore", weld_groove.groove_angle + weld_groove.brace_rotation)

        # moves the scanner towards the groove, and returns the angle it needs to be rotated to point towards the groove
        angle = scanner.move_laser_to_groove(x_initial = weld_groove.groove_width/2 - 0.02)

        # rotates the setup to point towards the groove
        scanner.rotate_laser('X', (pi / 2) - angle, 4)

        # move laser + keyframe insertion
        scanner.laser.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_start)

        scanner.move_laser(weld_groove.groove_width - 0.04)

        scanner.laser.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_end)

        # creates directory and saves files to the new directory
        dirname = str(i)
        path = "/home/oyvind/Blender-weldgroove/render/" + dirname
        if not os.path.exists(path):
            os.mkdir("../../Blender-weldgroove/render/" + dirname)
        os.chdir(path)
        nodes = bpy.data.scenes["Scene"].node_tree.nodes
        for node in nodes:
            if node.name[:4] == "File":
                node.base_path = path
        # exr files save to parent folder if using a variable as its base path
        
        print(os.getcwd())
        utils.add_handler(utils.save_laser_matrix)

        # for some reason the scene does not seem to have a camera attached when using denoising, causing the comopositing nodes to fail.
        # therefore, the camera is manually added to the scene objects camera attribute.
        for scene in bpy.data.scenes:
            scene.camera = bpy.data.objects["Camera"]
            
        ### setup scene
        if len(bpy.data.scenes) < 2:
            utils.luxcore_new_scene()
            utils.compositor()
        
        bpy.ops.render.render(animation=True)

        # as the exr files get saved to the wrong folder:
        file_nums = [str(x+1) for x in range(bpy.context.scene.frame_end)]
        for file in file_nums: 
            while len(file) < 4:
                file = '0' + file
            if os.path.exists(path + file + '.exr'):
                os.rename(path + file + '.exr', path + '/' + file + '.exr')

        for i in range(1, bpy.context.scene.frame_end + 1 ):
            i = str(i)
            if os.path.exists("/home/oyvind/Blender-weldgroove/render/" + i + '.npy'):
                os.rename("/home/oyvind/Blender-weldgroove/render/" + i + '.npy', path + '/' + i + '.npy')
        

        bpy.ops.object.select_all(action='DESELECT')
        

        # save the current step
        np.save("/home/oyvind/Blender-weldgroove/render/i", i)
        


else:
        
    # deletes pre-existing objects from the scene
    for obj in bpy.data.objects:
        if obj.name[:4] != "Weld" and obj.name[-3:] != "003" and obj.name[-3:] != "002":
            bpy.data.objects.remove(obj)

    
    
    # create the weld groove
    groove_angle = np.random.randint(30,46)
    #groove_dist = np.random.choice([0.00, 0.003, 0.01])
    brace_rotation = np.random.randint(-20, 40)
    
    # Using several welds in one file led to crashes, so groove dist and accompanying weld is defined manually
    groove_dist = 0.003
    
    while groove_angle + brace_rotation < 20:
        print("Current values gives too small groove opening, sampling new values...")
        groove_angle = np.random.randint(30,46)
        brace_rotation = np.random.randint(-20, 45)
        
    
    weld_groove = groove.WeldGroove(groove_angle=groove_angle, groove_dist=groove_dist, groove_width = 0.4, brace_rotation=brace_rotation)

    ### MATERIAL ###

    # uses the factory method associated with the desired material and render engine
    mat = material.define_material.luxcore_brushed_iron()

    # makes the newly defined material the active one
    if weld_groove.brace.data.materials:
        weld_groove.brace.data.materials[0] = mat.material
    else:
        weld_groove.brace.data.materials.append(mat.material)
    if weld_groove.leg.data.materials:
        weld_groove.leg.data.materials[0] = mat.material
    else:
        weld_groove.leg.data.materials.append(mat.material)

    # apply smart uv projection
    weld_groove.apply_smart_projection(weld_groove.brace)
    weld_groove.apply_smart_projection(weld_groove.leg)

    ### LASER SCANNER SETUP ###

    # adds laser scanner + camera setup    
    scanner = laser_setup.LaserSetup("luxcore", weld_groove.groove_angle + weld_groove.brace_rotation)

    # moves the scanner towards the groove, and returns the angle it needs to be rotated to point towards the groove
    angle = scanner.move_laser_to_groove(x_initial = weld_groove.groove_width/2 - 0.02)

    # rotates the setup to point towards the groove
    scanner.rotate_laser('X', (pi / 2) - angle, 4)
    
    ### setup scene

    # for some reason the scene does not seem to have a camera attached when using denoising, causing the comopositing nodes to fail.
    # therefore, the camera is manually added to the scene objects camera attribute.
    for scene in bpy.data.scenes:
        scene.camera = bpy.data.objects["Camera"]
    
    
    #if len(bpy.data.scenes) < 2:
    utils.luxcore_new_scene()
    utils.compositor()
    
