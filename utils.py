import bpy
import numpy as np
import os

def delete_object(object):
     
# deletes any of the defined objects already in the scene

    for obj in bpy.data.objects:
        if obj.name[:len(object)] == object:
            obj.select_set(True)
    bpy.ops.object.delete()
    
    
def apply_smart_project(object):
    bpy.ops.object.select_all(action='DESELECT')
    object.select_set(True)
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_linked(delimit={'SEAM'})
    #bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.editmode_toggle()
    

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
      
        
def save_laser_matrix(dummy):
    i = 1

    while os.path.exists("/home/oyvind/Blender-weldgroove/render/" + str(i) + '.npy'):
        i += 1
    bpy.context.view_layer.update()
    np.save("/home/oyvind/Blender-weldgroove/render/" + str(i), bpy.data.objects['Spot'].matrix_world)
    print(bpy.data.objects['Spot'].matrix_world)

def add_handler(func):
    bpy.app.handlers.render_post.clear()
    bpy.app.handlers.render_post.append(func)



def luxcore_main_scene(halt_time, engine='bidir'):
    
    main_scene = bpy.data.scenes[0]
    main_scene.world = bpy.data.worlds['World']
    # render engine
    main_scene.render.engine = 'LUXCORE'
    
    if engine == 'bidir':
        main_scene.luxcore.config.device = 'CPU'
        main_scene.luxcore.config.engine = 'BIDIR'
    else:
        main_scene.luxcore.config.device = 'OCL'
        main_scene.luxcore.config.engine = 'PATH'
    
    main_scene.luxcore.config.path.depth_total = 13
    bpy.context.scene.luxcore.config.path.hybridbackforward_enable = True
    
    main_scene.luxcore.denoiser.enabled = True
    main_scene.luxcore.denoiser.type = 'OIDN'
    
    main_scene.luxcore.halt.enable = True
    main_scene.luxcore.halt.use_time = True
    main_scene.luxcore.halt.time = halt_time

    main_scene.render.resolution_x = 2448
    main_scene.render.resolution_y = 2048
    main_scene.render.resolution_percentage = 90    # 80 first 282
    
    ### setup diffuse scene
    luxcore_new_scene(engine)
    compositor()
    
    main_scene.view_layers['View Layer'].luxcore.aovs.position = True

    main_scene.world.luxcore.use_cycles_settings = False
    
    # environment
    
    main_scene.world.luxcore.image = bpy.data.images.load("/home/oyvind/Downloads/industrial_pipe_and_valve_02_4k.exr")
    main_scene.world.luxcore.light = 'infinite'
    

def luxcore_new_scene(engine='bidir'):
    
    if len(bpy.data.scenes) >= 2:
        for i in range(1, len(bpy.data.scenes)):
            bpy.data.scenes.remove(bpy.data.scenes[i])
            
    new_scene = bpy.data.scenes[0].copy()
    new_scene.use_nodes = False

    
    bpy.context.window.scene = new_scene
    
    new_scene.use_nodes = False
    
    if engine == 'bidir':
        new_scene.luxcore.config.device = 'CPU'
        new_scene.luxcore.config.engine = 'BIDIR'
    else:
        new_scene.luxcore.config.device = 'OCL'
        new_scene.luxcore.config.engine = 'PATH'
    
    new_scene.luxcore.halt.time = 2
    
    new_scene.luxcore.config.path.depth_total = 1
    new_scene.luxcore.config.path.depth_diffuse = 1
    new_scene.luxcore.config.path.depth_specular = 1
    new_scene.luxcore.config.path.depth_glossy = 1
    new_scene.luxcore.config.path.hybridbackforward_enable = False
    new_scene.luxcore.halt.time = 2
    
    new_scene.view_layers['View Layer'].luxcore.aovs.direct_diffuse = True
    new_scene.view_layers['View Layer'].luxcore.aovs.position = False
    
    if 'diffuse' in bpy.data.materials.keys():
        new_scene.view_layers['View Layer'].material_override = bpy.data.materials['diffuse']
    else:
        raise Exception('No diffuse material defined.')
        
   # new_scene.world.luxcore.image_user['image'] = None
    
    if len(bpy.data.worlds) > 1:
        for world in bpy.data.worlds:
            if world.name != "World":
                print(f'Deleting {world.name}')
                bpy.data.worlds.remove(world)
    bpy.data.worlds.new('NoHDRI')
    new_scene.world = bpy.data.worlds[0]
    new_scene.world.luxcore.light = 'none'
    
    bpy.context.window.scene = bpy.data.scenes[0]
    

    
    
def compositor():
    scene = bpy.data.scenes[0]
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    links = scene.node_tree.links
    
    for node in nodes:
        if node.name != "Composite":
            nodes.remove(node)
    
    render_layer_mask = nodes.new("CompositorNodeRLayers")
    render_layer_mask.scene = bpy.data.scenes[1]
    render_layer_mask.location = (-100, 700)
    
    render_layer_images = nodes.new("CompositorNodeRLayers")
    render_layer_images.scene = bpy.data.scenes[0]
    render_layer_images.location = (-100, 150)
    
    mask = nodes.new("CompositorNodeOutputFile")
    mask.location = (300,700)
    mask.file_slots.new("mask")
    mask.format.color_mode = 'BW'
    
    position = nodes.new("CompositorNodeOutputFile")
    position.location = (300,-100)
    position.format.file_format = "OPEN_EXR_MULTILAYER"
    
    noisy_img = nodes.new("CompositorNodeOutputFile")
    noisy_img.location = (300,200)
    
    denoised_img = nodes.new("CompositorNodeOutputFile")
    denoised_img.location = (300, 50)
    denoised_img.file_slots.new("Image_Denoised")
    
    
    links.new(render_layer_mask.outputs[3], mask.inputs[1])
    links.new(render_layer_images.outputs[0], nodes["Composite"].inputs[0])
    links.new(render_layer_images.outputs[0], noisy_img.inputs[0])
    links.new(render_layer_images.outputs[2], denoised_img.inputs[1])
    links.new(render_layer_images.outputs[3], position.inputs[0])
    
if __name__ == "__main__":
    luxcore_main_scene()
    luxcore_new_scene()
    compositor()
