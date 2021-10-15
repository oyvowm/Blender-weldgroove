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
    while os.path.exists("/home/oyvind/ip/render/" + str(i) + '.npy'):
        i += 1    
    np.save("/home/oyvind/ip/render/" + str(i), bpy.data.objects['Spot'].matrix_world)

def add_handler(func):
    bpy.app.handlers.render_post.clear()
    bpy.app.handlers.render_post.append(func)


def luxcore_scene():
    
    # render engine
    bpy.context.scene.render.engine = 'LUXCORE'
    bpy.context.scene.render.engine = 'LUXCORE'
    
    bpy.context.scene.luxcore.config.path.depth_total = 6
    bpy.context.scene.luxcore.config.path.hybridbackforward_enable = True
    
    bpy.context.scene.luxcore.denoiser.enabled = True
    bpy.context.scene.luxcore.denoiser.type = 'OIDN'
    
    bpy.context.scene.luxcore.halt.enable = True
    bpy.context.scene.luxcore.halt.use_time = True
    bpy.context.scene.luxcore.halt.time = 30

    bpy.context.scene.render.resolution_x = 2448
    bpy.context.scene.render.resolution_y = 2048
    bpy.context.scene.render.resolution_percentage = 80
    
    # keyframes
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 100
    bpy.context.scene.frame_step = 1

    bpy.context.scene.world.luxcore.use_cycles_settings = False
    
    # environment
    bpy.ops.image.open(filepath="//Downloads/industrial_pipe_and_valve_02_4k.exr", directory="/home/oyvind/Downloads/", files=[{"name":"industrial_pipe_and_valve_02_4k.exr", "name":"industrial_pipe_and_valve_02_4k.exr"}], relative_path=True, show_multiview=False)

    # compositor
    compositor()
    
    
def compositor():
    scene = bpy.context.scene
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    
    noisy_img = nodes.new("CompositorNodeOutputFile")
    noisy_img.location = (100,0)
    
    
    
if __name__ == "__main__":
    luxcore_scene()