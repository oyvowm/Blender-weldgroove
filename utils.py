import bpy
import numpy as np


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

def luxcore_scene():
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
    
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 100
    bpy.context.scene.frame_step = 1

    bpy.context.scene.world.luxcore.use_cycles_settings = False
    bpy.ops.image.open(filepath="//Downloads/industrial_pipe_and_valve_02_4k.exr", directory="/home/oyvind/Downloads/", files=[{"name":"industrial_pipe_and_valve_02_4k.exr", "name":"industrial_pipe_and_valve_02_4k.exr"}], relative_path=True, show_multiview=False)

    