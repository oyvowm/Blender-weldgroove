import bpy

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
    