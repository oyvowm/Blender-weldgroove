import bpy
import os
# Add in a variable for the file paths later
def define_material(material_name):
    
    # overwrites any existing materials of the same name
    if material_name in bpy.data.materials:
        new_material = bpy.data.materials[material_name]
    else:
        new_material = bpy.data.materials.new(name=material_name)

    new_material.use_nodes = True
    nodes = new_material.node_tree.nodes

    # removes all eventual manually added nodes
    for node in nodes:
        if node.type != "OUTPUT_MATERIAL" and node.type != "BSDF_PRINCIPLED":
            nodes.remove(node)

    # defines variables for the material output and BSDF nodes
    material_output = nodes.get("Material Output")
    BSDF = nodes.get("Principled BSDF")

    # adds image texture nodes
    
    root = "/home/oyvmjo/Blender-weldgroove/pbr/"

    # Base Colour
    base_color = nodes.new(type="ShaderNodeTexImage")
    base_color.location = [-400, 300]
    base_color.image = bpy.data.images.load(os.path.join(root, material_name) + "_BaseColor.png")
    base_color.projection = 'FLAT'
    links = new_material.node_tree.links
    links.new(base_color.outputs[0], BSDF.inputs['Base Color'])

    # Roughness
    Roughness = nodes.new(type="ShaderNodeTexImage")
    Roughness.location = [-400, 40]
    Roughness.image = bpy.data.images.load(os.path.join(root, material_name) + "_Roughness.png")
    Roughness.projection = 'FLAT'
    Roughness.image.colorspace_settings.name = 'Non-Color'
    links.new(Roughness.outputs[0], BSDF.inputs['Roughness'])

    # Normal
    Normal = nodes.new(type="ShaderNodeTexImage")
    Normal.location = [-400, -220]
    Normal.image = bpy.data.images.load(os.path.join(root, material_name) + "_Normal.png")
    Normal.projection = 'FLAT'
    Normal.image.colorspace_settings.name = 'Non-Color'
    links.new(Normal.outputs[0], BSDF.inputs['Normal'])

    # adds a mapper node
    mapper = nodes.new(type="ShaderNodeMapping")
    mapper.location = [base_color.location[0] - 200, base_color.location[1]]
    links.new(mapper.outputs[0], base_color.inputs[0])
    links.new(mapper.outputs[0], Roughness.inputs[0])
    links.new(mapper.outputs[0], Normal.inputs[0])

    # adds a texture coordinate node
    texture_coordinate = nodes.new(type="ShaderNodeTexCoord")
    texture_coordinate.location = [mapper.location[0] - 200, mapper.location[1]]
    links.new(texture_coordinate.outputs[2], mapper.inputs[0])
    
    
    
    return new_material

if __name__ == "__main__":
    mat = define_material("Galvanized Steel")
    
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_linked(delimit={'SEAM'})
    #bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.editmode_toggle()
    
    # makes the newly defined material the active one
    if bpy.context.active_object.data.materials:
        bpy.context.active_object.data.materials[0] = mat
    else:
        bpy.context.active_object.data.materials.append(mat)
    
    