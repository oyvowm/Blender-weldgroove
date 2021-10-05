import bpy
import os
# Add in a variable for the file paths later
'''
class define_material():
    def __init__(self, material_name, func):
        #self.material_name = material_name
        self.root = "/home/oyvind/Blender-weldgroove/pbr/" 
        
        # overwrites any existing materials of the same name
        if material_name in bpy.data.materials:
            new_material = bpy.data.materials[material_name]
        else:
            new_material = bpy.data.materials.new(name=material_name)
            
        self.mat = func(material_name, root)

'''     

def luxcore(material_name):
    # overwrites any existing materials of the same name
    if material_name in bpy.data.materials:
        new_material = bpy.data.materials[material_name]
    else:
        new_material = bpy.data.materials.new(name=material_name)
        
    tree_name = "Nodes_" + new_material.name
    node_tree = bpy.data.node_groups.new(name=tree_name, type="luxcore_material_nodes")
    new_material.luxcore.node_tree = node_tree

    # User counting does not work reliably with Python PointerProperty.
    # Sometimes, the material this tree is linked to is not counted as user.
    node_tree.use_fake_user = True    

    nodes = node_tree.nodes
    
    # output node
    output = nodes.new("LuxCoreNodeMatOutput")
    
    # "disney" node
    disney = nodes.new("LuxCoreNodeMatDisney")
    disney.location = (-290, 100)
    links = node_tree.links
    links.new(disney.outputs[0], output.inputs[0])

    # image texture nodes
    root = "/home/oyvind/Blender-weldgroove/pbr/"
    
    base_colour = nodes.new("LuxCoreNodeTexImagemap")
    base_colour.image = bpy.data.images.load(os.path.join(root, "brushed_iron_01") + "/BrushedIron01_4K_BaseColor.png")
    base_colour.location = (-550, 700)
    links.new(base_colour.outputs[0], disney.inputs[0])
    
    roughness = nodes.new("LuxCoreNodeTexImagemap")
    roughness.image = bpy.data.images.load(os.path.join(root, "brushed_iron_01") + "/BrushedIron01_4K_Roughness.png")
    roughness.location = (-550, 250)
    roughness.gamma = 1.0
    links.new(roughness.outputs[0], disney.inputs[5])
    
    normal = nodes.new("LuxCoreNodeTexImagemap")
    normal.image  =bpy.data.images.load(os.path.join(root, "brushed_iron_01") + "/BrushedIron01_4K_Normal.png")
    normal.location = (-550, -200)
    normal.gamma = 1.0
    links.new(normal.outputs[0], disney.inputs[-2])
'''
node_tree.links.new(glossy2.outputs[0], output.inputs[0])

# Load image, for docs on this see https://docs.blender.org/api/current/bpy_extras.image_utils.html#module-bpy_extras.image_utils
import bpy_extras
diffuse_img = bpy_extras.image_utils.load_image("/home/simon/Downloads/diamond_CPU.jpg")

# Create imagemap node
diffuse_img_node = nodes.new("LuxCoreNodeTexImagemap")
diffuse_img_node.location = -200, 200
diffuse_img_node.image = diffuse_img
node_tree.links.new(diffuse_img_node.outputs[0], glossy2.inputs["Diffuse Color"])

###############################################################

# Assign to object (if you want)
obj = bpy.context.active_object
if obj.material_slots:
    obj.material_slots[obj.active_material_index].material = mat
else:
    obj.data.materials.append(mat)

# For viewport render, we have to update the luxcore object
# because the newly created material is not yet assigned there
obj.update_tag()        ''' 
            
            

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
    
    root = "/home/oyvind/Blender-weldgroove/pbr/"

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
    mat = luxcore("hmm")
    
    # makes the newly defined material the active one
    if bpy.context.active_object.data.materials:
        bpy.context.active_object.data.materials[0] = mat
    else:
        bpy.context.active_object.data.materials.append(mat)
    
    