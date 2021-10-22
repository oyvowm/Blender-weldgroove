import bpy
import os
# Add in a variable for the file paths later

class define_material():
    def __init__(self, material_name, func):
        #self.material_name = material_name
        root = "/home/oyvind/Blender-weldgroove/pbr/" 
        
        # overwrites any existing materials of the same name
        if material_name in bpy.data.materials:
            new_material = bpy.data.materials[material_name]
        else:
            new_material = bpy.data.materials.new(name=material_name)
            
        self.material = func(new_material, root)
        
    
    @classmethod
    def luxcore_brushed_iron(cls):
        return cls("brushed_iron_01/BrushedIron01_4K", cls.luxcore_material)
    
    @classmethod
    def cycles_brushed_iron(cls):
        return cls("brushed_iron_01/BrushedIron01_4K", cls.cycles_material)


    def luxcore_material(new_material, root):

            
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
        
        base_colour = nodes.new("LuxCoreNodeTexImagemap")
        base_colour.image = bpy.data.images.load(os.path.join(root, new_material.name) + "_BaseColor.png")
        base_colour.location = (-550, 700)
        links.new(base_colour.outputs[0], disney.inputs[0])
        
        metallic = nodes.new("LuxCoreNodeTexImagemap")
        metallic.image = bpy.data.images.load(os.path.join(root, new_material.name) + "_Metallic.png")
        metallic.location = (-550, 250)
        metallic.gamma = 1.0
        links.new(metallic.outputs[0], disney.inputs[2])
        
        roughness = nodes.new("LuxCoreNodeTexImagemap")
        roughness.image = bpy.data.images.load(os.path.join(root, new_material.name) + "_Roughness.png")
        roughness.location = (-550, -200)
        roughness.gamma = 1.0
        links.new(roughness.outputs[0], disney.inputs[5])
        
        normal = nodes.new("LuxCoreNodeTexImagemap")
        normal.image = bpy.data.images.load(os.path.join(root, new_material.name) + "_Normal.png")
        normal.location = (-550, -650)
        normal.is_normal_map = True
        # if the used PBR uses directx normal maps 
        normal.normal_map_orientation = "directx"
        # for some reason the link does not show up when using the shading editor, but it should be connected
        links.new(normal.outputs[0], disney.inputs[-2])
        
        uv_map = nodes.new("LuxCoreNodeTexMapping2D")
        uv_map.uvmap = "UVMap"
        uv_map.location = (-800, 100)
        links.new(uv_map.outputs[0], base_colour.inputs[0])
        links.new(uv_map.outputs[0], metallic.inputs[0])
        links.new(uv_map.outputs[0], roughness.inputs[0])
        links.new(uv_map.outputs[0], normal.inputs[0])
        
        return new_material
            

    def cycles_material(new_material, root):
        

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

        # Base Colour
        base_color = nodes.new(type="ShaderNodeTexImage")
        base_color.location = [-400, 300]
        base_color.image = bpy.data.images.load(os.path.join(root, new_material.name) + "_BaseColor.png")
        base_color.projection = 'FLAT'
        links = new_material.node_tree.links
        links.new(base_color.outputs[0], BSDF.inputs['Base Color'])

        # Roughness
        Roughness = nodes.new(type="ShaderNodeTexImage")
        Roughness.location = [-400, 40]
        Roughness.image = bpy.data.images.load(os.path.join(root, new_material.name) + "_Roughness.png")
        Roughness.projection = 'FLAT'
        Roughness.image.colorspace_settings.name = 'Non-Color'
        links.new(Roughness.outputs[0], BSDF.inputs['Roughness'])

        # Normal
        Normal = nodes.new(type="ShaderNodeTexImage")
        Normal.location = [-400, -220]
        Normal.image = bpy.data.images.load(os.path.join(root, new_material.name) + "_Normal.png")
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

def diffuse_material():
    """
    Creates a 'matte' material to be used for extracting the non-distorted laser line for the mask.
    """
    
    if "diffuse" in bpy.data.materials:
        diffuse = bpy.data.materials["diffuse"]
    else:
        diffuse = bpy.data.materials.new(name="diffuse")
        
    tree_name = "Nodes_" + diffuse.name
    node_tree = bpy.data.node_groups.new(name=tree_name, type="luxcore_material_nodes")
    diffuse.luxcore.node_tree = node_tree

    # User counting does not work reliably with Python PointerProperty.
    # Sometimes, the material this tree is linked to is not counted as user.
    node_tree.use_fake_user = True    
    nodes = node_tree.nodes
    links = node_tree.links
    
    
    output = nodes.new("LuxCoreNodeMatOutput")
    
    matte = nodes.new("LuxCoreNodeMatMatte")
    matte.location = (-300, 0)
        
    links.new(matte.outputs[0], output.inputs[0])
    
    

if __name__ == "__main__":
    mat = luxcore_material("hmm")
    
    # makes the newly defined material the active one
    if bpy.context.active_object.data.materials:
        bpy.context.active_object.data.materials[0] = mat
    else:
        bpy.context.active_object.data.materials.append(mat)
    
    