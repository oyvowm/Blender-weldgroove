import bpy

#galvanized_steel = bpy.data.materials.new(name="Galvanized Steel")
galvanized_steel = bpy.context.active_object.active_material
 
galvanized_steel.use_nodes = True
nodes = galvanized_steel.node_tree.nodes

# removes all manually added nodes
for node in nodes:
    if node.type != "OUTPUT_MATERIAL" and node.type != "BSDF_PRINCIPLED":
        nodes.remove(node)

material_output = nodes.get("Material Output")
BSDF = nodes.get("Principled BSDF")

baseColor = nodes.new(type="ShaderNodeTexImage")
baseColor.location = [-400, 300]
baseColor.image = bpy.data.images.load("C:\\Users\\oyvin\\Documents\\blender\\galvanized_steel\\GalvanizedSteel01_4K_BaseColor.png")

links = galvanized_steel.node_tree.links
links.new(baseColor.outputs[0], BSDF.inputs[0])

mapper = nodes.new(type="ShaderNodeMapping")
mapper.location = [baseColor.location[0] - 200, baseColor.location[1]]

texture_coordinate = nodes.new(type="ShaderNodeTexCoord")

texture_coordinate.location = [mapper.location[0] - 200, mapper.location[1]]