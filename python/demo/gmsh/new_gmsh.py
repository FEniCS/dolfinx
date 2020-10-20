import gmsh
import meshio
import numpy as np
gmsh.initialize()

gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)


gmsh.model.add("sphere")
sphere_tag = gmsh.model.occ.addSphere(0, 0, 0, 1)
box1_tag = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
box2_tag = gmsh.model.occ.addBox(-2, -0.5, -0.5, 1.5, 0.8, 0.8)

cut = gmsh.model.occ.cut([(3, sphere_tag)], [(3, box1_tag)])
fuse = gmsh.model.occ.fragment(cut[0], [(3, box2_tag)])

gmsh.model.occ.synchronize()


# Get tags for all volumes of the fused geometry
volume_ids = []
for i in range(len(fuse[0])):
    volume_ids.append(fuse[0][i][1])


def create_boundary_tags_from_volume_id(volume_ids):
    """
    Creates unique markers for boundaries of a set of volumes,
    identified by their IDs.
    One unique marker for external boundaries of a volume.
    One marker for internal boundaries between volume ids.
    """
    # Get boundary tags for each volume
    boundary_tags = {i: [] for i in range(1, len(volume_ids) + 1)}
    for i, tag in enumerate(volume_ids):
        boundary = gmsh.model.getBoundary((3, tag))
        for (dim, boundary_id) in boundary:
            if dim == 2:
                boundary_tags[i + 1].append(boundary_id)
    surface_tag = 1
    keys = list(boundary_tags.keys())
    for i in range(len(keys)):
        unique = boundary_tags[keys[i]]
        for j in range(len(keys)):
            # Identify all internal boundaries between volume i and j
            if i != j:
                intersection = np.intersect1d(
                    unique, boundary_tags[keys[j]])
                unique = np.setdiff1d(unique, intersection)

                if len(intersection) > 0 and i < j:
                    gmsh.model.addPhysicalGroup(2, intersection, surface_tag)
                    gmsh.model.setPhysicalName(
                        2, surface_tag,
                        "Interface between Vol{0:d} and Vol{1:d}".
                        format(i, j))
                    surface_tag += 1
        if len(unique) > 0:
            gmsh.model.addPhysicalGroup(2, unique, surface_tag)
            gmsh.model.setPhysicalName(
                2, surface_tag, "Boundary of Vol{0:d}".
                format(i))
            surface_tag += 1


create_boundary_tags_from_volume_id(volume_ids)

# Add physical tags for volumes
gmsh.model.addPhysicalGroup(3, [volume_ids[0], volume_ids[1]], 15)
gmsh.model.setPhysicalName(3, 15, "Sphere cut by Box 1")
gmsh.model.addPhysicalGroup(3, [volume_ids[2]], 9)
gmsh.model.setPhysicalName(3, 9, "Box 2 cut by Sphere")
gmsh.option.setNumber("Mesh.SaveAll", 0)

# Generate the three dimensional mesh
gmsh.model.mesh.generate(3)


# Create meshio mesh
phys_grps = gmsh.model.getPhysicalGroups()

# Sort mesh nodes according to their index in gmsh
idx, points, _ = gmsh.model.mesh.getNodes()
points = points.reshape(-1, 3)
idx -= 1
srt = np.argsort(idx)
assert np.all(idx[srt] == np.arange(len(idx)))
points = points[srt]

meshes = {}
for dim, tag in phys_grps:
    print("Physical name:", gmsh.model.getPhysicalName(dim, tag))
    ent = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
    print("Entities", ent)
    for e in ent:
        element_types, elem_tags, node_tags =\
            gmsh.model.mesh.getElements(dim, e)
        assert(len(element_types) == 1)
        print(element_types)
        element_data = []
        for (e, t, n) in zip(element_types, elem_tags, node_tags):
            name, dim, order, num_nodes, local_coords, numfirstordernodes = \
                gmsh.model.mesh.getElementProperties(e)
            meshio_type = meshio.gmsh.gmsh_to_meshio_type[e]
            print("Celltype", meshio_type)
            if meshio_type in meshes.keys():
                meshes[meshio_type]["cells"] = \
                    np.concatenate((meshes[meshio_type]["cells"],
                                    n.reshape(-1, num_nodes) - 1), axis=0)
                meshes[meshio_type]["cell_data"] = \
                    np.concatenate((meshes[meshio_type]["cell_data"],
                                    np.full(len(t), tag)), axis=0)
            else:
                meshes[meshio_type] = {"cells": n.reshape(-1, num_nodes) - 1,
                                       "cell_data": np.full(len(t), tag)}

for cell_type in meshes.keys():
    mesh = meshio.Mesh(
        points, cells={cell_type: meshes[cell_type]["cells"]},
        cell_data={"Physical Tags {0:s}".format(cell_type):
                   [meshes[cell_type]["cell_data"]]})
    meshio.write("mesh_{0:s}.xdmf".format(cell_type), mesh)


gmsh.write("t1.msh")
