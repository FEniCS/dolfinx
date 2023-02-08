# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Mesh generation with Gmsh
#
# Copyright (C) 2020-2022 Garth N. Wells and Jørgen S. Dokken
#
# This demo shows how to create meshes uses the Gmsh Python interface.
# It is implemented in {download}`demo_gmsh.py`.
#
# The required modules are imported. The Gmsh Python module is required.

# +
import sys

try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)


from dolfinx.io import XDMFFile, gmshio

from mpi4py import MPI

# -

# Generate a mesh on each rank with the Gmsh API, and create a DOLFINx
# mesh on each rank with corresponding mesh tags for the cells of the
# mesh.

# +
gmsh.initialize()

# Choose if Gmsh output is verbose
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("Sphere")
model.setCurrent("Sphere")
sphere = model.occ.addSphere(0, 0, 0, 1, tag=1)

# Synchronize OpenCascade representation with gmsh model
model.occ.synchronize()

# Add physical marker for cells. It is important to call this function
# after OpenCascade synchronization
model.add_physical_group(dim=3, tags=[sphere])

# Generate the mesh
model.mesh.generate(dim=3)

# Create a DOLFINx mesh (same mesh on each rank)
msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
msh.name = "Sphere"
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

with XDMFFile(msh.comm, f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_meshtags(cell_markers)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    file.write_meshtags(facet_markers)

# -

# Create a distributed (parallel) mesh with affine geometry. The mesh is
# generated on rank 0, then a distributed DOLFINx mesh is created.
# Create mesh tags on exterior facets.

# +

mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    # Generate a mesh using Gmsh
    model.add("Sphere minus box")
    model.setCurrent("Sphere minus box")

    sphere_dim_tags = model.occ.addSphere(0, 0, 0, 1)
    box_dim_tags = model.occ.addBox(0, 0, 0, 1, 1, 1)
    model_dim_tags = model.occ.cut([(3, sphere_dim_tags)], [(3, box_dim_tags)])
    model.occ.synchronize()

    # Add physical tag 1 for exterior surfaces
    boundary = model.getBoundary(model_dim_tags[0], oriented=False)
    boundary_ids = [b[1] for b in boundary]
    model.addPhysicalGroup(2, boundary_ids, tag=1)
    model.setPhysicalName(2, 1, "Sphere surface")

    # Add physical tag 2 for the volume
    volume_entities = [model[1] for model in model.getEntities(3)]
    model.addPhysicalGroup(3, volume_entities, tag=2)
    model.setPhysicalName(3, 2, "Sphere volume")

    model.mesh.generate(dim=3)

# Create DOLFINx distributed mesh
msh, mt, ft = gmshio.model_to_mesh(model, mesh_comm, model_rank)
msh.name = "ball_d1"
mt.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_facets"

with XDMFFile(msh.comm, "out_gmsh/mesh.xdmf", "w") as file:
    file.write_mesh(msh)
    msh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
    file.write_meshtags(ft, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
# -

# Create a distributed (parallel) mesh with quadratic geometry. Generate
# the Gmsh mesh on rank 0, and then build a distributed DOLFINx mesh.

# +
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    # Using model.setCurrent(name) lets you change between models
    model.setCurrent("Sphere minus box")

    # Generate second order mesh and output gmsh messages to terminal
    model.mesh.generate(3)
    gmsh.option.setNumber("General.Terminal", 1)
    model.mesh.setOrder(2)
    gmsh.option.setNumber("General.Terminal", 0)

msh, ct, ft = gmshio.model_to_mesh(model, mesh_comm, model_rank)
msh.name = "ball_d2"
ct.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_surface"


with XDMFFile(msh.comm, "out_gmsh/mesh.xdmf", "a") as file:
    file.write_mesh(msh)
    file.write_meshtags(ct, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
    file.write_meshtags(ft, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
# -

# Create a distributed (parallel) 2nd order hexahedral mesh. Generate
# the Gmsh mesh on rank 0, then build a distributed DOLFINx mesh.

# +
model_rank = 0
mesh_comm = MPI.COMM_WORLD
if mesh_comm.rank == model_rank:
    # Generate
    model.add("Hexahedral mesh")
    model.setCurrent("Hexahedral mesh")

    # Recombine tetrahedra to hexahedra
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    circle = model.occ.addDisk(0, 0, 0, 1, 1)
    circle_inner = model.occ.addDisk(0, 0, 0, 0.5, 0.5)
    cut = model.occ.cut([(2, circle)], [(2, circle_inner)])[0]
    extruded_geometry = model.occ.extrude(cut, 0, 0, 0.5, numElements=[5], recombine=True)
    model.occ.synchronize()

    model.addPhysicalGroup(2, [cut[0][1]], tag=1)
    model.setPhysicalName(2, 1, "2D cylinder")
    boundary_entities = model.getEntities(2)
    other_boundary_entities = []
    for entity in boundary_entities:
        if entity != cut[0][1]:
            other_boundary_entities.append(entity[1])
    model.addPhysicalGroup(2, other_boundary_entities, tag=3)
    model.setPhysicalName(2, 3, "Remaining boundaries")

    model.mesh.generate(3)
    model.mesh.setOrder(2)
    volume_entities = []
    for entity in extruded_geometry:
        if entity[0] == 3:
            volume_entities.append(entity[1])
    model.addPhysicalGroup(3, volume_entities, tag=1)
    model.setPhysicalName(3, 1, "Mesh volume")

msh, mt, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank)
msh.name = "hex_d2"
mt.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_surface"

with XDMFFile(msh.comm, "out_gmsh/mesh.xdmf", "a") as file:
    file.write_mesh(msh)
    file.write_meshtags(mt, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")
    file.write_meshtags(ft, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry")

# -
