# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# (demo-gmsh)=
#
# # Mesh generation with Gmsh
#
# Copyright (C) 2020 Garth N. Wells and Jørgen S. Dokken

# +
import sys

try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)

import numpy as np

from dolfinx.graph import create_adjacencylist
from dolfinx.io import (XDMFFile, cell_perm_gmsh, distribute_entity_data,
                        extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import CellType, create_mesh, meshtags_from_entities

from mpi4py import MPI

# -

# Generate a mesh on each rank with the gmsh API, and create a DOLFINx
# mesh on each rank.

# +
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("Sphere")
model.setCurrent("Sphere")
model.occ.addSphere(0, 0, 0, 1, tag=1)

# Generate mesh
model.occ.synchronize()
model.mesh.generate(3)


# Sort mesh nodes according to their index in gmsh (Starts at 1)
x = extract_gmsh_geometry(model, model_name="Sphere")

# Extract cells from gmsh (Only interested in tetrahedrons)
element_types, element_tags, node_tags = model.mesh.getElements(dim=3)
assert len(element_types) == 1
name, dim, order, num_nodes, local_coords, num_first_order_nodes = model.mesh.getElementProperties(element_types[0])
cells = node_tags[0].reshape(-1, num_nodes) - 1

msh = create_mesh(MPI.COMM_SELF, cells, x, ufl_mesh_from_gmsh(element_types[0], x.shape[1]))

with XDMFFile(MPI.COMM_SELF, f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
    file.write_mesh(msh)
# -

# Create a distributed (parallel) mesh with affine geometry. Generate
# mesh on rank 0, then build a distributed mesh

# +
if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh

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

    model.mesh.generate(3)

    # Sort mesh nodes according to their index in gmsh
    x = extract_gmsh_geometry(model, model_name="Sphere minus box")

    # Broadcast cell type data and geometric dimension
    gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("tetrahedron", 1), root=0)

    # Get mesh data for dim (0, tdim) for all physical entities
    topologies = extract_gmsh_topology_and_markers(model, "Sphere minus box")
    cells = topologies[gmsh_cell_id]["topology"]
    cell_data = topologies[gmsh_cell_id]["cell_data"]
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = model.mesh.getElementType("triangle", 1)
    marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
    facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)
else:
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)


msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, 3))
msh.name = "ball_d1"
entities, values = distribute_entity_data(msh, 2, marked_facets, facet_values)

msh.topology.create_connectivity(2, 0)
mt = meshtags_from_entities(msh, 2, create_adjacencylist(entities), values)
mt.name = "ball_d1_surface"

with XDMFFile(MPI.COMM_WORLD, "out_gmsh/mesh.xdmf", "w") as file:
    file.write_mesh(msh)
    msh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='ball_d1']/Geometry")
# -

# Create a distributed (parallel) mesh with quadratic geometry. Generate
# mesh on rank 0, then build a distributed mesh.

# +
if MPI.COMM_WORLD.rank == 0:
    # Using model.setCurrent(model_name) lets you change between models
    model.setCurrent("Sphere minus box")

    # Generate second order mesh and output gmsh messages to terminal
    model.mesh.generate(3)
    gmsh.option.setNumber("General.Terminal", 1)
    model.mesh.setOrder(2)
    gmsh.option.setNumber("General.Terminal", 0)

    # Sort mesh nodes according to their index in gmsh
    x = extract_gmsh_geometry(model, model.getCurrent())

    # Broadcast cell type data and geometric dimension
    gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("tetrahedron", 2), root=0)

    # Get mesh data for dim (0, tdim) for all physical entities
    topologies = extract_gmsh_topology_and_markers(model, model.getCurrent())
    cells = topologies[gmsh_cell_id]["topology"]
    cell_data = topologies[gmsh_cell_id]["cell_data"]

    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = model.mesh.getElementType("triangle", 2)
    marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
    facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)

else:
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 6)).astype(np.int64), np.empty((0,)).astype(np.int32)

# Permute the topology from GMSH to DOLFINx ordering
domain = ufl_mesh_from_gmsh(gmsh_cell_id, 3)

gmsh_tetra10 = cell_perm_gmsh(CellType.tetrahedron, 10)
cells = cells[:, gmsh_tetra10]

msh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
msh.name = "ball_d2"

# Permute also entities which are tagged
gmsh_triangle6 = cell_perm_gmsh(CellType.triangle, 6)
marked_facets = marked_facets[:, gmsh_triangle6]

entities, values = distribute_entity_data(msh, 2, marked_facets, facet_values)
msh.topology.create_connectivity(2, 0)
mt = meshtags_from_entities(msh, 2, create_adjacencylist(entities), values)
mt.name = "ball_d2_surface"
with XDMFFile(MPI.COMM_WORLD, "out_gmsh/mesh.xdmf", "a") as file:
    file.write_mesh(msh)
    msh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='ball_d2']/Geometry")

if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh with 2nd-order hexahedral cells using gmsh
    model.add("Hexahedral mesh")
    model.setCurrent("Hexahedral mesh")
    # Recombine tetrahedrons to hexahedrons
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

    # Sort mesh nodes according to their index in gmsh
    x = extract_gmsh_geometry(model, model.getCurrent())

    # Broadcast cell type data and geometric dimension
    gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("hexahedron", 2), root=0)

    # Get mesh data for dim (0, tdim) for all physical entities
    topologies = extract_gmsh_topology_and_markers(model, model.getCurrent())
    cells = topologies[gmsh_cell_id]["topology"]
    cell_data = topologies[gmsh_cell_id]["cell_data"]

    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = model.mesh.getElementType("quadrangle", 2)
    marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
    facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)
    gmsh.finalize()

    # Permute tagged entities
    gmsh_quad9 = cell_perm_gmsh(CellType.quadrilateral, 9)
    marked_facets = marked_facets[:, gmsh_quad9]
else:
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 9)).astype(np.int64), np.empty((0,)).astype(np.int32)

# Permute the mesh topology from GMSH ordering to DOLFINx ordering
domain = ufl_mesh_from_gmsh(gmsh_cell_id, 3)
gmsh_hex27 = cell_perm_gmsh(CellType.hexahedron, 27)
cells = cells[:, gmsh_hex27]

msh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
msh.name = "hex_d2"

entities, values = distribute_entity_data(msh, 2, marked_facets, facet_values)
msh.topology.create_connectivity(2, 0)
mt = meshtags_from_entities(msh, 2, create_adjacencylist(entities), values)
mt.name = "hex_d2_surface"

with XDMFFile(MPI.COMM_WORLD, "out_gmsh/mesh.xdmf", "a") as file:
    file.write_mesh(msh)
    msh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='hex_d2']/Geometry")
