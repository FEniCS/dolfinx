# Copyright (C) 2020 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# =========================================
# Mesh generation using the GMSH python API
# =========================================

import numpy as np
from dolfinx import cpp
from dolfinx.cpp.io import extract_local_entities, perm_gmsh
from dolfinx.io import (XDMFFile, extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import create_mesh, create_meshtags
from mpi4py import MPI

import gmsh

# Generating a mesh on each process rank
# ======================================
#
# Generate a mesh on each rank with the gmsh API, and create a DOLFIN-X mesh
# on each rank
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


mesh = create_mesh(MPI.COMM_SELF, cells, x, ufl_mesh_from_gmsh(element_types[0], x.shape[1]))

with XDMFFile(MPI.COMM_SELF, "mesh_rank_{}.xdmf".format(MPI.COMM_WORLD.rank), "w") as file:
    file.write_mesh(mesh)

# Create a distributed (parallel) mesh with affine geometry
# =========================================================
#
# Generate mesh on rank 0, then build a distributed mesh

if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh

    model.add("Sphere minus box")
    model.setCurrent("Sphere minus box")

    sphere_dim_tags = model.occ.addSphere(0, 0, 0, 1)
    box_dim_tags = model.occ.addBox(0, 0, 0, 1, 1, 1)
    model_dim_tags = model.occ.cut([(3, sphere_dim_tags)], [(3, box_dim_tags)])
    model.occ.synchronize()

    # Add physical tag 1 for exterior surfaces
    boundary = model.getBoundary((3, model_dim_tags[0][0][1]))
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
    marked_facets = topologies[gmsh_facet_id]["topology"]
    facet_values = topologies[gmsh_facet_id]["cell_data"]
else:
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 3)), np.empty((0,))


mesh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, 3))
mesh.name = "ball_d1"
local_entities, local_values = extract_local_entities(mesh, 2, marked_facets, facet_values)


mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "ball_d1_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='ball_d1']/Geometry")

# Create a distributed (parallel) mesh with quadratic geometry
# ============================================================
#
# Generate mesh on rank 0, then build a distributed mesh

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
    marked_facets = topologies[gmsh_facet_id]["topology"]
    facet_values = topologies[gmsh_facet_id]["cell_data"]

else:
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 6)), np.empty((0,))

# Permute the topology from GMSH to DOLFIN-X ordering
domain = ufl_mesh_from_gmsh(gmsh_cell_id, 3)

gmsh_tetra10 = perm_gmsh(cpp.mesh.CellType.tetrahedron, 10)
cells = cells[:, gmsh_tetra10]

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "ball_d2"

# Permute also entities which are tagged
gmsh_triangle6 = perm_gmsh(cpp.mesh.CellType.triangle, 6)
marked_facets = marked_facets[:, gmsh_triangle6]

local_entities, local_values = extract_local_entities(mesh, 2, marked_facets, facet_values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "ball_d2_surface"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(2, 3)
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
    marked_facets = topologies[gmsh_facet_id]["topology"]
    facet_values = topologies[gmsh_facet_id]["cell_data"]
    gmsh.finalize()
else:
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 6)), np.empty((0,))


# Permute the mesh topology from GMSH ordering to DOLFIN-X ordering
domain = ufl_mesh_from_gmsh(gmsh_cell_id, 3)
gmsh_hex27 = perm_gmsh(cpp.mesh.CellType.hexahedron, 27)
cells = cells[:, gmsh_hex27]

mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
mesh.name = "hex_d2"

# Permute also entities which are tagged
gmsh_quad9 = perm_gmsh(cpp.mesh.CellType.quadrilateral, 9)
marked_facets = marked_facets[:, gmsh_quad9]

local_entities, local_values = extract_local_entities(mesh, 2, marked_facets, facet_values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities), np.int32(local_values))
mt.name = "hex_d2_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='hex_d2']/Geometry")
