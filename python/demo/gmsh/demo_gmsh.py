# Copyright (C) 2020 Garth N. Wells and Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
#
# =====================================
# Mesh generation using the GMSH python API
# =====================================

from dolfinx.mesh import create_mesh, create_meshtags
from dolfinx.io import XDMFFile, ufl_mesh_from_gmsh
from dolfinx.cpp.io import perm_gmsh, extract_local_entities
from dolfinx import cpp
from mpi4py import MPI
import gmsh
import numpy as np


def gather_gmsh_physical_entities():
    """
    Convenience function that extracts all entitites with
    physical tags, creating mesh data structures for each
    cell type.
    Returns a dictionary where the key is the gmsh cell type
    and its content is a dictionary of cells and cell data.

    """
    phys_grps = gmsh.model.getPhysicalGroups()
    meshes = {}
    for dim, tag in phys_grps:
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        for entity in entities:
            element_types, elem_tags, node_tags =\
                gmsh.model.mesh.getElements(dim, tag=entity)
            for (e_type, e_tags, n_tags) in zip(
                    element_types, elem_tags, node_tags):
                name, dim, order, num_nodes, local_coords, _ = \
                    gmsh.model.mesh.getElementProperties(e_type)
                e_cells = n_tags.reshape(-1, num_nodes) - 1
                if e_type in meshes.keys():
                    meshes[e_type]["cells"] =\
                        np.concatenate((meshes[e_type]["cells"],
                                        e_cells), axis=0)
                    meshes[e_type]["cell_data"] = \
                        np.concatenate((meshes[e_type]["cell_data"],
                                        np.full(len(e_tags), tag)), axis=0)
                else:
                    meshes[e_type] = {"cells":
                                      e_cells,
                                      "cell_data": np.full(len(e_tags), tag)}

    return meshes


def sort_gmsh_points():
    """
    Extracts geometrical nodes from all entities in gmsh
    and sorts them starting from index 0.
    """
    idx, points, _ = gmsh.model.mesh.getNodes()
    points = points.reshape(-1, 3)
    idx -= 1
    srt = np.argsort(idx)
    assert np.all(idx[srt] == np.arange(len(idx)))
    x = points[srt]
    return x


# Generating a mesh on each process rank
# ======================================
#
# Generate a mesh on each rank with the gmsh API, and create a DOLFIN-X mesh
# on each rank

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("Sphere")
gmsh.model.setCurrent("Sphere")

gmsh.model.occ.addSphere(0, 0, 0, 1, tag=1)

# Generate mesh
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)


# Sort mesh nodes according to their index in gmsh (Starts at 1)
x = sort_gmsh_points()

# Extract cells from gmsh (Only interested in tetrahedrons)
element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim=3)
name, dim, order, num_nodes, local_coords, num_first_order_nodes = \
    gmsh.model.mesh.getElementProperties(element_types[0])
cells = node_tags[0].reshape(-1, num_nodes) - 1


mesh = create_mesh(MPI.COMM_SELF, cells, x,
                   ufl_mesh_from_gmsh(element_types[0], x.shape[1]))

with XDMFFile(MPI.COMM_SELF,
              "mesh_rank_{}.xdmf".format(MPI.COMM_WORLD.rank), "w") as file:
    file.write_mesh(mesh)

# Create a distributed (parallel) mesh with affine geometry
# =========================================================
#
# Generate mesh on rank 0, then build a distributed mesh

if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh

    gmsh.model.add("Sphere minus box")
    gmsh.model.setCurrent("Sphere minus box")

    sphere_dim_tags = gmsh.model.occ.addSphere(0, 0, 0, 1)
    box_dim_tags = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    model_dim_tags = gmsh.model.occ.cut(
        [(3, sphere_dim_tags)], [(3, box_dim_tags)])
    gmsh.model.occ.synchronize()

    # Add physical tag 1 for exterior surfaces
    boundary = gmsh.model.getBoundary((3,
                                       model_dim_tags[0][0][1]))
    boundary_ids = [b[1] for b in boundary]
    gmsh.model.addPhysicalGroup(2, boundary_ids, tag=1)
    gmsh.model.setPhysicalName(2, 1, "Sphere surface")

    # Add physical tag 2 for the volume
    volume_entities = [model[1] for model in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, volume_entities, tag=2)
    gmsh.model.setPhysicalName(3, 2, "Sphere volume")

    gmsh.model.mesh.generate(3)

    # Sort mesh nodes according to their index in gmsh
    x = sort_gmsh_points()

    # Broadcast cell type data and geometric dimension
    gmsh_cell_id = MPI.COMM_WORLD.bcast(
        gmsh.model.mesh.getElementType("tetrahedron", 1), root=0)

    # Get mesh data for dim (0, tdim) for all physical entities
    meshes = gather_gmsh_physical_entities()
    cells = meshes[gmsh_cell_id]["cells"]
    cell_data = meshes[gmsh_cell_id]["cell_data"]
    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = gmsh.model.mesh.getElementType("triangle", 1)
    marked_facets = meshes[gmsh_facet_id]["cells"]
    facet_values = meshes[gmsh_facet_id]["cell_data"]
else:
    gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
    num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
    marked_facets, facet_values = np.empty((0, 3)), np.empty((0,))


mesh = create_mesh(MPI.COMM_WORLD, cells, x,
                   ufl_mesh_from_gmsh(gmsh_cell_id, 3))
mesh.name = "ball_d1"
local_entities, local_values = extract_local_entities(
    mesh, 2, marked_facets, facet_values)


mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities),
                     np.int32(local_values))
mt.name = "ball_d1_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
    file.write_mesh(mesh)

    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(
        mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='ball_d1']/Geometry")

# Create a distributed (parallel) mesh with quadratic geometry
# ============================================================
#
# Generate mesh on rank 0, then build a distributed mesh

if MPI.COMM_WORLD.rank == 0:
    # Using gmsh.model.setCurrent(model_name) lets you change between models
    gmsh.model.setCurrent("Sphere minus box")

    # Generate second order mesh and output gmsh messages to terminal
    gmsh.model.mesh.generate(3)
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.mesh.setOrder(2)
    gmsh.option.setNumber("General.Terminal", 0)

    # Sort mesh nodes according to their index in gmsh
    x = sort_gmsh_points()

    # Broadcast cell type data and geometric dimension
    gmsh_cell_id = MPI.COMM_WORLD.bcast(
        gmsh.model.mesh.getElementType("tetrahedron", 2), root=0)

    # Get mesh data for dim (0, tdim) for all physical entities
    meshes = gather_gmsh_physical_entities()
    cells = meshes[gmsh_cell_id]["cells"]
    cell_data = meshes[gmsh_cell_id]["cell_data"]

    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = gmsh.model.mesh.getElementType("triangle", 2)
    marked_facets = meshes[gmsh_facet_id]["cells"]
    facet_values = meshes[gmsh_facet_id]["cell_data"]

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

local_entities, local_values = extract_local_entities(
    mesh, 2, marked_facets, facet_values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2, cpp.graph.AdjacencyList_int32(local_entities),
                     np.int32(local_values))
mt.name = "ball_d2_surface"
with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)
    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(
        mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='ball_d2']/Geometry")

if MPI.COMM_WORLD.rank == 0:
    # Generate a mesh with 2nd-order hexahedral cells using pygmsh
    gmsh.model.add("Hexahedral mesh")
    gmsh.model.setCurrent("Hexahedral mesh")
    # Recombine tetrahedrons to hexahedrons
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    circle = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
    circle_inner = gmsh.model.occ.addDisk(0, 0, 0, 0.5, 0.5)
    cut = gmsh.model.occ.cut([(2, circle)], [(2, circle_inner)])[0]
    extruded_geometry = gmsh.model.occ.extrude(cut, 0, 0, 0.5,
                                               numElements=[5], recombine=True)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [cut[0][1]], tag=1)
    gmsh.model.setPhysicalName(2, 1, "2D cylinder")
    boundary_entities = gmsh.model.getEntities(2)
    other_boundary_entities = []
    for entity in boundary_entities:
        if entity != cut[0][1]:
            other_boundary_entities.append(entity[1])
    gmsh.model.addPhysicalGroup(2, other_boundary_entities, tag=3)
    gmsh.model.setPhysicalName(2, 3, "Remaining boundaries")

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.setOrder(2)
    volume_entities = []
    for entity in extruded_geometry:
        if entity[0] == 3:
            volume_entities.append(entity[1])
    gmsh.model.addPhysicalGroup(3, volume_entities, tag=1)
    gmsh.model.setPhysicalName(3, 1, "Mesh volume")

    # Sort mesh nodes according to their index in gmsh
    x = sort_gmsh_points()

    # Broadcast cell type data and geometric dimension
    gmsh_cell_id = MPI.COMM_WORLD.bcast(
        gmsh.model.mesh.getElementType("hexahedron", 2), root=0)

    # Get mesh data for dim (0, tdim) for all physical entities
    meshes = gather_gmsh_physical_entities()
    cells = meshes[gmsh_cell_id]["cells"]
    cell_data = meshes[gmsh_cell_id]["cell_data"]

    num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
    gmsh_facet_id = gmsh.model.mesh.getElementType("quadrangle", 2)
    marked_facets = meshes[gmsh_facet_id]["cells"]
    facet_values = meshes[gmsh_facet_id]["cell_data"]
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

local_entities, local_values = extract_local_entities(
    mesh, 2, marked_facets, facet_values)
mesh.topology.create_connectivity(2, 0)
mt = create_meshtags(mesh, 2,
                     cpp.graph.AdjacencyList_int32(local_entities),
                     np.int32(local_values))
mt.name = "hex_d2_surface"

with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "a") as file:
    file.write_mesh(mesh)

    mesh.topology.create_connectivity(2, 3)
    file.write_meshtags(
        mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='hex_d2']/Geometry")
