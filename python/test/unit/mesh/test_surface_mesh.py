# Copyright (C) 2020 Jørgen S. Dokken and Chris Richardson
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
import dolfinx.cpp.mesh as cmesh
import dolfinx.fem
import dolfinx.io
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from mpi4py import MPI


def create_boundary_mesh(mesh, comm, orient=False):
    """
    Create a mesh consisting of all exterior facets of a mesh
    Input:
      mesh   - The mesh
      comm   - The MPI communicator
      orient - Boolean flag for reorientation of facets to have
               consistent outwards-pointing normal (default: True)
    Output:
      bmesh - The boundary mesh
      bmesh_to_geometry - Map from cells of the boundary mesh
                          to the geometry of the original mesh
    """
    ext_facets = cmesh.exterior_facet_indices(mesh)
    boundary_geometry = cmesh.entities_to_geometry(
        mesh, mesh.topology.dim - 1, ext_facets, orient)
    facet_type = dolfinx.cpp.mesh.to_string(cmesh.cell_entity_type(
        mesh.topology.cell_type, mesh.topology.dim - 1))
    facet_cell = ufl.Cell(facet_type,
                          geometric_dimension=mesh.geometry.dim)
    degree = mesh.ufl_domain().ufl_coordinate_element().degree()
    ufl_domain = ufl.Mesh(ufl.VectorElement("Lagrange", facet_cell, degree))
    bmesh = dolfinx.mesh.create_mesh(
        comm, boundary_geometry, mesh.geometry.x, ufl_domain)
    return bmesh, boundary_geometry


@pytest.mark.parametrize("celltype",
                         [cmesh.CellType.tetrahedron,
                          cmesh.CellType.hexahedron])
def test_b_mesh_mapping(celltype):
    """
    Creates a boundary mesh and checks that the geometrical entities
    are mapped to the correct cells.
    """
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2, cell_type=celltype)

    b_mesh, bndry_to_mesh = create_boundary_mesh(mesh, MPI.COMM_SELF)

    # Compute map from boundary mesh topology to boundary mesh geometry
    b_mesh.topology.create_connectivity(
        b_mesh.topology.dim, b_mesh.topology.dim)
    b_imap = b_mesh.topology.index_map(b_mesh.topology.dim)
    tdim_entities = np.arange(b_imap.size_local * b_imap.block_size,
                              dtype=np.int32)
    boundary_geometry = cmesh.entities_to_geometry(
        b_mesh, b_mesh.topology.dim, tdim_entities, False)

    # Compare geometry maps
    for i in range(boundary_geometry.shape[0]):
        assert(
            np.allclose(b_mesh.geometry.x[boundary_geometry[i]],
                        mesh.geometry.x[bndry_to_mesh[i]]))

    # Check that boundary mesh integrated has the correct area
    b_volume = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.Constant(b_mesh, 1) * ufl.dx), op=MPI.SUM)
    mesh_surface = mesh.mpi_comm().allreduce(dolfinx.fem.assemble_scalar(
        dolfinx.Constant(mesh, 1) * ufl.ds), op=MPI.SUM)
    assert(np.isclose(b_volume, mesh_surface))


@pytest.mark.parametrize("celltype",
                         [cmesh.CellType.tetrahedron])
def test_b_mesh_orientation(celltype):
    """
    Test orientation of boundary facets on 3D meshes
    """
    mesh = dolfinx.BoxMesh(MPI.COMM_WORLD,
                           [np.array([-0.5, -0.5, -0.5]),
                            np.array([0.5, 0.5, 0.5])],
                           [2, 2, 2], cell_type=celltype)

    b_mesh, bndry_to_mesh = create_boundary_mesh(mesh, MPI.COMM_SELF, True)
    bdim = b_mesh.topology.dim
    b_mesh.topology.create_connectivity(bdim, bdim - 1)
    b_mesh.topology.create_connectivity(bdim - 1, bdim)
    b_mesh.topology.create_connectivity(bdim, 0)

    num_cells = b_mesh.topology.index_map(bdim).size_local
    num_cell_vertices = \
        cmesh.cell_num_vertices(cmesh.cell_entity_type(celltype, bdim))
    entity_geometry = np.zeros((num_cells, num_cell_vertices),
                               dtype=np.int32)

    xdofs = b_mesh.geometry.dofmap
    for i in range(num_cells):
        xc = xdofs.links(i)
        for j in range(num_cell_vertices):
            entity_geometry[i, j] = xc[j]

    # Compute dot(p0, cross(p1-p0, p2-p0)) for every facet
    # to check that the orientation is correct
    # p0 is vector from centre of mesh
    for i in range(num_cells):
        p = b_mesh.geometry.x[entity_geometry[i, :]]
        p[1] -= p[0]
        p[2] -= p[0]
        assert(np.linalg.det(p) > 0)
