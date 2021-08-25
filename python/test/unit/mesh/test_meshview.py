# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfinx
from mpi4py import MPI
import numpy as np
import ufl


def test_meshview():
    mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 4, 2, 3)
    dim = mesh.topology.dim

    cells = dolfinx.mesh.locate_entities(mesh, dim, lambda x: x[0] <= 0.5)

    mv = dolfinx.cpp.mesh.MeshView(mesh, dim, cells)

    parent_dofmap_g = mesh.geometry.dofmap
    child_dofmap = mv.geometry_dofmap
    parent_cells = mv.parent_entities

    for i, cell in enumerate(parent_cells):
        assert np.allclose(parent_dofmap_g.links(cell), child_dofmap.links(i))

    e_to_v = mesh.topology.connectivity(dim, 0)
    e_to_v_mv = mv.topology.connectivity(dim, 0)

    for i in range(e_to_v_mv.num_nodes):
        assert np.allclose(e_to_v.links(parent_cells[i]), mv.parent_vertices[e_to_v_mv.links(i)])


def test_meshview_facets():
    """
    Test that facet geometry dofmap is extracted correctly when creating a mesh view
    """
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 4, 2)
    dim = mesh.topology.dim - 1

    facets = dolfinx.mesh.locate_entities_boundary(mesh, dim, lambda x: x[0] <= 0.5)

    mv = dolfinx.cpp.mesh.MeshView(mesh, dim, facets)
    child_dofmap = mv.geometry_dofmap
    parent_facets = mv.parent_entities

    geom_entities = dolfinx.cpp.mesh.entities_to_geometry(mesh, dim, parent_facets, False)
    for i in range(len(parent_facets)):
        assert np.allclose(geom_entities[i], child_dofmap.links(i))

    e_to_v = mesh.topology.connectivity(dim, 0)
    e_to_v_mv = mv.topology.connectivity(dim, 0)

    for i in range(e_to_v_mv.num_nodes):
        assert np.allclose(e_to_v.links(parent_facets[i]), mv.parent_vertices[e_to_v_mv.links(i)])
