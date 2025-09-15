# Copyright (C) 2022 Michal Habera and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx.cpp.mesh import cell_entity_type, cell_num_entities
from dolfinx.graph import adjacencylist
from dolfinx.io import distribute_entity_data
from dolfinx.mesh import (
    CellType,
    create_unit_cube,
    entities_to_geometry,
    locate_entities,
    meshtags_from_entities,
)
from ufl import Measure

celltypes_3D = [CellType.tetrahedron, CellType.hexahedron]


@pytest.mark.parametrize("cell_type", celltypes_3D)
def test_create(cell_type):
    comm = MPI.COMM_WORLD
    mesh = create_unit_cube(comm, 6, 6, 6, cell_type)

    marked_lines = locate_entities(mesh, 1, lambda x: np.isclose(x[1], 0.5))
    f_v = mesh.topology.connectivity(1, 0).array.reshape(-1, 2)

    entities = adjacencylist(f_v[marked_lines])
    values = np.full(marked_lines.shape[0], 2, dtype=np.int32)
    mt = meshtags_from_entities(mesh, 1, entities, values)

    assert hasattr(mt, "ufl_id")
    assert mt.indices.shape == marked_lines.shape
    assert mt.values.dtype == np.int32
    assert mt.values.shape[0] == entities.num_nodes


def test_ufl_id():
    """Test that UFL can process MeshTags (tests ufl_id attribute)"""
    comm = MPI.COMM_WORLD
    msh = create_unit_cube(comm, 6, 6, 6)
    tdim = msh.topology.dim
    marked_facets = locate_entities(msh, tdim - 1, lambda x: np.isclose(x[1], 1))
    f_v = msh.topology.connectivity(tdim - 1, 0).array.reshape(-1, 3)

    entities = adjacencylist(f_v[marked_facets])
    values = np.full(marked_facets.shape[0], 2, dtype=np.int32)
    ft = meshtags_from_entities(msh, tdim - 1, entities, values)
    ds = Measure("ds", domain=msh, subdomain_data=ft, subdomain_id=(2, 3))
    a = 1 * ds
    assert isinstance(a.subdomain_data(), dict)


def test_distribute_entity_data():
    comm = MPI.COMM_WORLD

    msh = create_unit_cube(comm, 6, 6, 6, cell_type=CellType.tetrahedron)

    def top(x):
        return np.isclose(x[2], 1)

    def bottom(x):
        return np.isclose(x[2], 0)

    # Mark some facets with unique indices
    top_facets = locate_entities(msh, msh.topology.dim - 1, top)
    bottom_facets = locate_entities(msh, msh.topology.dim - 1, bottom)
    top_values = np.full(top_facets.shape[0], 1, dtype=np.int32)
    bottom_values = np.full(bottom_facets.shape[0], 2, dtype=np.int32)

    # Concatenate and sort
    stacked_facets = np.hstack([top_facets, bottom_facets])
    stacked_values = np.hstack([top_values, bottom_values])
    facet_sort = np.argsort(stacked_facets)
    sorted_facets = stacked_facets[facet_sort]
    sorted_values = stacked_values[facet_sort]

    # Convert local facet indices into facets defined by their global
    # vertex indices
    msh.topology.create_connectivity(msh.topology.dim - 1, 0)
    facets_as_vertices = []
    for facet in sorted_facets:
        facets_as_vertices.append(msh.topology.connectivity(msh.topology.dim - 1, 0).array[facet])
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    geom_vertices = entities_to_geometry(msh, msh.topology.dim - 1, np.array(sorted_facets))
    global_input = msh.geometry.input_global_indices[geom_vertices]

    # Gather all global vertices on rank 0 and distribute from there
    input_facets_as_vertices = msh.comm.gather(global_input, root=0)
    input_values = msh.comm.gather(sorted_values, root=0)

    facet_type = cell_entity_type(msh.topology.cell_type, msh.topology.dim - 1, 0)
    num_vertices_per_facet = cell_num_entities(facet_type, 0)
    if msh.comm.rank == 0:
        input_facets_as_vertices = np.vstack(input_facets_as_vertices)
        input_values = np.hstack(input_values)
    else:
        input_facets_as_vertices = np.empty((0, num_vertices_per_facet), dtype=np.int64)
        input_values = np.empty(0, dtype=np.int32)

    new_local_entities, new_values = distribute_entity_data(
        msh, msh.topology.dim - 1, input_facets_as_vertices, input_values
    )

    # Create meshtag from local entities
    new_mt = meshtags_from_entities(
        msh, msh.topology.dim - 1, adjacencylist(new_local_entities), new_values
    )

    np.testing.assert_allclose(new_mt.indices, sorted_facets)
    np.testing.assert_allclose(new_mt.values, sorted_values)
