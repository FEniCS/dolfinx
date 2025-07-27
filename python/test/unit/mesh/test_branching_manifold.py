# Copyright (C) 2025 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import basix
import ufl
from dolfinx.cpp.mesh import create_cell_partitioner
from dolfinx.mesh import (
    CellType,
    GhostMode,
    compute_midpoints,
    create_mesh,
    create_unit_cube,
    create_unit_square,
)


@pytest.mark.parametrize(
    "dim,cell_type",
    [
        (2, CellType.triangle),
        (2, CellType.quadrilateral),
        (3, CellType.hexahedron),
        (3, CellType.tetrahedron),
    ],
)
def test_edge_skeleton_mesh(dim, cell_type):
    """Creates the edge skeleton mesh of a regular unit square/cube and checks for correct
    connectivity information.

    The edge skeleton mesh is the mesh formed by the edges of another mesh (edges -> cell). In
    particular this is a branching mesh.
    """

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        if dim == 2:
            mesh = create_unit_square(MPI.COMM_SELF, 4, 4, cell_type=cell_type)
        else:
            mesh = create_unit_cube(MPI.COMM_SELF, 2, 2, 2, cell_type=cell_type)

        top = mesh.topology
        top.create_connectivity(1, 0)
        e_to_v = top.connectivity(1, 0)
        new_x = mesh.geometry.x[:, :-1] if dim == 2 else mesh.geometry.x
        cells = e_to_v.array.reshape(-1, 2)
    else:
        new_x = np.empty((0, dim), dtype=np.float64)
        cells = np.empty((0, dim), dtype=np.int64)

    element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(dim,)))

    if cell_type == CellType.quadrilateral:
        max_facet_to_cell_links = 4
    elif cell_type == CellType.triangle:
        max_facet_to_cell_links = 6
    elif cell_type == CellType.hexahedron:
        max_facet_to_cell_links = 6
    elif cell_type == CellType.tetrahedron:
        max_facet_to_cell_links = 14

    skeleton_mesh = create_mesh(
        comm,
        cells,
        new_x,
        element,
        create_cell_partitioner(GhostMode.shared_facet, max_facet_to_cell_links),
    )

    skeleton_top = skeleton_mesh.topology
    skeleton_top.create_connectivity(0, 1)
    skeleton_f_to_c = skeleton_top.connectivity(0, 1)

    # debug ouput
    # import febug
    # febug.plot_entity_indices(skeleton_mesh, 1).show() # .save_graphic(f"test_{comm.rank}.svg")

    skeleton_im_f = skeleton_mesh.topology.index_map(0)

    def on_boundary(x):
        return np.any(np.isclose(x[:dim], 0)) or np.any(np.isclose(x[:dim], 1))

    for facet in range(skeleton_im_f.size_local):
        matched = len(skeleton_f_to_c.links(facet)) == max_facet_to_cell_links
        assert matched or on_boundary(skeleton_mesh.geometry.x[facet])


# TODO: tetrahedron
@pytest.mark.parametrize("cell_type", [CellType.hexahedron])
def test_facet_skeleton_mesh(cell_type):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        mesh = create_unit_cube(MPI.COMM_SELF, 3, 3, 3, cell_type=cell_type)

        top = mesh.topology
        top.create_connectivity(2, 0)
        f_to_v = top.connectivity(2, 0)
        new_x = mesh.geometry.x
        cells = f_to_v.array.reshape(-1, 4)  # TODO: cell_type
    else:
        new_x = np.empty((0, 3), dtype=np.float64)
        cells = np.empty((0, 4), dtype=np.int64)

    element = ufl.Mesh(basix.ufl.element("Lagrange", "quadrilateral", 1, shape=(3,)))

    if cell_type == CellType.hexahedron:
        max_facet_to_cell_links = 4

    skeleton_mesh = create_mesh(
        comm,
        cells,
        new_x,
        element,
        create_cell_partitioner(GhostMode.shared_facet, max_facet_to_cell_links),
    )

    skeleton_top = skeleton_mesh.topology

    if comm.size > 1:
        pytest.skip("Branching mesh c->e, e->v connectivity not yet implemented.")

    skeleton_top.create_connectivity(1, 2)
    skeleton_f_to_c = skeleton_top.connectivity(1, 2)

    # import febug
    # febug.plot_entity_indices(mesh, 2).show() # .save_graphic(f"test_{comm.rank}.svg")

    skeleton_im_f = skeleton_mesh.topology.index_map(1)

    def on_boundary(x):
        return np.any(np.isclose(x, 0)) or np.any(np.isclose(x, 1))

    for facet in range(skeleton_im_f.size_local):
        matched = len(skeleton_f_to_c.links(facet)) == max_facet_to_cell_links

        midpoint = compute_midpoints(skeleton_mesh, 1, np.array([facet], dtype=np.int32))[0]
        assert matched or on_boundary(midpoint)
