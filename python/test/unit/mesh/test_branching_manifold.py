# Copyright (C) 2025 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np

import basix
import ufl
from dolfinx.cpp.mesh import create_cell_partitioner
from dolfinx.mesh import (
    CellType,
    GhostMode,
    create_mesh,
    create_unit_square,
)


def test_edge_skeleton_mesh():
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        mesh = create_unit_square(MPI.COMM_SELF, 4, 4, cell_type=CellType.quadrilateral)
        top = mesh.topology
        top.create_connectivity(1, 0)
        e_to_v = top.connectivity(1, 0)
        new_x = mesh.geometry.x[:, :-1]
        cells = e_to_v.array.reshape(-1, 2)
    else:
        new_x = np.empty((0, 2), dtype=np.float64)
        cells = np.empty((0, 2), dtype=np.int64)

    element = ufl.Mesh(basix.ufl.element("Lagrange", "interval", 1, shape=(2,)))

    skeleton_mesh = create_mesh(
        comm, cells, new_x, element, create_cell_partitioner(GhostMode.shared_facet, 4)
    )

    skeleton_top = skeleton_mesh.topology
    skeleton_top.create_connectivity(0, 1)
    skeleton_f_to_c = skeleton_top.connectivity(0, 1)

    # debug ouput
    # import febug
    # febug.plot_entity_indices(skeleton_mesh, 1).save_graphic(f"test_{comm.rank}.svg")

    skeleton_im_f = skeleton_mesh.topology.index_map(0)

    for facet in range(skeleton_im_f.size_local):
        links = skeleton_f_to_c.links(facet)
        matched = len(links) == 4
        x = skeleton_mesh.geometry.x[facet]
        on_boundary = (
            np.isclose(x[0], 0) or np.isclose(x[0], 1) or np.isclose(x[1], 0) or np.isclose(x[1], 1)
        )
        assert matched or on_boundary
