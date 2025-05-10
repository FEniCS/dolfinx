# Copyright (C) 2025 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import default_scalar_type, fem, mesh


@pytest.mark.parametrize(
    "cell_type",
    [
        mesh.CellType.interval,
        mesh.CellType.triangle,
        mesh.CellType.quadrilateral,
        mesh.CellType.tetrahedron,
        # mesh.CellType.pyramid,
        mesh.CellType.prism,
        mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
def test_point_source_rank_0_full_domain_1D(cell_type, ghost_mode):
    comm = MPI.COMM_WORLD

    msh = None
    cell_dim = mesh.cell_dim(cell_type)
    if cell_dim == 1:
        msh = mesh.create_unit_interval(comm, 4, dtype=default_scalar_type, ghost_mode=ghost_mode)
    elif cell_dim == 2:
        msh = mesh.create_unit_square(
            comm, 4, 4, cell_type=cell_type, dtype=default_scalar_type, ghost_mode=ghost_mode
        )
    elif cell_dim == 3:
        msh = mesh.create_unit_cube(
            comm, 4, 4, 4, cell_type=cell_type, dtype=default_scalar_type, ghost_mode=ghost_mode
        )

    x = ufl.SpatialCoordinate(msh)
    F = fem.form(x[0] * ufl.dP)

    expected_value_l = np.sum(msh.geometry.x[: msh.topology.index_map(0).size_local, 0])
    value_l = fem.assemble_scalar(F)
    assert expected_value_l == pytest.approx(value_l)

    expected_value = comm.allreduce(expected_value_l)
    value = comm.allreduce(value_l)
    assert expected_value == pytest.approx(value)

    # Split domain into first half of vertices (1) and second half of vertices (2)
    vertices = np.arange(0, msh.topology.index_map(0).size_local, dtype=np.int32)
    tags = np.full_like(vertices, 1)
    tags[tags.size // 2 :] = 2
    meshtags = mesh.meshtags(msh, 0, vertices, tags)

    # Test dp(1)
    dP = ufl.Measure("dP", domain=msh, subdomain_data=meshtags)
    F = fem.form(x[0] * dP(1))
    expected_value_l = np.sum(msh.geometry.x[: msh.topology.index_map(0).size_local // 2, 0])
    value_l = fem.assemble_scalar(F)
    assert expected_value_l == pytest.approx(value_l)

    expected_value = comm.allreduce(expected_value_l)
    value = comm.allreduce(value_l)
    assert expected_value == pytest.approx(value)

    # Test dp(2)
    F = fem.form(x[0] * dP(2))
    expected_value_l = np.sum(
        msh.geometry.x[
            msh.topology.index_map(0).size_local // 2 : msh.topology.index_map(0).size_local, 0
        ]
    )
    value_l = fem.assemble_scalar(F)
    assert expected_value_l == pytest.approx(value_l)

    expected_value = comm.allreduce(expected_value_l)
    value = comm.allreduce(value_l)
    assert expected_value == pytest.approx(value)

    # TODO: failing
    # Test dp(1) + dp(2)
    # F = fem.form(x[0] * (dP(1) + dP(2)))
    # expected_value_l = np.sum(msh.geometry.x[:msh.topology.index_map(0).size_local, 0])
    # value_l = fem.assemble_scalar(F)
    # assert expected_value_l == pytest.approx(value_l)

    # expected_value = comm.allreduce(expected_value_l)
    # value = comm.allreduce(value_l)
    # assert expected_value == pytest.approx(value)
