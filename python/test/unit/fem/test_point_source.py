# Copyright (C) 2025 Paul T. Kühner
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from dolfinx import fem, mesh


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
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_point_source_rank_0(cell_type, ghost_mode, dtype):
    comm = MPI.COMM_WORLD
    rdtype = np.real(dtype(0)).dtype

    msh = None
    cell_dim = mesh.cell_dim(cell_type)
    if cell_dim == 1:
        msh = mesh.create_unit_interval(comm, 4, dtype=rdtype, ghost_mode=ghost_mode)
    elif cell_dim == 2:
        msh = mesh.create_unit_square(
            comm, 4, 4, cell_type=cell_type, dtype=rdtype, ghost_mode=ghost_mode
        )
    elif cell_dim == 3:
        msh = mesh.create_unit_cube(
            comm, 4, 4, 4, cell_type=cell_type, dtype=rdtype, ghost_mode=ghost_mode
        )

    def check(form, coordinate_range, weighted=False):
        a, b = coordinate_range
        weights = np.arange(a, b, dtype=rdtype) if weighted else np.ones(b - a, dtype=rdtype)

        expected_value_l = np.sum(msh.geometry.x[a:b, 0] * weights)
        value_l = fem.assemble_scalar(form)
        assert expected_value_l == pytest.approx(value_l, abs=1e4 * np.finfo(rdtype).eps)

        expected_value = comm.allreduce(expected_value_l)
        value = comm.allreduce(value_l)
        assert expected_value == pytest.approx(value, abs=5e4 * np.finfo(rdtype).eps)

    num_vertices = msh.topology.index_map(0).size_local
    x = ufl.SpatialCoordinate(msh)

    # Full domain
    check(fem.form(x[0] * ufl.dP, dtype=dtype), (0, num_vertices))

    # Split domain into first half of vertices (1) and second half of vertices (2)
    vertices = np.arange(0, msh.topology.index_map(0).size_local, dtype=np.int32)
    tags = np.full_like(vertices, 1)
    tags[tags.size // 2 :] = 2
    meshtags = mesh.meshtags(msh, 0, vertices, tags)
    dP = ufl.Measure("dP", domain=msh, subdomain_data=meshtags)

    check(fem.form(x[0] * dP(1), dtype=dtype), (0, num_vertices // 2))
    check(fem.form(x[0] * dP(2), dtype=dtype), (num_vertices // 2, num_vertices))
    check(fem.form(x[0] * (dP(1) + dP(2)), dtype=dtype), (0, num_vertices))

    V = fem.functionspace(msh, ("P", 1))
    u = fem.Function(V, dtype=dtype)
    u.x.array[:] = np.arange(0, u.x.array.size, dtype=dtype)

    check(fem.form(u * x[0] * ufl.dP, dtype=dtype), (0, num_vertices), weighted=True)
    check(fem.form(u * x[0] * dP(1), dtype=dtype), (0, num_vertices // 2), weighted=True)
    check(fem.form(u * x[0] * dP(2), dtype=dtype), (num_vertices // 2, num_vertices), weighted=True)
    check(fem.form(u * x[0] * (dP(1) + dP(2)), dtype=dtype), (0, num_vertices), weighted=True)


# @pytest.mark.parametrize(
#     "cell_type",
#     [
#         mesh.CellType.interval,
#         mesh.CellType.triangle,
#         mesh.CellType.quadrilateral,
#         mesh.CellType.tetrahedron,
#         # mesh.CellType.pyramid,
#         mesh.CellType.prism,
#         mesh.CellType.hexahedron,
#     ],
# )
# @pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
# @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
# def test_point_source_rank_1(cell_type, ghost_mode, dtype):
#     comm = MPI.COMM_WORLD
#     rdtype = np.real(dtype(0)).dtype

#     msh = None
#     cell_dim = mesh.cell_dim(cell_type)
#     if cell_dim == 1:
#         msh = mesh.create_unit_interval(comm, 4, ghost_mode=ghost_mode, dtype=rdtype)
#     elif cell_dim == 2:
#         msh = mesh.create_unit_square(
#             comm, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
#         )
#     elif cell_dim == 3:
#         msh = mesh.create_unit_cube(
#             comm, 4, 4, 4, cell_type=cell_type, ghost_mode=ghost_mode, dtype=rdtype
#         )

#     num_vertices = msh.topology.index_map(0).size_local

#     def check(form, coordinate_range, weighted=False):
#         a, b = coordinate_range
#         weights = np.arange(a, b, dtype=rdtype) if weighted else np.ones(b - a, dtype=rdtype)
#         expected_value_l = np.zeros(num_vertices, dtype=rdtype)
#         expected_value_l[a:b] = msh.geometry.x[a:b, 0] * weights
#         value_l = fem.assemble_vector(form)
#         equal_l = np.allclose(
#             expected_value_l, np.real(value_l.array[:num_vertices]), atol=1e3 * np.finfo(rdtype).eps
#         )
#         assert equal_l
#         assert comm.allreduce(equal_l, MPI.BAND)

#     x = ufl.SpatialCoordinate(msh)
#     V = fem.functionspace(msh, ("P", 1))
#     v = ufl.conj(ufl.TestFunction(V))

#     # Full domain
#     check(fem.form(x[0] * v * ufl.dP, dtype=dtype), (0, num_vertices))

#     # Split domain into first half of vertices (1) and second half of vertices (2)
#     vertices = np.arange(0, msh.topology.index_map(0).size_local, dtype=np.int32)
#     tags = np.full_like(vertices, 1)
#     tags[tags.size // 2 :] = 2
#     meshtags = mesh.meshtags(msh, 0, vertices, tags)
#     dP = ufl.Measure("dP", domain=msh, subdomain_data=meshtags)

#     check(fem.form(x[0] * v * dP(1), dtype=dtype), (0, num_vertices // 2))
#     check(fem.form(x[0] * v * dP(2), dtype=dtype), (num_vertices // 2, num_vertices))
#     check(fem.form(x[0] * v * (dP(1) + dP(2)), dtype=dtype), (0, num_vertices))

#     V = fem.functionspace(msh, ("P", 1))
#     u = fem.Function(V, dtype=dtype)
#     u.x.array[:] = np.arange(u.x.array.size)

#     check(fem.form(u * x[0] * v * ufl.dP, dtype=dtype), (0, num_vertices), weighted=True)
#     check(fem.form(u * x[0] * v * dP(1), dtype=dtype), (0, num_vertices // 2), weighted=True)
#     check(
#         fem.form(u * x[0] * v * dP(2), dtype=dtype),
#         (num_vertices // 2, num_vertices),
#         weighted=True,
#     )
#     check(fem.form(u * x[0] * v * (dP(1) + dP(2)), dtype=dtype), (0, num_vertices), weighted=True)
