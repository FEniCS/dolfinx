# Copyright (C) 2026 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Tests for the transpose of non-matching interpolation."""

from mpi4py import MPI

import numpy as np
import pytest

import basix
import basix.ufl
from dolfinx import fem, mesh
from dolfinx.graph import partitioner

dtypes = [np.float32, np.float64, np.complex64, np.complex128]


def _real_dtype(dtype: np.dtype) -> np.dtype:
    return np.real(np.zeros(1, dtype=dtype)).dtype


def _random_function(V: fem.FunctionSpace, dtype: np.dtype, seed: int) -> fem.Function:
    rng = np.random.default_rng(seed + MPI.COMM_WORLD.rank)
    u = fem.Function(V, dtype=dtype)
    n = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    values = rng.uniform(-1, 1, n)
    if np.issubdtype(dtype, np.complexfloating):
        values = values + 1j * rng.uniform(-1, 1, n)
    u.x.array[:n] = values
    u.x.scatter_forward()
    return u


def _owned_dot(u: fem.Function, v: fem.Function) -> complex:
    """Bilinear (unconjugated) product of the owned coefficients."""
    V = u.function_space
    n = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    return MPI.COMM_WORLD.allreduce(np.sum(u.x.array[:n] * v.x.array[:n]), op=MPI.SUM)


def _owned_norm(u: fem.Function) -> float:
    V = u.function_space
    n = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    return np.sqrt(MPI.COMM_WORLD.allreduce(np.sum(np.abs(u.x.array[:n]) ** 2), op=MPI.SUM))


def _check_adjoint(V0, V1, cells1, data, dtype):
    """Check <P a, b> = <a, P^T b> for random a in V0 and b in V1."""
    a = _random_function(V0, dtype, 1)
    b = _random_function(V1, dtype, 2)

    Pa = fem.Function(V1, dtype=dtype)
    Pa.interpolate_nonmatching(a, cells1, data)
    PTb = _random_function(V0, dtype, 3)  # Overwritten, including ghosts
    PTb.interpolate_nonmatching_transpose(b, cells1, data)

    lhs = _owned_dot(Pa, b)
    rhs = _owned_dot(a, PTb)
    scale = max(_owned_norm(Pa) * _owned_norm(b), _owned_norm(a) * _owned_norm(PTb), 1.0)
    eps = np.finfo(_real_dtype(dtype)).eps
    assert abs(lhs - rhs) < 500 * eps * scale

    # Ghost values are updated
    ghosts = PTb.x.array.copy()
    PTb.x.scatter_forward()
    np.testing.assert_array_equal(PTb.x.array, ghosts)
    return PTb


def _meshes(cell_type0, cell_type1, xtype, ghost_mode):
    comm = MPI.COMM_WORLD
    if mesh.cell_dim(cell_type0) == 2:
        msh0 = mesh.create_unit_square(
            comm, 5, 4, cell_type=cell_type0, dtype=xtype, ghost_mode=ghost_mode
        )
        msh1 = mesh.create_rectangle(
            comm,
            [np.array([0.1, 0.05]), np.array([0.9, 0.95])],
            [6, 7],
            cell_type=cell_type1,
            dtype=xtype,
            ghost_mode=ghost_mode,
        )
    else:
        msh0 = mesh.create_unit_cube(
            comm, 3, 2, 2, cell_type=cell_type0, dtype=xtype, ghost_mode=ghost_mode
        )
        msh1 = mesh.create_box(
            comm,
            [np.array([0.1, 0.05, 0.1]), np.array([0.9, 0.95, 0.85])],
            [3, 4, 3],
            cell_type=cell_type1,
            dtype=xtype,
            ghost_mode=ghost_mode,
        )
    return msh0, msh1


def _all_cells(msh: mesh.Mesh) -> np.ndarray:
    tdim = msh.topology.dim
    return np.arange(msh.topology.index_map(tdim).size_local, dtype=np.int32)


@pytest.mark.parametrize("ghost_mode", [mesh.GhostMode.none, mesh.GhostMode.shared_facet])
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize(
    "cell_types,element0,element1",
    [
        (
            (mesh.CellType.triangle, mesh.CellType.triangle),
            ("Lagrange", 1),
            ("Lagrange", 1),
        ),
        (
            (mesh.CellType.quadrilateral, mesh.CellType.triangle),
            ("Lagrange", 2),
            ("Lagrange", 3),
        ),
        (
            (mesh.CellType.triangle, mesh.CellType.quadrilateral),
            ("Lagrange", 2, (2,)),
            ("DG", 1, (2,)),
        ),
        (
            (mesh.CellType.triangle, mesh.CellType.triangle),
            ("Lagrange", 2, (2,)),
            ("RT", 1),
        ),
        (
            (mesh.CellType.triangle, mesh.CellType.triangle),
            ("Lagrange", 2),
            ("Legendre DG", 2),
        ),
        (
            (mesh.CellType.tetrahedron, mesh.CellType.tetrahedron),
            ("N1curl", 2),
            ("N1curl", 2),
        ),
        (
            (mesh.CellType.hexahedron, mesh.CellType.tetrahedron),
            ("Lagrange", 2),
            ("Lagrange", 2),
        ),
    ],
)
def test_adjoint(cell_types, element0, element1, dtype, ghost_mode):
    xtype = _real_dtype(dtype)
    msh0, msh1 = _meshes(*cell_types, xtype, ghost_mode)
    if element1[0] == "Legendre DG":
        element1 = basix.ufl.element(
            "DG",
            msh1.basix_cell(),
            element1[1],
            lagrange_variant=basix.LagrangeVariant.legendre,
            dtype=xtype,
        )
    V0 = fem.functionspace(msh0, element0)
    V1 = fem.functionspace(msh1, element1)
    cells1 = _all_cells(msh1)
    data = fem.create_interpolation_data(V1, V0, cells1)
    _check_adjoint(V0, V1, cells1, data, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adjoint_cell_subset(dtype):
    msh0, msh1 = _meshes(mesh.CellType.triangle, mesh.CellType.triangle, dtype, mesh.GhostMode.none)
    V0 = fem.functionspace(msh0, ("Lagrange", 2, (2,)))
    V1 = fem.functionspace(msh1, ("N1curl", 1))
    cells1 = _all_cells(msh1)[::2].copy()
    data = fem.create_interpolation_data(V1, V0, cells1)
    _check_adjoint(V0, V1, cells1, data, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unowned_points(dtype):
    """Points outside the mesh of V0 contribute nothing."""
    comm = MPI.COMM_WORLD
    msh0 = mesh.create_unit_square(comm, 4, 4, dtype=dtype)
    msh1 = mesh.create_rectangle(
        comm, [np.array([0.5, 0.5]), np.array([1.5, 1.5])], [5, 5], dtype=dtype
    )
    V0 = fem.functionspace(msh0, ("Lagrange", 1))
    V1 = fem.functionspace(msh1, ("Lagrange", 2))
    cells1 = _all_cells(msh1)
    data = fem.create_interpolation_data(V1, V0, cells1, allow_extrapolation=False)
    _check_adjoint(V0, V1, cells1, data, dtype)

    # A function supported away from the mesh of V0 maps to zero
    b = fem.Function(V1, dtype=dtype)
    b.interpolate(lambda x: np.where((x[0] > 1.2) & (x[1] > 1.2), 1.0, 0.0))
    c = fem.Function(V0, dtype=dtype)
    c.interpolate_nonmatching_transpose(b, cells1, data)
    assert np.all(c.x.array == 0)


def _refined(dtype, redistribute=False):
    msh0 = mesh.create_unit_square(MPI.COMM_WORLD, 4, 3, dtype=dtype)
    msh0.topology.create_entities(1)
    if redistribute:
        msh1, parent_cells, _ = mesh.refine(msh0, partitioner=partitioner())
    else:
        msh1, parent_cells, _ = mesh.refine(msh0)
    return msh0, msh1, parent_cells


@pytest.mark.parametrize("redistribute", [False, True])
@pytest.mark.parametrize("dtype", dtypes)
def test_hierarchy_interpolation_data(dtype, redistribute):
    xtype = _real_dtype(dtype)
    msh0, msh1, parent_cells = _refined(xtype, redistribute)
    V0 = fem.functionspace(msh0, ("Lagrange", 2))
    V1 = fem.functionspace(msh1, ("Lagrange", 2))
    cells1 = _all_cells(msh1)
    data = fem.create_hierarchy_interpolation_data(V1, V0, parent_cells)

    # Without redistribution, routing is local to each process
    comm = MPI.COMM_WORLD
    assert len(set(comm.allgather(data.process_local))) == 1
    if not redistribute:
        assert np.all(data.src_owner == comm.rank)
        assert np.all(data.dest_owner == comm.rank)
        assert data.process_local

    # Same forward and transpose operators as the search-based data. The
    # default padding misses points on cell boundaries in float32.
    eps = np.finfo(xtype).eps
    data_search = fem.create_interpolation_data(V1, V0, cells1, padding=100 * eps)
    assert len(set(comm.allgather(data_search.process_local))) == 1
    if comm.size == 1:
        assert data_search.process_local
    a = _random_function(V0, dtype, 4)
    b = _random_function(V1, dtype, 5)
    u, u_search = fem.Function(V1, dtype=dtype), fem.Function(V1, dtype=dtype)
    u.interpolate_nonmatching(a, cells1, data)
    u_search.interpolate_nonmatching(a, cells1, data_search)
    np.testing.assert_allclose(u.x.array, u_search.x.array, atol=100 * eps)
    v, v_search = fem.Function(V0, dtype=dtype), fem.Function(V0, dtype=dtype)
    v.interpolate_nonmatching_transpose(b, cells1, data)
    v_search.interpolate_nonmatching_transpose(b, cells1, data_search)
    np.testing.assert_allclose(v.x.array, v_search.x.array, atol=1000 * eps)

    _check_adjoint(V0, V1, cells1, data, dtype)


def test_hierarchy_interpolation_data_invalid_parents():
    msh0, msh1, parent_cells = _refined(np.float64)
    V0 = fem.functionspace(msh0, ("Lagrange", 1))
    V1 = fem.functionspace(msh1, ("Lagrange", 1))
    with pytest.raises(ValueError):
        fem.create_hierarchy_interpolation_data(V1, V0, parent_cells[:-1])
    if parent_cells.size > 0:
        invalid = parent_cells.copy()
        invalid[0] = -1
        with pytest.raises(ValueError):
            fem.create_hierarchy_interpolation_data(V1, V0, invalid)
