# Copyright (C) 2009-2019 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the fem interface"""

import sys

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type
from dolfinx.fem import functionspace
from dolfinx.mesh import (
    CellType,
    create_mesh,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
)

xfail = pytest.mark.xfail(strict=True)


@pytest.fixture
def mesh():
    return create_unit_square(MPI.COMM_WORLD, 4, 4)


@pytest.mark.skip
@pytest.mark.parametrize(
    "mesh_factory",
    [
        (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
        (create_unit_square, (MPI.COMM_WORLD, 4, 4, CellType.quadrilateral)),
    ],
)
def test_tabulate_dofs(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    W0 = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
    W1 = element(
        "Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,), dtype=default_real_type
    )
    W = functionspace(mesh, W0 * W1)

    L0 = W.sub(0)
    L1 = W.sub(1)
    L01 = L1.sub(0)
    L11 = L1.sub(1)

    map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map.size_local + map.num_ghosts
    for c in range(num_cells):
        dofs0 = L0.dofmap.cell_dofs(c)
        dofs1 = L01.dofmap.cell_dofs(c)
        dofs2 = L11.dofmap.cell_dofs(c)
        dofs3 = L1.dofmap.cell_dofs(c)
        assert len(np.intersect1d(dofs0, dofs1)) == 0
        assert len(np.intersect1d(dofs0, dofs2)) == 0
        assert len(np.intersect1d(dofs1, dofs2)) == 0
        assert np.array_equal(np.append(dofs1, dofs2), dofs3)


def test_entity_dofs(mesh):
    """Test that num entity dofs is correctly wrapped to dolfinx::DofMap"""
    gdim = mesh.geometry.dim

    V = functionspace(mesh, ("Lagrange", 1))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 0

    V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    bs = V.dofmap.dof_layout.block_size
    assert V.dofmap.dof_layout.num_entity_dofs(0) * bs == 2
    assert V.dofmap.dof_layout.num_entity_dofs(1) * bs == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) * bs == 0

    V = functionspace(mesh, ("Lagrange", 2))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 0

    V = functionspace(mesh, ("Lagrange", 3))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 2
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 1

    V = functionspace(mesh, ("DG", 0))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 1

    V = functionspace(mesh, ("DG", 1))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 3

    V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    bs = V.dofmap.dof_layout.block_size
    for i, cdofs in enumerate([[0, 1], [2, 3], [4, 5]]):
        dofs = [bs * d + b for d in V.dofmap.dof_layout.entity_dofs(0, i) for b in range(bs)]
        assert all(d == cd for d, cd in zip(dofs, cdofs))


@pytest.mark.skip
@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    "mesh_factory",
    [
        (create_unit_square, (MPI.COMM_WORLD, 2, 2)),
        (create_unit_square, (MPI.COMM_WORLD, 2, 2, CellType.quadrilateral)),
    ],
)
def test_entity_closure_dofs(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    tdim = mesh.topology.dim

    for degree in (1, 2, 3):
        V = functionspace(mesh, ("Lagrange", degree))
        for d in range(tdim + 1):
            map = mesh.topology.index_map(d)
            num_entities = map.size_local + map.num_ghosts
            covered = set()
            covered2 = set()
            all_entities = np.array([entity for entity in range(num_entities)], dtype=np.uintp)
            for entity in all_entities:
                entities = np.array([entity], dtype=np.uintp)
                dofs_on_this_entity = V.dofmap.entity_dofs(mesh, d, entities)
                closure_dofs = V.dofmap.entity_closure_dofs(mesh, d, entities)
                assert len(dofs_on_this_entity) == V.dofmap.dof_layout.num_entity_dofs(d)
                assert len(dofs_on_this_entity) <= len(closure_dofs)
                covered.update(dofs_on_this_entity)
                covered2.update(closure_dofs)
            dofs_on_all_entities = V.dofmap.entity_dofs(mesh, d, all_entities)
            closure_dofs_on_all_entities = V.dofmap.entity_closure_dofs(mesh, d, all_entities)
            assert (
                len(dofs_on_all_entities) == V.dofmap.dof_layout.num_entity_dofs(d) * num_entities
            )
            assert covered == set(dofs_on_all_entities)
            assert covered2 == set(closure_dofs_on_all_entities)

        d = tdim
        map = mesh.topology.index_map(d)
        num_entities = map.size_local + map.num_ghosts
        all_cells = np.array([entity for entity in range(num_entities)], dtype=np.uintp)
        assert set(V.dofmap.entity_closure_dofs(mesh, d, all_cells)) == set(range(V.dim))


def test_block_size():
    meshes = [
        create_unit_square(MPI.COMM_WORLD, 8, 8),
        create_unit_cube(MPI.COMM_WORLD, 4, 4, 4),
        create_unit_square(MPI.COMM_WORLD, 8, 8, CellType.quadrilateral),
        create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, CellType.hexahedron),
    ]
    for mesh in meshes:
        P2 = element("Lagrange", mesh.basix_cell(), 2, dtype=default_real_type)
        V = functionspace(mesh, P2)
        assert V.dofmap.bs == 1

        # Only BlockedElements have index_map_bs > 1
        V = functionspace(mesh, mixed_element([P2, P2]))
        assert V.dofmap.index_map_bs == 1

        for i in range(1, 6):
            W = functionspace(mesh, mixed_element(i * [P2]))
            assert W.dofmap.index_map_bs == 1

        gdim = mesh.geometry.dim
        V = functionspace(mesh, ("Lagrange", 2, (gdim,)))
        assert V.dofmap.index_map_bs == mesh.geometry.dim


@pytest.mark.skip
def test_block_size_real():
    mesh = create_unit_interval(MPI.COMM_WORLD, 12)
    V = element("DG", mesh.basix_cell(), 0, dtype=default_real_type)
    R = element("R", mesh.basix_cell(), 0, dtype=default_real_type)
    X = functionspace(mesh, V * R)
    assert X.dofmap.index_map_bs == 1


@pytest.mark.skip
@pytest.mark.parametrize(
    "mesh_factory",
    [
        (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
        (create_unit_square, (MPI.COMM_WORLD, 4, 4, CellType.quadrilateral)),
    ],
)
def test_local_dimension(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)

    v = element("Lagrange", mesh.basix_cell(), 1, dtype=default_real_type)
    q = element(
        "Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,), dtype=default_real_type
    )
    w = v * q

    V = functionspace(mesh, v)
    Q = functionspace(mesh, q)
    W = functionspace(mesh, w)
    for space in [V, Q, W]:
        dofmap = space.dofmap
        local_to_global_map = dofmap.tabulate_local_to_global_dofs()
        ownership_range = dofmap.index_set.size_local * dofmap.index_set.block_size
        dim1 = dofmap().index_map.size_local()
        dim2 = dofmap().index_map.num_ghosts()
        assert dim1 == ownership_range[1] - ownership_range[0]
        assert dim1 + dim2 == local_to_global_map.size


@pytest.mark.skip
def test_readonly_view_local_to_global_unwoned(mesh):
    """Test that local_to_global_unwoned() returns readonly
    view into the data; in particular test lifetime of data owner"""
    V = functionspace(mesh, "P", 1)
    dofmap = V.dofmap
    index_map = dofmap().index_map

    rc = sys.getrefcount(dofmap)
    l2gu = dofmap.local_to_global_unowned()
    assert sys.getrefcount(dofmap) == rc + 1 if l2gu.size else rc
    assert not l2gu.flags.writeable
    assert all(l2gu < V.dofmap.global_dimension())
    del l2gu
    assert sys.getrefcount(dofmap) == rc

    rc = sys.getrefcount(index_map)
    l2gu = index_map.local_to_global_unowned()
    assert sys.getrefcount(index_map) == rc + 1 if l2gu.size else rc
    assert not l2gu.flags.writeable
    assert all(l2gu < V.dofmap.global_dimension())
    del l2gu
    assert sys.getrefcount(index_map) == rc


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    "points, celltype, order",
    [
        (np.array([[0, 0], [1, 0], [0, 2], [1, 2]], dtype=np.float64), CellType.quadrilateral, 1),
        (
            np.array(
                [[0, 0], [1, 0], [0, 2], [1, 2], [0.5, 0], [0, 1], [1, 1], [0.5, 2], [0.5, 1]],
                dtype=np.float64,
            ),
            CellType.quadrilateral,
            2,
        ),
        (
            np.array([[0, 0], [1, 0], [0, 2], [0.5, 1], [0, 1], [0.5, 0]], dtype=np.float64),
            CellType.triangle,
            2,
        ),
        (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 2, 0],
                    [1, 2, 0],
                    [0, 0, 3],
                    [1, 0, 3],
                    [0, 2, 3],
                    [1, 2, 3],
                ],
                dtype=np.float64,
            ),
            CellType.hexahedron,
            1,
        ),
        (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 2, 0],
                    [1, 2, 0],
                    [0, 0, 3],
                    [1, 0, 3],
                    [0, 2, 3],
                    [1, 2, 3],
                    [0.5, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1.5],
                    [1, 1, 0],
                    [1, 0, 1.5],
                    [0.5, 2, 0],
                    [0, 2, 1.5],
                    [1, 2, 1.5],
                    [0.5, 0, 3],
                    [0, 1, 3],
                    [1, 1, 3],
                    [0.5, 2, 3],
                    [0.5, 1, 0],
                    [0.5, 0, 1.5],
                    [0, 1, 1.5],
                    [1, 1, 1.5],
                    [0.5, 2, 1.5],
                    [0.5, 1, 3],
                    [0.5, 1, 1.5],
                ],
                dtype=np.float64,
            ),
            CellType.hexahedron,
            2,
        ),
    ],
)
def test_higher_order_coordinate_map(points, celltype, order):
    """Computes physical coordinates of a cell, based on the coordinate map."""
    cells = np.array([range(len(points))])
    domain = ufl.Mesh(
        element("Lagrange", celltype.name, order, shape=(points.shape[1],), dtype=default_real_type)
    )
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

    V = functionspace(mesh, ("Lagrange", 2))
    X = V.element.interpolation_points()
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cmap = mesh.geometry.cmap

    x_coord_new = np.zeros([len(points), mesh.geometry.dim])

    i = 0
    for node in range(len(points)):
        x_coord_new[i] = x_g[coord_dofs[0, node], : mesh.geometry.dim]
        i += 1
    x = cmap.push_forward(X, x_coord_new)

    assert np.allclose(x[:, 0], X[:, 0], atol=100 * np.finfo(mesh.geometry.x.dtype).eps)
    assert np.allclose(x[:, 1], 2 * X[:, 1], atol=100 * np.finfo(mesh.geometry.x.dtype).eps)

    if mesh.geometry.dim == 3:
        assert np.allclose(x[:, 2], 3 * X[:, 2], atol=100 * np.finfo(mesh.geometry.x.dtype).eps)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("order", [1, 2])
def test_higher_order_tetra_coordinate_map(order):
    """Compute physical coordinates of a cell from the coordinate map."""
    celltype = CellType.tetrahedron
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
            [0, 4 / 3, 1],
            [0, 2 / 3, 2],
            [2 / 3, 0, 1],
            [1 / 3, 0, 2],
            [2 / 3, 2 / 3, 0],
            [1 / 3, 4 / 3, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 2 / 3, 0],
            [0, 4 / 3, 0],
            [1 / 3, 0, 0],
            [2 / 3, 0, 0],
            [1 / 3, 2 / 3, 1],
            [0, 2 / 3, 1],
            [1 / 3, 0, 1],
            [1 / 3, 2 / 3, 0],
        ],
        dtype=np.float64,
    )

    assert order <= 2
    if order == 1:
        points = np.array([points[0, :], points[1, :], points[2, :], points[3, :]])
    elif order == 2:
        points = np.array(
            [
                points[0, :],
                points[1, :],
                points[2, :],
                points[3, :],
                [0, 1, 3 / 2],
                [1 / 2, 0, 3 / 2],
                [1 / 2, 1, 0],
                [0, 0, 3 / 2],
                [0, 1, 0],
                [1 / 2, 0, 0],
            ]
        )
    cells = np.array([range(len(points))])
    domain = ufl.Mesh(
        element("Lagrange", celltype.name, order, shape=(3,), dtype=default_real_type)
    )
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    V = functionspace(mesh, ("Lagrange", order))
    X = V.element.interpolation_points()
    x_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x

    x_coord_new = np.zeros([len(points), mesh.geometry.dim])
    for node in range(points.shape[0]):
        x_coord_new[node] = x_g[x_dofs[0, node], : mesh.geometry.dim]

    x = mesh.geometry.cmap.push_forward(X, x_coord_new)
    assert np.allclose(x[:, 0], X[:, 0], atol=100 * np.finfo(mesh.geometry.x.dtype).eps)
    assert np.allclose(x[:, 1], 2 * X[:, 1], atol=100 * np.finfo(mesh.geometry.x.dtype).eps)
    assert np.allclose(x[:, 2], 3 * X[:, 2], atol=100 * np.finfo(mesh.geometry.x.dtype).eps)


@pytest.mark.skip_in_parallel
def test_transpose_dofmap():
    dofmap = np.array([[0, 2, 1], [3, 2, 1], [4, 3, 1]], dtype=np.int32)
    transpose = dolfinx.fem.transpose_dofmap(dofmap, 3)
    assert np.array_equal(transpose.array, [0, 2, 5, 8, 1, 4, 3, 7, 6])


def test_empty_rank_collapse():
    """Test that dofmap with no dofs on a rank can be collapsed"""
    if MPI.COMM_WORLD.rank == 0:
        nodes = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
        cells = np.array([[0, 1], [1, 2]], dtype=np.int64)
    else:
        nodes = np.empty((0, 1), dtype=np.float64)
        cells = np.empty((0, 2), dtype=np.int64)
    c_el = element("Lagrange", "interval", 1, shape=(1,))

    def self_partitioner(comm: MPI.Intracomm, n, m, topo):
        dests = np.full(len(topo[0]) // 2, comm.rank, dtype=np.int32)
        offsets = np.arange(len(topo[0]) // 2 + 1, dtype=np.int32)
        return dolfinx.graph.adjacencylist(dests, offsets)

    mesh = create_mesh(MPI.COMM_WORLD, cells, nodes, c_el, partitioner=self_partitioner)

    el = element("Lagrange", "interval", 1, shape=(2,))
    V = functionspace(mesh, el)
    V_0, _ = V.sub(0).collapse()
    assert V.dofmap.index_map.size_local == V_0.dofmap.index_map.size_local
