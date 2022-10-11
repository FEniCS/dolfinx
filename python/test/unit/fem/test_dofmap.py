# Copyright (C) 2009-2019 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the fem interface"""

import sys

import numpy as np
import pytest

import dolfinx
import ufl
from dolfinx.fem import FunctionSpace, VectorFunctionSpace
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import (CellType, create_mesh, create_unit_cube,
                          create_unit_interval, create_unit_square)
from ufl import FiniteElement, MixedElement, VectorElement

from mpi4py import MPI

xfail = pytest.mark.xfail(strict=True)


@pytest.fixture
def mesh():
    return create_unit_square(MPI.COMM_WORLD, 4, 4)


@pytest.mark.skip
@pytest.mark.parametrize(
    'mesh_factory', [(create_unit_square, (MPI.COMM_WORLD, 4, 4)),
                     (create_unit_square,
                      (MPI.COMM_WORLD, 4, 4, CellType.quadrilateral))])
def test_tabulate_dofs(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    W0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, W0 * W1)

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
    V = FunctionSpace(mesh, ("Lagrange", 1))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 0

    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    bs = V.dofmap.dof_layout.block_size
    assert V.dofmap.dof_layout.num_entity_dofs(0) * bs == 2
    assert V.dofmap.dof_layout.num_entity_dofs(1) * bs == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) * bs == 0

    V = FunctionSpace(mesh, ("Lagrange", 2))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 0

    V = FunctionSpace(mesh, ("Lagrange", 3))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 2
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 1

    V = FunctionSpace(mesh, ("DG", 0))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 1

    V = FunctionSpace(mesh, ("DG", 1))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 3

    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    bs = V.dofmap.dof_layout.block_size

    for i, cdofs in enumerate([[0, 1], [2, 3], [4, 5]]):
        dofs = [bs * d + b for d in V.dofmap.dof_layout.entity_dofs(0, i)
                for b in range(bs)]
        assert all(d == cd for d, cd in zip(dofs, cdofs))


@pytest.mark.skip
@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    'mesh_factory', [(create_unit_square, (MPI.COMM_WORLD, 2, 2)),
                     (create_unit_square,
                      (MPI.COMM_WORLD, 2, 2, CellType.quadrilateral))])
def test_entity_closure_dofs(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    tdim = mesh.topology.dim

    for degree in (1, 2, 3):
        V = FunctionSpace(mesh, ("Lagrange", degree))
        for d in range(tdim + 1):
            map = mesh.topology.index_map(d)
            num_entities = map.size_local + map.num_ghosts
            covered = set()
            covered2 = set()
            all_entities = np.array([entity for entity in range(num_entities)], dtype=np.uintp)
            for entity in all_entities:
                entities = np.array([entity], dtype=np.uintp)
                dofs_on_this_entity = V.dofmap.entity_dofs(mesh, d, entities)
                closure_dofs = V.dofmap.entity_closure_dofs(
                    mesh, d, entities)
                assert len(dofs_on_this_entity) == V.dofmap.dof_layout.num_entity_dofs(d)
                assert len(dofs_on_this_entity) <= len(closure_dofs)
                covered.update(dofs_on_this_entity)
                covered2.update(closure_dofs)
            dofs_on_all_entities = V.dofmap.entity_dofs(
                mesh, d, all_entities)
            closure_dofs_on_all_entities = V.dofmap.entity_closure_dofs(
                mesh, d, all_entities)
            assert len(dofs_on_all_entities) == V.dofmap.dof_layout.num_entity_dofs(d) * num_entities
            assert covered == set(dofs_on_all_entities)
            assert covered2 == set(closure_dofs_on_all_entities)

        d = tdim
        map = mesh.topology.index_map(d)
        num_entities = map.size_local + map.num_ghosts
        all_cells = np.array([entity for entity in range(num_entities)], dtype=np.uintp)
        assert set(V.dofmap.entity_closure_dofs(mesh, d, all_cells)) == set(range(V.dim))


def test_block_size(mesh):
    meshes = [
        create_unit_square(MPI.COMM_WORLD, 8, 8),
        create_unit_cube(MPI.COMM_WORLD, 4, 4, 4),
        create_unit_square(MPI.COMM_WORLD, 8, 8, CellType.quadrilateral),
        create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, CellType.hexahedron)
    ]
    for mesh in meshes:
        P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)

        V = FunctionSpace(mesh, P2)
        assert V.dofmap.bs == 1

        # Only VectorElements have index_map_bs > 1
        V = FunctionSpace(mesh, MixedElement([P2, P2]))
        assert V.dofmap.index_map_bs == 1

        for i in range(1, 6):
            W = FunctionSpace(mesh, MixedElement(i * [P2]))
            assert W.dofmap.index_map_bs == 1

        V = VectorFunctionSpace(mesh, ("Lagrange", 2))
        assert V.dofmap.index_map_bs == mesh.geometry.dim


@pytest.mark.skip
def test_block_size_real():
    mesh = create_unit_interval(MPI.COMM_WORLD, 12)
    V = FiniteElement('DG', mesh.ufl_cell(), 0)
    R = FiniteElement('R', mesh.ufl_cell(), 0)
    X = FunctionSpace(mesh, V * R)
    assert X.dofmap.index_map_bs == 1


@pytest.mark.skip
@pytest.mark.parametrize(
    'mesh_factory', [(create_unit_square, (MPI.COMM_WORLD, 4, 4)),
                     (create_unit_square,
                      (MPI.COMM_WORLD, 4, 4, CellType.quadrilateral))])
def test_local_dimension(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)

    v = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    q = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    w = v * q

    V = FunctionSpace(mesh, v)
    Q = FunctionSpace(mesh, q)
    W = FunctionSpace(mesh, w)

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
    view into the data; in particular test lifetime of data
    owner"""
    V = FunctionSpace(mesh, "P", 1)
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
@pytest.mark.parametrize("points, celltype, order", [
    (np.array([[0, 0], [1, 0], [0, 2], [1, 2]]),
     CellType.quadrilateral, 1),
    (np.array([[0, 0], [1, 0], [0, 2], [1, 2],
               [0.5, 0], [0, 1], [1, 1], [0.5, 2], [0.5, 1]]),
     CellType.quadrilateral, 2),
    (np.array([[0, 0], [1, 0], [0, 2], [0.5, 1], [0, 1], [0.5, 0]]),
     CellType.triangle, 2),
    (np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0],
               [0, 0, 3], [1, 0, 3], [0, 2, 3], [1, 2, 3]]),
     CellType.hexahedron, 1),
    (np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 0],
               [0, 0, 3], [1, 0, 3], [0, 2, 3], [1, 2, 3],
               [0.5, 0, 0], [0, 1, 0], [0, 0, 1.5], [1, 1, 0],
               [1, 0, 1.5], [0.5, 2, 0], [0, 2, 1.5], [1, 2, 1.5],
               [0.5, 0, 3], [0, 1, 3], [1, 1, 3], [0.5, 2, 3],
               [0.5, 1, 0], [0.5, 0, 1.5], [0, 1, 1.5], [1, 1, 1.5],
               [0.5, 2, 1.5], [0.5, 1, 3], [0.5, 1, 1.5]]),
     CellType.hexahedron, 2)
])
def test_higher_order_coordinate_map(points, celltype, order):
    """Computes physical coordinates of a cell, based on the coordinate map."""
    cells = np.array([range(len(points))])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", celltype.name, order))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

    V = FunctionSpace(mesh, ("Lagrange", 2))
    X = V.element.interpolation_points()
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cmap = mesh.geometry.cmap

    x_coord_new = np.zeros([len(points), mesh.geometry.dim])

    i = 0
    for node in range(len(points)):
        x_coord_new[i] = x_g[coord_dofs.links(0)[node], :mesh.geometry.dim]
        i += 1
    x = cmap.push_forward(X, x_coord_new)

    assert np.allclose(x[:, 0], X[:, 0])
    assert np.allclose(x[:, 1], 2 * X[:, 1])

    if mesh.geometry.dim == 3:
        assert np.allclose(x[:, 2], 3 * X[:, 2])


@pytest.mark.skip_in_parallel
# @pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("order", [1, 2])
def test_higher_order_tetra_coordinate_map(order):
    """
    Computes physical coordinates of a cell, based on the coordinate map.
    """
    celltype = CellType.tetrahedron
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3],
                       [0, 4 / 3, 1], [0, 2 / 3, 2],
                       [2 / 3, 0, 1], [1 / 3, 0, 2],
                       [2 / 3, 2 / 3, 0], [1 / 3, 4 / 3, 0],
                       [0, 0, 1], [0, 0, 2],
                       [0, 2 / 3, 0], [0, 4 / 3, 0],
                       [1 / 3, 0, 0], [2 / 3, 0, 0],
                       [1 / 3, 2 / 3, 1], [0, 2 / 3, 1],
                       [1 / 3, 0, 1], [1 / 3, 2 / 3, 0]])

    if order == 1:
        points = np.array([points[0, :], points[1, :], points[2, :], points[3, :]])
    elif order == 2:
        points = np.array([points[0, :], points[1, :], points[2, :], points[3, :],
                           [0, 1, 3 / 2], [1 / 2, 0, 3 / 2], [1 / 2, 1, 0], [0, 0, 3 / 2],
                           [0, 1, 0], [1 / 2, 0, 0]])
    cells = np.array([range(len(points))])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", celltype.name, order))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    V = FunctionSpace(mesh, ("Lagrange", order))
    X = V.element.interpolation_points()
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x

    cmap = mesh.geometry.cmap
    x_coord_new = np.zeros([len(points), mesh.geometry.dim])

    i = 0
    for node in range(len(points)):
        x_coord_new[i] = x_g[coord_dofs.links(0)[node], :mesh.geometry.dim]
        i += 1

    x = cmap.push_forward(X, x_coord_new)
    assert np.allclose(x[:, 0], X[:, 0])
    assert np.allclose(x[:, 1], 2 * X[:, 1])
    assert np.allclose(x[:, 2], 3 * X[:, 2])


@pytest.mark.skip_in_parallel
def test_transpose_dofmap():
    dofmap = create_adjacencylist(np.array([[0, 2, 1], [3, 2, 1], [4, 3, 1]], dtype=np.int32))
    transpose = dolfinx.fem.transpose_dofmap(dofmap, 3)
    assert np.array_equal(transpose.array, [0, 2, 5, 8, 1, 4, 3, 7, 6])
