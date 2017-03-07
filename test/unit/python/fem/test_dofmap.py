#!/usr/bin/env py.test

"""Unit tests for the fem interface"""

# Copyright (C) 2009-2014 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
import pytest
import numpy as np
from dolfin import *

from dolfin_utils.test import *


@fixture
def mesh():
    return UnitSquareMesh(4, 4)


@fixture
def V(mesh):
    return FunctionSpace(mesh, "Lagrange", 1)


@fixture
def Q(mesh):
    return VectorFunctionSpace(mesh, "Lagrange", 1)


@fixture
def W(mesh):
    V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    Q = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    return FunctionSpace(mesh, V*Q)


reorder_dofs = set_parameters_fixture("reorder_dofs_serial", [True, False])


def test_tabulate_all_coordinates(mesh, V, W):
    D = mesh.geometry().dim()
    V_dofmap = V.dofmap()
    W_dofmap = W.dofmap()

    all_coords_V = V.tabulate_dof_coordinates()
    all_coords_W = W.tabulate_dof_coordinates()
    local_size_V = V_dofmap.ownership_range()[1]-V_dofmap.ownership_range()[0]
    local_size_W = W_dofmap.ownership_range()[1]-W_dofmap.ownership_range()[0]

    assert all_coords_V.shape == (D*local_size_V,)
    assert all_coords_W.shape == (D*local_size_W,)

    all_coords_V = all_coords_V.reshape(local_size_V, D)
    all_coords_W = all_coords_W.reshape(local_size_W, D)

    checked_V = [False]*local_size_V
    checked_W = [False]*local_size_W

    # Check that all coordinates are within the cell it should be
    for cell in cells(mesh):
        dofs_V = V_dofmap.cell_dofs(cell.index())
        for di in dofs_V:
            if di >= local_size_V:
                continue
            assert cell.contains(Point(all_coords_V[di]))
            checked_V[di] = True

        dofs_W = W_dofmap.cell_dofs(cell.index())
        for di in dofs_W:
            if di >= local_size_W:
                continue
            assert cell.contains(Point(all_coords_W[di]))
            checked_W[di] = True

    # Assert that all dofs have been checked by the above
    assert all(checked_V)
    assert all(checked_W)


def test_tabulate_dofs(mesh, W):

    L0 = W.sub(0)
    L1 = W.sub(1)
    L01 = L1.sub(0)
    L11 = L1.sub(1)

    for i, cell in enumerate(cells(mesh)):
        dofs0 = L0.dofmap().cell_dofs(cell.index())
        dofs1 = L01.dofmap().cell_dofs(cell.index())
        dofs2 = L11.dofmap().cell_dofs(cell.index())
        dofs3 = L1.dofmap().cell_dofs(cell.index())

        assert np.array_equal(dofs0, L0.dofmap().cell_dofs(i))
        assert np.array_equal(dofs1, L01.dofmap().cell_dofs(i))
        assert np.array_equal(dofs2, L11.dofmap().cell_dofs(i))
        assert np.array_equal(dofs3, L1.dofmap().cell_dofs(i))

        assert len(np.intersect1d(dofs0, dofs1)) == 0
        assert len(np.intersect1d(dofs0, dofs2)) == 0
        assert len(np.intersect1d(dofs1, dofs2)) == 0
        assert np.array_equal(np.append(dofs1, dofs2), dofs3)


def test_tabulate_coord_periodic():

    class PeriodicBoundary2(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < DOLFIN_EPS

        def map(self, x, y):
            y[0] = x[0] - 1.0
            y[1] = x[1]

    # Create periodic boundary condition
    periodic_boundary = PeriodicBoundary2()

    mesh = UnitSquareMesh(4, 4)

    V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    Q = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    W = V*Q

    V = FunctionSpace(mesh, V, constrained_domain=periodic_boundary)
    W = FunctionSpace(mesh, W, constrained_domain=periodic_boundary)

    L0 = W.sub(0)
    L1 = W.sub(1)
    L01 = L1.sub(0)
    L11 = L1.sub(1)

    coord0 = np.zeros((3, 2), dtype="d")
    coord1 = np.zeros((3, 2), dtype="d")
    coord2 = np.zeros((3, 2), dtype="d")
    coord3 = np.zeros((3, 2), dtype="d")

    for cell in cells(mesh):
        V.element().tabulate_dof_coordinates(cell, coord0)
        L0.element().tabulate_dof_coordinates(cell, coord1)
        L01.element().tabulate_dof_coordinates(cell, coord2)
        L11.element().tabulate_dof_coordinates(cell, coord3)
        coord4 = L1.element().tabulate_dof_coordinates(cell)

        assert (coord0 == coord1).all()
        assert (coord0 == coord2).all()
        assert (coord0 == coord3).all()
        assert (coord4[:3] == coord0).all()
        assert (coord4[3:] == coord0).all()


def test_tabulate_dofs_periodic():

    class PeriodicBoundary2(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] < DOLFIN_EPS

        def map(self, x, y):
            y[0] = x[0] - 1.0
            y[1] = x[1]

    mesh = UnitSquareMesh(5, 5)

    # Create periodic boundary
    periodic_boundary = PeriodicBoundary2()

    V = FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    Q = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    W = V*Q

    V = FunctionSpace(mesh, V, constrained_domain=periodic_boundary)
    Q = FunctionSpace(mesh, Q, constrained_domain=periodic_boundary)
    W = FunctionSpace(mesh, W, constrained_domain=periodic_boundary)

    L0 = W.sub(0)
    L1 = W.sub(1)
    L01 = L1.sub(0)
    L11 = L1.sub(1)

    # Check dimensions
    assert V.dim() == 110
    assert Q.dim() == 220
    assert L0.dim() == V.dim()
    assert L1.dim() == Q.dim()
    assert L01.dim() == V.dim()
    assert L11.dim() == V.dim()

    for i, cell in enumerate(cells(mesh)):
        dofs0 = L0.dofmap().cell_dofs(cell.index())
        dofs1 = L01.dofmap().cell_dofs(cell.index())
        dofs2 = L11.dofmap().cell_dofs(cell.index())
        dofs3 = L1.dofmap().cell_dofs(cell.index())

        assert np.array_equal(dofs0, L0.dofmap().cell_dofs(i))
        assert np.array_equal(dofs1, L01.dofmap().cell_dofs(i))
        assert np.array_equal(dofs2, L11.dofmap().cell_dofs(i))
        assert np.array_equal(dofs3, L1.dofmap().cell_dofs(i))

        assert len(np.intersect1d(dofs0, dofs1)) == 0
        assert len(np.intersect1d(dofs0, dofs2)) == 0
        assert len(np.intersect1d(dofs1, dofs2)) == 0
        assert np.array_equal(np.append(dofs1, dofs2), dofs3)


def test_global_dof_builder():
    mesh = UnitSquareMesh(3, 3)

    V = VectorElement("CG", mesh.ufl_cell(), 1)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    R = FiniteElement("R",  mesh.ufl_cell(), 0)

    W = FunctionSpace(mesh, MixedElement([Q, Q, Q, R]))
    W = FunctionSpace(mesh, MixedElement([Q, Q, R, Q]))
    W = FunctionSpace(mesh, V*R)
    W = FunctionSpace(mesh, R*V)


def test_dof_to_vertex_map(mesh, reorder_dofs):

    def _test_maps_consistency(space):
        v2d = vertex_to_dof_map(space)
        d2v = dof_to_vertex_map(space)
        assert len(v2d) == len(d2v)
        assert np.all(v2d[d2v] == np.arange(len(v2d)))
        assert np.all(d2v[v2d] == np.arange(len(d2v)))

    # Check for both reordered and UFC ordered dofs
    v = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    q = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    w = v*q

    V = FunctionSpace(mesh, v)
    Q = FunctionSpace(mesh, q)
    W = FunctionSpace(mesh, w)

    _test_maps_consistency(V)
    _test_maps_consistency(Q)
    _test_maps_consistency(W)

    u = Function(V)
    e = Expression("x[0] + x[1]", degree=1)
    u.interpolate(e)

    vert_values = mesh.coordinates().sum(1)
    func_values = np.empty(len(vert_values))
    u.vector().get_local(func_values, vertex_to_dof_map(V))
    assert round(max(abs(func_values - vert_values)), 7) == 0

    c0 = Constant((1, 2))
    u0 = Function(Q)
    u0.interpolate(c0)

    vert_values = np.zeros(mesh.num_vertices()*2)
    u1 = Function(Q)
    vert_values[::2] = 1
    vert_values[1::2] = 2

    dim = Q.dofmap().index_map().size(IndexMap.MapSize_OWNED)
    u1.vector().set_local(vert_values[dof_to_vertex_map(Q)[:dim]].copy())
    assert round((u0.vector()-u1.vector()).sum() - 0.0, 7) == 0

    W = FunctionSpace(mesh, "DG", 0)
    with pytest.raises(RuntimeError):
        dof_to_vertex_map(W)

    W = FunctionSpace(mesh, q*FiniteElement("R", mesh.ufl_cell(), 0))
    with pytest.raises(RuntimeError):
        dof_to_vertex_map(W)

    W = FunctionSpace(mesh, "CG", 2)
    with pytest.raises(RuntimeError):
        dof_to_vertex_map(W)

    W = VectorFunctionSpace(mesh, "CG", 1)
    with pytest.raises(RuntimeError):
        dof_to_vertex_map(W.sub(0))


def test_entity_dofs(mesh):

    # Test that num entity dofs is correctly wrapped to
    # dolfin::DofMap
    V = FunctionSpace(mesh, "CG", 1)
    assert V.dofmap().num_entity_dofs(0) == 1
    assert V.dofmap().num_entity_dofs(1) == 0
    assert V.dofmap().num_entity_dofs(2) == 0

    V = VectorFunctionSpace(mesh, "CG", 1)
    assert V.dofmap().num_entity_dofs(0) == 2
    assert V.dofmap().num_entity_dofs(1) == 0
    assert V.dofmap().num_entity_dofs(2) == 0

    V = FunctionSpace(mesh, "CG", 2)
    assert V.dofmap().num_entity_dofs(0) == 1
    assert V.dofmap().num_entity_dofs(1) == 1
    assert V.dofmap().num_entity_dofs(2) == 0

    V = FunctionSpace(mesh, "CG", 3)
    assert V.dofmap().num_entity_dofs(0) == 1
    assert V.dofmap().num_entity_dofs(1) == 2
    assert V.dofmap().num_entity_dofs(2) == 1

    V = FunctionSpace(mesh, "DG", 0)
    assert V.dofmap().num_entity_dofs(0) == 0
    assert V.dofmap().num_entity_dofs(1) == 0
    assert V.dofmap().num_entity_dofs(2) == 1

    V = FunctionSpace(mesh, "DG", 1)
    assert V.dofmap().num_entity_dofs(0) == 0
    assert V.dofmap().num_entity_dofs(1) == 0
    assert V.dofmap().num_entity_dofs(2) == 3

    V = VectorFunctionSpace(mesh, "CG", 1)

    # Note this numbering is dependent on FFC and can change This test
    # is here just to check that we get correct numbers mapped from
    # ufc generated code to dolfin
    for i, cdofs in enumerate([[0, 3], [1, 4], [2, 5]]):
        dofs = V.dofmap().tabulate_entity_dofs(0, i)
        assert all(d == cd for d, cd in zip(dofs, cdofs))


@skip_in_parallel
def test_entity_closure_dofs():
    mesh = UnitSquareMesh(1, 1)
    tdim = mesh.topology().dim()

    for degree in (1, 2, 3):
        V = FunctionSpace(mesh, "CG", degree)
        for d in range(tdim + 1):
            covered = set()
            covered2 = set()
            all_entities = np.array([entity for entity in range(mesh.num_entities(d))], dtype=np.uintp)
            for entity in all_entities:
                entities = np.array([entity], dtype=np.uintp)
                dofs_on_this_entity = V.dofmap().entity_dofs(mesh, d, entities)
                closure_dofs = V.dofmap().entity_closure_dofs(mesh, d, entities)
                assert len(dofs_on_this_entity) == V.dofmap().num_entity_dofs(d)
                assert len(dofs_on_this_entity) <= len(closure_dofs)
                covered.update(dofs_on_this_entity)
                covered2.update(closure_dofs)
            dofs_on_all_entities = V.dofmap().entity_dofs(mesh, d, all_entities)
            closure_dofs_on_all_entities = V.dofmap().entity_closure_dofs(mesh, d, all_entities)
            assert len(dofs_on_all_entities) == V.dofmap().num_entity_dofs(d) * mesh.num_entities(d)
            assert covered == set(dofs_on_all_entities)
            assert covered2 == set(closure_dofs_on_all_entities)
        d = tdim
        all_cells = np.array([entity for entity in range(mesh.num_entities(d))], dtype=np.uintp)
        assert set(V.dofmap().entity_closure_dofs(mesh, d, all_cells)) == set(range(V.dim()))


def test_clear_sub_map_data_scalar(mesh):
    V = FunctionSpace(mesh, "CG", 2)
    with pytest.raises(ValueError):
        V.sub(1)

    V = VectorFunctionSpace(mesh, "CG", 2)
    V1 = V.sub(1)

    # Clean sub-map data
    V.dofmap().clear_sub_map_data()

    # Can still get previously computed map
    V1 = V.sub(1)

    # New sub-map should throw an error
    with pytest.raises(RuntimeError):
        V.sub(0)


def test_clear_sub_map_data_vector(mesh):
    mesh = UnitSquareMesh(8, 8)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, P1*P1)

    # Check block size
    assert W.dofmap().block_size() == 2

    W.dofmap().clear_sub_map_data()
    with pytest.raises(RuntimeError):
        W0 = W.sub(0)
    with pytest.raises(RuntimeError):
        W1 = W.sub(1)


def test_block_size(mesh):
    meshes = [UnitSquareMesh(8, 8), UnitCubeMesh(4, 4, 4)]
    for mesh in meshes:
        P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)

        V = FunctionSpace(mesh, P2)
        assert V.dofmap().block_size() == 1

        V = FunctionSpace(mesh, P2*P2)
        assert V.dofmap().block_size() == 2

        for i in range(1, 6):
            W = FunctionSpace(mesh, MixedElement(i*[P2]))
            assert W.dofmap().block_size() == i

        V = VectorFunctionSpace(mesh, "Lagrange", 2)
        assert V.dofmap().block_size() == mesh.geometry().dim()


def test_block_size_real(mesh):
    mesh = UnitIntervalMesh(12)
    V = FiniteElement('DG', mesh.ufl_cell(), 0)
    R = FiniteElement('R',  mesh.ufl_cell(), 0)
    X = FunctionSpace(mesh, V*R)
    assert X.dofmap().block_size() == 1


@skip_in_serial
def test_mpi_dofmap_stats(mesh):

    V = FunctionSpace(mesh, "CG", 1)
    assert len(V.dofmap().shared_nodes()) > 0
    neighbours = V.dofmap().neighbours()
    for processes in V.dofmap().shared_nodes().values():
        for process in processes:
            assert process in neighbours

    for owner in V.dofmap().off_process_owner():
        assert owner in neighbours


def test_local_dimension(V, Q, W):
    for space in [V, Q, W]:
        dofmap = space.dofmap()
        local_to_global_map = dofmap.tabulate_local_to_global_dofs()
        ownership_range = dofmap.ownership_range()
        dim1 = dofmap.index_map().size(IndexMap.MapSize_OWNED)
        dim2 = dofmap.index_map().size(IndexMap.MapSize_UNOWNED)
        dim3 = dofmap.index_map().size(IndexMap.MapSize_ALL)
        assert dim1 == ownership_range[1] - ownership_range[0]
        assert dim3 == local_to_global_map.size
        assert dim1 + dim2 == dim3
        # with pytest.raises(RuntimeError):
        #    dofmap.index_map().size('foo')


@skip_in_parallel
def test_dofs_dim(mesh, V, Q, W):
    """Test function GenericDofMap::dofs(mesh, dim)"""
    meshes = [UnitIntervalMesh(10),
              UnitSquareMesh(6, 6),
              UnitCubeMesh(2, 2, 2)]

    for mesh in meshes:
        tdim = mesh.topology().dim()
        spaces = [FunctionSpace(mesh, "Discontinuous Lagrange", 1),
                  FunctionSpace(mesh, "Discontinuous Lagrange", 2),
                  FunctionSpace(mesh, "Lagrange", 1),
                  FunctionSpace(mesh, "Lagrange", 2),
                  FunctionSpace(mesh, "Lagrange", 3)]

        if tdim > 1:
            N1 = "Nedelec 1st kind H(curl)"
            vspaces = [VectorFunctionSpace(mesh, N1, 1),
                       VectorFunctionSpace(mesh, N1, 2),
                       VectorFunctionSpace(mesh, "RT", 1)]
            spaces = spaces + vspaces

        for V in spaces:
            dofmap = V.dofmap()
            for dim in range(0, mesh.topology().dim()):
                edofs = dofmap.dofs(mesh, dim)
                num_mesh_entities = mesh.num_entities(dim)
                dofs_per_entity = dofmap.num_entity_dofs(dim)
                assert len(edofs) == dofs_per_entity*num_mesh_entities
