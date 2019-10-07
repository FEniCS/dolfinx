"""Unit tests for the fem interface"""

# Copyright (C) 2009-2018 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import sys

import numpy as np
import pytest

from dolfin import (MPI, FunctionSpace, MeshEntity, UnitCubeMesh,
                    UnitIntervalMesh, UnitSquareMesh, VectorFunctionSpace, cpp,
                    fem)
from dolfin.cpp.mesh import CellType
from dolfin_utils.test.skips import skip_in_parallel
from ufl import FiniteElement, MixedElement, VectorElement

xfail = pytest.mark.xfail(strict=True)


@pytest.fixture
def mesh():
    return UnitSquareMesh(MPI.comm_world, 4, 4)


@pytest.mark.skip
@pytest.mark.parametrize(
    'mesh_factory',
    [
        (UnitIntervalMesh, (
            MPI.comm_world,
            8,
        )),
        (UnitSquareMesh, (MPI.comm_world, 4, 4)),
        (UnitCubeMesh, (MPI.comm_world, 2, 2, 2)),
        # cell.contains(Point) does not work correctly
        # for quad/hex cells once it is fixed, this test will pass
        pytest.param((UnitSquareMesh,
                      (MPI.comm_world, 4, 4, CellType.quadrilateral)),
                     marks=pytest.mark.xfail),
        pytest.param((UnitCubeMesh,
                      (MPI.comm_world, 2, 2, 2, CellType.hexahedron)),
                     marks=pytest.mark.xfail)
    ])
def test_tabulate_all_coordinates(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    W0 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, W0 * W1)

    D = mesh.geometry.dim
    V_dofmap = V.dofmap
    W_dofmap = W.dofmap

    all_coords_V = V.tabulate_dof_coordinates()
    all_coords_W = W.tabulate_dof_coordinates()
    local_size_V = V_dofmap().index_map.size_local * V_dofmap().index_map.block_size
    local_size_W = W_dofmap().index_map.size_local * W_dofmap().index_map.block_size

    all_coords_V = all_coords_V.reshape(local_size_V, D)
    all_coords_W = all_coords_W.reshape(local_size_W, D)

    checked_V = [False] * local_size_V
    checked_W = [False] * local_size_W

    # Check that all coordinates are within the cell it should be
    for i in range(mesh.num_cells()):
        cell = MeshEntity(mesh, mesh.topology.dim, i)
        dofs_V = V_dofmap.cell_dofs(i)
        for di in dofs_V:
            if di >= local_size_V:
                continue
            assert cell.contains(all_coords_V[di])
            checked_V[di] = True

        dofs_W = W_dofmap.cell_dofs(cell.index())
        for di in dofs_W:
            if di >= local_size_W:
                continue
            assert cell.contains(all_coords_W[di])
            checked_W[di] = True

    # Assert that all dofs have been checked by the above
    assert all(checked_V)
    assert all(checked_W)


@pytest.mark.skip
@pytest.mark.parametrize(
    'mesh_factory', [(UnitSquareMesh, (MPI.comm_world, 4, 4)),
                     (UnitSquareMesh,
                      (MPI.comm_world, 4, 4, CellType.quadrilateral))])
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

    for i in range(mesh.num_cells()):
        dofs0 = L0.dofmap.cell_dofs(i)
        dofs1 = L01.dofmap.cell_dofs(i)
        dofs2 = L11.dofmap.cell_dofs(i)
        dofs3 = L1.dofmap.cell_dofs(i)
        assert len(np.intersect1d(dofs0, dofs1)) == 0
        assert len(np.intersect1d(dofs0, dofs2)) == 0
        assert len(np.intersect1d(dofs1, dofs2)) == 0
        assert np.array_equal(np.append(dofs1, dofs2), dofs3)


@pytest.mark.skip
@pytest.mark.parametrize(
    'mesh_factory', [(UnitSquareMesh, (MPI.comm_world, 4, 4)),
                     (UnitSquareMesh,
                      (MPI.comm_world, 4, 4, CellType.quadrilateral))])
def test_tabulate_coord_periodic(mesh_factory):
    def periodic_boundary(x):
        return x[0] < np.finfo(float).eps

    func, args = mesh_factory
    mesh = func(*args)

    V = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    Q = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    W = V * Q

    V = FunctionSpace(mesh, V, constrained_domain=periodic_boundary)
    W = FunctionSpace(mesh, W, constrained_domain=periodic_boundary)

    L0 = W.sub(0)
    L1 = W.sub(1)
    L01 = L1.sub(0)
    L11 = L1.sub(1)

    sdim = V.element.space_dimension()
    coord0 = np.zeros((sdim, 2), dtype="d")
    coord1 = np.zeros((sdim, 2), dtype="d")
    coord2 = np.zeros((sdim, 2), dtype="d")
    coord3 = np.zeros((sdim, 2), dtype="d")

    for i in range(mesh.num_cells()):
        cell = MeshEntity(mesh, mesh.topology.dim, i)
        coord0 = V.element.tabulate_dof_coordinates(cell)
        coord1 = L0.element.tabulate_dof_coordinates(cell)
        coord2 = L01.element.tabulate_dof_coordinates(cell)
        coord3 = L11.element.tabulate_dof_coordinates(cell)
        coord4 = L1.element.tabulate_dof_coordinates(cell)

        assert (coord0 == coord1).all()
        assert (coord0 == coord2).all()
        assert (coord0 == coord3).all()
        assert (coord4[:sdim] == coord0).all()
        assert (coord4[sdim:] == coord0).all()


@pytest.mark.skip
@pytest.mark.parametrize(
    'mesh_factory', [(UnitSquareMesh, (MPI.comm_world, 3, 3)),
                     (UnitSquareMesh,
                      (MPI.comm_world, 3, 3, CellType.quadrilateral))])
def test_global_dof_builder(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)

    V = VectorElement("CG", mesh.ufl_cell(), 1)
    Q = FiniteElement("CG", mesh.ufl_cell(), 1)
    R = FiniteElement("R", mesh.ufl_cell(), 0)

    W = FunctionSpace(mesh, MixedElement([Q, Q, Q, R]))
    W = FunctionSpace(mesh, MixedElement([Q, Q, R, Q]))
    W = FunctionSpace(mesh, V * R)
    W = FunctionSpace(mesh, R * V)
    assert (W)


def test_entity_dofs(mesh):
    """Test that num entity dofs is correctly wrapped to dolfin::DofMap"""
    V = FunctionSpace(mesh, ("CG", 1))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 0

    V = VectorFunctionSpace(mesh, ("CG", 1))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 2
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 0
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 0

    V = FunctionSpace(mesh, ("CG", 2))
    assert V.dofmap.dof_layout.num_entity_dofs(0) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(1) == 1
    assert V.dofmap.dof_layout.num_entity_dofs(2) == 0

    V = FunctionSpace(mesh, ("CG", 3))
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

    V = VectorFunctionSpace(mesh, ("CG", 1))

    # Note this numbering is dependent on FFC and can change This test
    # is here just to check that we get correct numbers mapped from ufc
    # generated code to dolfin
    for i, cdofs in enumerate([[0, 3], [1, 4], [2, 5]]):
        dofs = V.dofmap.dof_layout.entity_dofs(0, i)
        assert all(d == cd for d, cd in zip(dofs, cdofs))


@pytest.mark.skip
@skip_in_parallel
@pytest.mark.parametrize(
    'mesh_factory', [(UnitSquareMesh, (MPI.comm_world, 2, 2)),
                     (UnitSquareMesh,
                      (MPI.comm_world, 2, 2, CellType.quadrilateral))])
def test_entity_closure_dofs(mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)
    tdim = mesh.topology.dim

    for degree in (1, 2, 3):
        V = FunctionSpace(mesh, ("CG", degree))
        for d in range(tdim + 1):
            covered = set()
            covered2 = set()
            all_entities = np.array(
                [entity for entity in range(mesh.num_entities(d))],
                dtype=np.uintp)
            for entity in all_entities:
                entities = np.array([entity], dtype=np.uintp)
                dofs_on_this_entity = V.dofmap.entity_dofs(mesh, d, entities)
                closure_dofs = V.dofmap.entity_closure_dofs(
                    mesh, d, entities)
                assert len(dofs_on_this_entity) == V.dofmap.dof_layout.num_entity_dofs(
                    d)
                assert len(dofs_on_this_entity) <= len(closure_dofs)
                covered.update(dofs_on_this_entity)
                covered2.update(closure_dofs)
            dofs_on_all_entities = V.dofmap.entity_dofs(
                mesh, d, all_entities)
            closure_dofs_on_all_entities = V.dofmap.entity_closure_dofs(
                mesh, d, all_entities)
            assert len(dofs_on_all_entities) == V.dofmap.dof_layout.num_entity_dofs(
                d) * mesh.num_entities(d)
            assert covered == set(dofs_on_all_entities)
            assert covered2 == set(closure_dofs_on_all_entities)
        d = tdim
        all_cells = np.array(
            [entity for entity in range(mesh.num_entities(d))], dtype=np.uintp)
        assert set(V.dofmap.entity_closure_dofs(mesh, d, all_cells)) == set(
            range(V.dim))


@pytest.mark.skip
def test_clear_sub_map_data_scalar(mesh):
    V = FunctionSpace(mesh, ("CG", 2))
    with pytest.raises(ValueError):
        V.sub(1)

    V = VectorFunctionSpace(mesh, ("CG", 2))
    V1 = V.sub(1)
    assert (V1)

    # Clean sub-map data
    V.dofmap.clear_sub_map_data()

    # Can still get previously computed map
    V1 = V.sub(1)

    # New sub-map should throw an error
    with pytest.raises(RuntimeError):
        V.sub(0)


@pytest.mark.skip
def test_clear_sub_map_data_vector(mesh):
    mesh = UnitSquareMesh(8, 8)
    P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, P1 * P1)

    # Check block size
    assert W.dofmap.index_map.block_size == 2

    W.dofmap.clear_sub_map_data()
    with pytest.raises(RuntimeError):
        W0 = W.sub(0)
        assert (W0)
    with pytest.raises(RuntimeError):
        W1 = W.sub(1)
        assert (W1)


@pytest.mark.skip
def test_block_size(mesh):
    meshes = [
        UnitSquareMesh(8, 8),
        UnitCubeMesh(4, 4, 4),
        UnitSquareMesh(8, 8, CellType.quadrilateral),
        UnitCubeMesh(4, 4, 4, CellType.hexahedron)
    ]
    for mesh in meshes:
        P2 = FiniteElement("Lagrange", mesh.ufl_cell(), 2)

        V = FunctionSpace(mesh, P2)
        assert V.dofmap.block_size == 1

        V = FunctionSpace(mesh, P2 * P2)
        assert V.dofmap.index_map.block_size == 2

        for i in range(1, 6):
            W = FunctionSpace(mesh, MixedElement(i * [P2]))
            assert W.dofmap.index_map.block_size == i

        V = VectorFunctionSpace(mesh, ("Lagrange", 2))
        assert V.dofmap.index_map.block_size == mesh.geometry.dim


@pytest.mark.skip
def test_block_size_real(mesh):
    mesh = UnitIntervalMesh(12)
    V = FiniteElement('DG', mesh.ufl_cell(), 0)
    R = FiniteElement('R', mesh.ufl_cell(), 0)
    X = FunctionSpace(mesh, V * R)
    assert X.dofmap.index_map.block_size == 1


@pytest.mark.skip
@pytest.mark.parametrize(
    'mesh_factory', [(UnitSquareMesh, (MPI.comm_world, 4, 4)),
                     (UnitSquareMesh,
                      (MPI.comm_world, 4, 4, CellType.quadrilateral))])
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
        # with pytest.raises(RuntimeError):
        #    dofmap().index_map.size('foo')


# Failures in FFC on quads/hexes
xfail_ffc = pytest.mark.xfail(raises=Exception)


@skip_in_parallel
@pytest.mark.parametrize('space', [
    "FunctionSpace(UnitIntervalMesh(MPI.comm_world, 10),                                        ('P', 1))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.triangle),                ('P', 1))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.tetrahedron),            ('P', 1))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.quadrilateral),           ('Q', 1))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.hexahedron),             ('Q', 1))",
    "FunctionSpace(UnitIntervalMesh(MPI.comm_world, 10),                                        ('P', 2))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.triangle),                ('P', 2))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.tetrahedron),            ('P', 2))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.quadrilateral),           ('Q', 2))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.hexahedron),             ('Q', 2))",
    "FunctionSpace(UnitIntervalMesh(MPI.comm_world, 10),                                        ('P', 3))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.triangle),                ('P', 3))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.tetrahedron),            ('P', 3))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.quadrilateral),           ('Q', 3))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.hexahedron),             ('Q', 3))",
    "FunctionSpace(UnitIntervalMesh(MPI.comm_world, 10),                                        ('DP', 1))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.triangle),                ('DP', 1))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.tetrahedron),            ('DP', 1))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.quadrilateral),           ('DQ', 1))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.hexahedron),             ('DQ', 1))",
    "FunctionSpace(UnitIntervalMesh(MPI.comm_world, 10),                                        ('DP', 2))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.triangle),                ('DP', 2))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.tetrahedron),            ('DP', 2))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.quadrilateral),           ('DQ', 2))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.hexahedron),             ('DQ', 2))",
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.triangle),                ('N1curl', 1))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.tetrahedron),            ('N1curl', 1))",
    pytest.param(
        "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.quadrilateral),       ('N1curl', 1))",
        marks=pytest.mark.xfail),
    pytest.param(
        "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.hexahedron),         ('N1curl', 1))",
        marks=pytest.mark.xfail),
    pytest.param(
        "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.triangle),            ('N1curl', 2))",
        marks=pytest.mark.xfail),
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.tetrahedron),            ('N1curl', 2))",
    pytest.param(
        "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.quadrilateral),       ('N1curl', 2))",
        marks=pytest.mark.xfail),
    pytest.param(
        "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.hexahedron),         ('N1curl', 2))",
        marks=pytest.mark.xfail),
    "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.triangle),                ('RT', 1))",
    "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.tetrahedron),            ('RT', 1))",
    pytest.param(
        "FunctionSpace(UnitSquareMesh(MPI.comm_world, 6, 6, CellType.quadrilateral),       ('RT', 1))",
        marks=pytest.mark.xfail),
    pytest.param(
        "FunctionSpace(UnitCubeMesh(MPI.comm_world, 2, 2, 2, CellType.hexahedron),         ('RT', 1))",
        marks=pytest.mark.xfail)
])
def test_dofs_dim(space):
    """Test function DofMap::dofs(mesh, dim)"""
    V = eval(space)
    dofmap = V.dofmap
    mesh = V.mesh
    for dim in range(0, mesh.topology.dim):
        edofs = dofmap.dofs(mesh, dim)
        if mesh.topology.connectivity(dim, 0) is not None:
            num_mesh_entities = mesh.num_entities(dim)
            dofs_per_entity = dofmap.dof_layout.num_entity_dofs(dim)
            assert len(edofs) == dofs_per_entity * num_mesh_entities


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


@skip_in_parallel
def test_high_order_lagrange():
    """Test simple P3 Lagrange dofmap. Checks that dofs on a shared edged match."""
    def check(mesh, edges):
        """Compute the physical coordinates of the dofs on the given local edges"""
        V = FunctionSpace(mesh, ("Lagrange", 3))

        assert len(edges) == 2
        dofmap = V.dofmap
        dofs = [dofmap.cell_dofs(c) for c in range(len(edges))]
        edge_dofs_local = [dofmap.dof_layout.entity_dofs(1, e) for e in edges]
        for edofs in edge_dofs_local:
            assert len(edofs) == 2
        edge_dofs = [dofs[0][edge_dofs_local[0]], dofs[1][edge_dofs_local[1]]]
        assert set(edge_dofs[0]) == set(edge_dofs[1])

        X = V.element.dof_reference_coordinates()
        coord_dofs = mesh.coordinate_dofs().entity_points()
        x_g = mesh.geometry.points
        x_dofs = []
        cmap = fem.create_coordinate_map(mesh.ufl_domain())
        for c in range(len(edges)):
            x_coord_new = np.zeros([3, 2])
            for v in range(3):
                x_coord_new[v] = x_g[coord_dofs[c, v], :2]
            x = X.copy()
            cmap.compute_physical_coordinates(x, X, x_coord_new)
            x_dofs.append(x[edge_dofs_local[c]])

        return x_dofs

    # Create simple mesh
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = np.array([[0, 1, 2], [2, 3, 0], ])
    mesh = cpp.mesh.Mesh(MPI.comm_world, CellType.triangle, points,
                         cells, [], cpp.mesh.GhostMode.none)
    mesh.create_connectivity(2, 1)

    c21 = mesh.topology.connectivity(2, 1)
    e0 = c21.connections(0)[1]
    e1 = c21.connections(1)[1]
    assert e0 == e1

    # Check un-ordered mesh
    x0, x1 = check(mesh, [1, 1])
    assert not np.allclose(x0, x1)
    x0.sort(axis=0)
    x1.sort(axis=0)
    assert np.allclose(x0, x1)

    # Check ordered mesh
    cpp.mesh.Ordering.order_simplex(mesh)
    c21 = mesh.topology.connectivity(2, 1)
    e0 = c21.connections(0)[1]
    e1 = c21.connections(1)[2]
    assert e0 == e1
    x0, x1 = check(mesh, [1, 2])
    assert np.allclose(x0, x1)
