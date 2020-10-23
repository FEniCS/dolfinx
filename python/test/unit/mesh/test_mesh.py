# Copyright (C) 2006 Anders Logg
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import math
import sys

import FIAT
import numpy as np
import pytest
from dolfinx import (BoxMesh, RectangleMesh, UnitCubeMesh, UnitIntervalMesh,
                     UnitSquareMesh, cpp)
from dolfinx.cpp.mesh import CellType, is_simplex
from dolfinx.fem import assemble_scalar
from dolfinx_utils.test.fixtures import tempdir
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI
from ufl import dx

assert (tempdir)


@pytest.fixture
def mesh1d():
    """Create 1D mesh with degenerate cell"""
    mesh1d = UnitIntervalMesh(MPI.COMM_WORLD, 4)
    i1 = np.where((mesh1d.geometry.x
                   == (0.75, 0, 0)).all(axis=1))[0][0]
    i2 = np.where((mesh1d.geometry.x
                   == (1, 0, 0)).all(axis=1))[0][0]

    mesh1d.geometry.x[i2] = mesh1d.geometry.x[i1]
    return mesh1d


@pytest.fixture
def mesh2d():
    """Create 2D mesh with one equilateral triangle"""
    mesh2d = RectangleMesh(
        MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                         np.array([1., 1., 0.0])], [1, 1],
        CellType.triangle, cpp.mesh.GhostMode.none, 'left')
    i1 = np.where((mesh2d.geometry.x
                   == (1, 1, 0)).all(axis=1))[0][0]
    mesh2d.geometry.x[i1, :2] += 0.5 * (math.sqrt(3.0) - 1.0)
    return mesh2d


@pytest.fixture
def mesh3d():
    """Create 3D mesh with regular tetrahedron and degenerate cells"""
    mesh3d = UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1)
    i1 = np.where((mesh3d.geometry.x
                   == (0, 1, 0)).all(axis=1))[0][0]
    i2 = np.where((mesh3d.geometry.x
                   == (1, 1, 1)).all(axis=1))[0][0]

    mesh3d.geometry.x[i1][0] = 1.0
    mesh3d.geometry.x[i2][1] = 0.0
    return mesh3d


@pytest.fixture
def c0(mesh3d):
    """Original tetrahedron from UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1)"""
    return mesh3d, mesh3d.topology.dim, 0


@pytest.fixture
def c1(mesh3d):
    # Degenerate cell
    return mesh3d, mesh3d.topology.dim, 1


@pytest.fixture
def c5(mesh3d):
    # Regular tetrahedron with edge sqrt(2)
    return mesh3d, mesh3d.topology.dim, 5


@pytest.fixture
def interval():
    return UnitIntervalMesh(MPI.COMM_WORLD, 10)


@pytest.fixture
def square():
    return UnitSquareMesh(MPI.COMM_WORLD, 5, 5)


@pytest.fixture
def rectangle():
    return RectangleMesh(
        MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                         np.array([2.0, 2.0, 0.0])], [5, 5],
        CellType.triangle, cpp.mesh.GhostMode.none)


@pytest.fixture
def cube():
    return UnitCubeMesh(MPI.COMM_WORLD, 3, 3, 3)


@pytest.fixture
def box():
    return BoxMesh(MPI.COMM_WORLD, [np.array([0, 0, 0]),
                                    np.array([2, 2, 2])], [2, 2, 5], CellType.tetrahedron,
                   cpp.mesh.GhostMode.none)


@pytest.fixture
def mesh():
    return UnitSquareMesh(MPI.COMM_WORLD, 3, 3)


def new_comm(comm):
    new_group = comm.group.Incl([0])
    new_comm = comm.Create_group(new_group)
    return new_comm


def test_UFLCell(interval, square, rectangle, cube, box):
    import ufl
    assert ufl.interval == interval.ufl_cell()
    assert ufl.triangle == square.ufl_cell()
    assert ufl.triangle == rectangle.ufl_cell()
    assert ufl.tetrahedron == cube.ufl_cell()
    assert ufl.tetrahedron == box.ufl_cell()


def test_UFLDomain(interval, square, rectangle, cube, box):
    def _check_ufl_domain(mesh):
        domain = mesh.ufl_domain()
        assert mesh.geometry.dim == domain.geometric_dimension()
        assert mesh.topology.dim == domain.topological_dimension()
        assert mesh.ufl_cell() == domain.ufl_cell()

    _check_ufl_domain(interval)
    _check_ufl_domain(square)
    _check_ufl_domain(rectangle)
    _check_ufl_domain(cube)
    _check_ufl_domain(box)


def test_UnitSquareMeshDistributed():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 7)
    assert mesh.topology.index_map(0).size_global == 48
    assert mesh.topology.index_map(2).size_global == 70
    assert mesh.geometry.dim == 2
    assert mesh.mpi_comm().allreduce(mesh.topology.index_map(0).size_local, MPI.SUM) == 48


def test_UnitSquareMeshLocal():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(MPI.COMM_SELF, 5, 7)
    assert mesh.topology.index_map(0).size_global == 48
    assert mesh.topology.index_map(2).size_global == 70
    assert mesh.geometry.dim == 2


def test_UnitCubeMeshDistributed():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 7, 9)
    assert mesh.topology.index_map(0).size_global == 480
    assert mesh.topology.index_map(3).size_global == 1890
    assert mesh.geometry.dim == 3
    assert mesh.mpi_comm().allreduce(mesh.topology.index_map(0).size_local, MPI.SUM) == 480


def test_UnitCubeMeshLocal():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.COMM_SELF, 5, 7, 9)
    assert mesh.topology.index_map(0).size_global == 480
    assert mesh.topology.index_map(0).size_local == 480
    assert mesh.topology.index_map(3).size_global == 1890
    assert mesh.topology.index_map(3).size_local == 1890
    assert mesh.geometry.dim == 3


def test_UnitQuadMesh():
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 7, CellType.quadrilateral)
    assert mesh.topology.index_map(0).size_global == 48
    assert mesh.topology.index_map(2).size_global == 35
    assert mesh.geometry.dim == 2
    assert mesh.mpi_comm().allreduce(mesh.topology.index_map(0).size_local, MPI.SUM) == 48


def test_UnitHexMesh():
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 5, 7, 9, CellType.hexahedron)
    assert mesh.topology.index_map(0).size_global == 480
    assert mesh.topology.index_map(3).size_global == 315
    assert mesh.geometry.dim == 3
    assert mesh.mpi_comm().allreduce(mesh.topology.index_map(0).size_local, MPI.SUM) == 480


def test_hash():
    h1 = UnitSquareMesh(MPI.COMM_WORLD, 4, 4).hash()
    h2 = UnitSquareMesh(MPI.COMM_WORLD, 4, 5).hash()
    h3 = UnitSquareMesh(MPI.COMM_WORLD, 4, 4).hash()
    assert h1 == h3
    assert h1 != h2


@skip_in_parallel
def test_GetCoordinates():
    """Get coordinates of vertices"""
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    assert len(mesh.geometry.x) == 36


@skip_in_parallel
def xtest_cell_inradius(c0, c1, c5):
    assert cpp.mesh.inradius(c0[0], [c0[2]]) == pytest.approx((3.0 - math.sqrt(3.0)) / 6.0)
    assert cpp.mesh.inradius(c1[0], [c1[2]]) == pytest.approx(0.0)
    assert cpp.mesh.inradius(c5[0], [c5[2]]) == pytest.approx(math.sqrt(3.0) / 6.0)


@skip_in_parallel
def test_cell_circumradius(c0, c1, c5):
    assert cpp.mesh.circumradius(c0[0], [c0[2]], c0[1]) == pytest.approx(math.sqrt(3.0) / 2.0)
    # Implementation of diameter() does not work accurately
    # for degenerate cells - sometimes yields NaN
    r_c1 = cpp.mesh.circumradius(c1[0], [c1[2]], c1[1])
    assert math.isnan(r_c1)
    assert cpp.mesh.circumradius(c5[0], [c5[2]], c5[1]) == pytest.approx(math.sqrt(3.0) / 2.0)


@skip_in_parallel
def test_cell_h(c0, c1, c5):
    for c in [c0, c1, c5]:
        assert cpp.mesh.h(c[0], [c[2]], c[1]) == pytest.approx(math.sqrt(2.0))


@skip_in_parallel
def xtest_cell_radius_ratio(c0, c1, c5):
    assert cpp.mesh.radius_ratio(c0[0], c0[2]) == pytest.approx(math.sqrt(3.0) - 1.0)
    assert np.isnan(cpp.mesh.radius_ratio(c1[0], c1[2]))
    assert cpp.mesh.radius_ratio(c5[0], c5[2]) == pytest.approx(1.0)


@skip_in_parallel
def test_hmin_hmax(mesh1d, mesh2d, mesh3d):
    assert mesh1d.hmin() == pytest.approx(0.0)
    assert mesh1d.hmax() == pytest.approx(0.25)
    assert mesh2d.hmin() == pytest.approx(math.sqrt(2.0))
    assert mesh2d.hmax() == pytest.approx(math.sqrt(2.0))
    assert mesh3d.hmin() == pytest.approx(math.sqrt(2.0))
    assert mesh3d.hmax() == pytest.approx(math.sqrt(2.0))


@skip_in_parallel
def test_rmin_rmax(mesh1d, mesh2d, mesh3d):
    assert round(mesh1d.rmin() - 0.0, 7) == 0
    # assert round(mesh1d.rmax() - 0.125, 7) == 0
    # assert round(mesh2d.rmin() - 1.0 / (2.0 + math.sqrt(2.0)), 7) == 0
    # assert round(mesh2d.rmax() - math.sqrt(6.0) / 6.0, 7) == 0
    # assert round(mesh3d.rmin() - 0.0, 7) == 0
    # assert round(mesh3d.rmax() - math.sqrt(3.0) / 6.0, 7) == 0

# - Facilities to run tests on combination of meshes


mesh_factories = [
    (UnitIntervalMesh, (MPI.COMM_WORLD, 8)),
    (UnitSquareMesh, (MPI.COMM_WORLD, 4, 4)),
    (UnitCubeMesh, (MPI.COMM_WORLD, 2, 2, 2)),
    (UnitSquareMesh, (MPI.COMM_WORLD, 4, 4, CellType.quadrilateral)),
    (UnitCubeMesh, (MPI.COMM_WORLD, 2, 2, 2, CellType.hexahedron)),
    # FIXME: Add mechanism for testing meshes coming from IO
]


# FIXME: Fix this xfail
def xfail_ghosted_quads_hexes(mesh_factory, ghost_mode):
    """Xfail when mesh_factory on quads/hexes uses
    shared_vertex mode. Needs implementing.
    """
    if mesh_factory in [UnitSquareMesh, UnitCubeMesh]:
        if ghost_mode == cpp.mesh.GhostMode.shared_vertex:
            pytest.xfail(reason="Missing functionality in \'{}\' with \'{}\' mode".format(mesh_factory, ghost_mode))


@pytest.mark.parametrize('mesh_factory', mesh_factories)
def test_mesh_topology_against_fiat(mesh_factory, ghost_mode=cpp.mesh.GhostMode.none):
    """Test that mesh cells have topology matching to FIAT reference
    cell they were created from.
    """
    func, args = mesh_factory
    xfail_ghosted_quads_hexes(func, ghost_mode)
    mesh = func(*args)
    if not is_simplex(mesh.topology.cell_type):
        return

    # Create FIAT cell
    cell_name = cpp.mesh.to_string(mesh.topology.cell_type)
    fiat_cell = FIAT.ufc_cell(cell_name)

    # Initialize all mesh entities and connectivities
    mesh.topology.create_connectivity_all()

    map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map.size_local + map.num_ghosts
    for i in range(num_cells):
        # Get indices of cell vertices
        vertex_global_indices = mesh.topology.connectivity(mesh.topology.dim, 0).links(i)

        # Loop over all dimensions of reference cell topology
        for d, d_topology in fiat_cell.get_topology().items():

            # Get entities of dimension d on the cell
            entities = mesh.topology.connectivity(mesh.topology.dim, d).links(i)
            if len(entities) == 0:  # Fixup for highest dimension
                entities = (i, )

            # Loop over all entities of fixed dimension d
            for entity_index, entity_topology in d_topology.items():

                # Check that entity vertices map to cell vertices in correct order
                vertices = mesh.topology.connectivity(d, 0).links(entities[entity_index])
                vertices_dolfin = np.sort(vertices)
                vertices_fiat = np.sort(vertex_global_indices[np.array(entity_topology)])
                assert all(vertices_fiat == vertices_dolfin)


def test_mesh_topology_lifetime():
    """Check that lifetime of Mesh.topology is bound to underlying mesh object"""
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
    rc = sys.getrefcount(mesh)
    topology = mesh.topology
    assert sys.getrefcount(mesh) == rc + 1
    del topology
    assert sys.getrefcount(mesh) == rc


@skip_in_parallel
def test_small_mesh():
    mesh3d = UnitCubeMesh(MPI.COMM_WORLD, 1, 1, 1)
    gdim = mesh3d.geometry.dim
    assert mesh3d.topology.index_map(gdim).size_global == 6

    mesh2d = UnitSquareMesh(MPI.COMM_WORLD, 1, 1)
    gdim = mesh2d.geometry.dim
    assert mesh2d.topology.index_map(gdim).size_global == 2

    mesh1d = UnitIntervalMesh(MPI.COMM_WORLD, 2)
    gdim = mesh1d.geometry.dim
    assert mesh1d.topology.index_map(gdim).size_global == 2


def test_UnitHexMesh_assemble():
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 6, 7, 5, CellType.hexahedron)
    vol = assemble_scalar(1 * dx(mesh))
    vol = mesh.mpi_comm().allreduce(vol, MPI.SUM)
    assert(vol == pytest.approx(1, rel=1e-9))
