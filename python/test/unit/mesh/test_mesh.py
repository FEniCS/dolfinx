# Copyright (C) 2006 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import sys
from math import sqrt

import numpy
import pytest

import dolfin
import FIAT
from dolfin import (MPI, BoxMesh, Cell, Cells, CellType, MeshEntities,
                    MeshEntity, MeshFunction, Point, RectangleMesh,
                    UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh, Vertex,
                    cpp)
from dolfin.io import XDMFFile
from dolfin_utils.test.fixtures import fixture
from dolfin_utils.test.skips import skip_in_parallel


@fixture
def mesh1d():
    # Create 1D mesh with degenerate cell
    mesh1d = UnitIntervalMesh(MPI.comm_world, 4)
    mesh1d.geometry.points[4] = mesh1d.geometry.points[3]
    return mesh1d


@fixture
def mesh2d():
    # Create 2D mesh with one equilateral triangle
    mesh2d = RectangleMesh(
        MPI.comm_world, [Point(0, 0)._cpp_object,
                         Point(1, 1)._cpp_object], [1, 1],
        CellType.Type.triangle, cpp.mesh.GhostMode.none, 'left')
    mesh2d.geometry.points[3] += 0.5 * (sqrt(3.0) - 1.0)
    return mesh2d


@fixture
def mesh3d():
    # Create 3D mesh with regular tetrahedron and degenerate cells
    mesh3d = UnitCubeMesh(MPI.comm_world, 1, 1, 1)
    mesh3d.geometry.points[6][0] = 1.0
    mesh3d.geometry.points[3][1] = 0.0
    return mesh3d


@fixture
def c0(mesh3d):
    # Original tetrahedron from UnitCubeMesh(MPI.comm_world, 1, 1, 1)
    return Cell(mesh3d, 0)


@fixture
def c1(mesh3d):
    # Degenerate cell
    return Cell(mesh3d, 1)


@fixture
def c5(mesh3d):
    # Regular tetrahedron with edge sqrt(2)
    return Cell(mesh3d, 5)


@fixture
def interval():
    return UnitIntervalMesh(MPI.comm_world, 10)


@fixture
def square():
    return UnitSquareMesh(MPI.comm_world, 5, 5)


@fixture
def rectangle():
    return RectangleMesh(
        MPI.comm_world, [Point(0, 0)._cpp_object,
                         Point(2, 2)._cpp_object], [5, 5],
        CellType.Type.triangle, cpp.mesh.GhostMode.none)


@fixture
def cube():
    return UnitCubeMesh(MPI.comm_world, 3, 3, 3)


@fixture
def box():
    return BoxMesh(
        MPI.comm_world,
        [Point(0, 0, 0)._cpp_object,
         Point(2, 2, 2)._cpp_object], [2, 2, 5], CellType.Type.tetrahedron,
        cpp.mesh.GhostMode.none)


@fixture
def mesh():
    return UnitSquareMesh(MPI.comm_world, 3, 3)


@fixture
def f(mesh):
    return MeshFunction('int', mesh, 0, 0)


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
        assert mesh.id == domain.ufl_id()

    _check_ufl_domain(interval)
    _check_ufl_domain(square)
    _check_ufl_domain(rectangle)
    _check_ufl_domain(cube)
    _check_ufl_domain(box)


# pygmsh is problematic in parallel because it uses subprocess to call
# gmsh. To be robust, it would need to call MPI 'spawn'.
@pytest.mark.skip(
    reason="pymsh calling gmsh fails in container (related to file creation)")
@skip_in_parallel
def test_mesh_construction_pygmsh():

    import pygmsh

    if MPI.rank(MPI.comm_world) == 0:
        geom = pygmsh.opencascade.Geometry()
        geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
        points, cells, _, _, _ = pygmsh.generate_mesh(geom)
    else:
        points = numpy.zeros([0, 3])
        cells = {
            "tetra": numpy.zeros([0, 4], dtype=numpy.int64),
            "triangle": numpy.zeros([0, 3], dtype=numpy.int64),
            "line": numpy.zeros([0, 2], dtype=numpy.int64)
        }

    mesh = dolfin.cpp.mesh.Mesh(
        MPI.comm_world, dolfin.cpp.mesh.CellType.Type.tetrahedron, points,
        cells['tetra'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 3

    mesh = dolfin.cpp.mesh.Mesh(MPI.comm_world,
                                dolfin.cpp.mesh.CellType.Type.triangle, points,
                                cells['triangle'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 2

    mesh = dolfin.cpp.mesh.Mesh(MPI.comm_world,
                                dolfin.cpp.mesh.CellType.Type.interval, points,
                                cells['line'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 1

    if MPI.rank(MPI.comm_world) == 0:
        print("Generate mesh")
        geom = pygmsh.opencascade.Geometry()
        geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
        points, cells, _, _, _ = pygmsh.generate_mesh(
            geom, extra_gmsh_arguments=['-order', '2'])
        print("End Generate mesh", cells.keys())
    else:
        points = numpy.zeros([0, 3])
        cells = {
            "tetra10": numpy.zeros([0, 10], dtype=numpy.int64),
            "triangle6": numpy.zeros([0, 6], dtype=numpy.int64),
            "line3": numpy.zeros([0, 3], dtype=numpy.int64)
        }

    mesh = dolfin.cpp.mesh.Mesh(
        MPI.comm_world, dolfin.cpp.mesh.CellType.Type.tetrahedron, points,
        cells['tetra10'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 2
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 3

    mesh = dolfin.cpp.mesh.Mesh(
        MPI.comm_world, dolfin.cpp.mesh.CellType.Type.triangle, points,
        cells['triangle6'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 2
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 2


def test_UnitSquareMeshDistributed():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(MPI.comm_world, 5, 7)
    assert mesh.num_entities_global(0) == 48
    assert mesh.num_entities_global(2) == 70
    assert mesh.geometry.dim == 2


def test_UnitSquareMeshLocal():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(MPI.comm_self, 5, 7)
    assert mesh.num_vertices() == 48
    assert mesh.num_cells() == 70
    assert mesh.geometry.dim == 2


def test_UnitCubeMeshDistributed():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 1890
    assert mesh.geometry.dim == 3


def test_UnitCubeMeshLocal():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_self, 5, 7, 9)
    assert mesh.num_vertices() == 480
    assert mesh.num_cells() == 1890
    assert mesh.geometry.dim == 3


def test_UnitQuadMesh():
    mesh = UnitSquareMesh(MPI.comm_world, 5, 7, CellType.Type.quadrilateral)
    assert mesh.num_entities_global(0) == 48
    assert mesh.num_entities_global(2) == 35
    assert mesh.geometry.dim == 2


def test_UnitHexMesh():
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9, CellType.Type.hexahedron)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 315
    assert mesh.geometry.dim == 3


@skip_in_parallel
def test_Assign(mesh, f):
    """Assign value of mesh function."""
    f = f
    f[3] = 10
    v = Vertex(mesh, 3)
    assert f[v] == 10


@skip_in_parallel
def test_Write(f):
    """Construct and save a simple meshfunction."""
    f = f
    f[0] = 1
    f[1] = 2
    file = XDMFFile(f.mesh().mpi_comm(), "saved_mesh_function.xdmf")
    file.write(f)


def test_hash():
    h1 = UnitSquareMesh(MPI.comm_world, 4, 4).hash()
    h2 = UnitSquareMesh(MPI.comm_world, 4, 5).hash()
    h3 = UnitSquareMesh(MPI.comm_world, 4, 4).hash()
    assert h1 == h3
    assert h1 != h2


@skip_in_parallel
def test_GetCoordinates():
    """Get coordinates of vertices"""
    mesh = UnitSquareMesh(MPI.comm_world, 5, 5)
    assert len(mesh.geometry.points) == 36


def test_GetCells():
    """Get cells of mesh"""
    mesh = UnitSquareMesh(MPI.comm_world, 5, 5)
    assert MPI.sum(mesh.mpi_comm(), len(mesh.cells())) == 50


@skip_in_parallel
def test_cell_inradius(c0, c1, c5):
    assert round(c0.inradius() - (3.0 - sqrt(3.0)) / 6.0, 7) == 0
    assert round(c1.inradius() - 0.0, 7) == 0
    assert round(c5.inradius() - sqrt(3.0) / 6.0, 7) == 0


@skip_in_parallel
def test_cell_circumradius(c0, c1, c5):
    from math import isnan
    assert round(c0.circumradius() - sqrt(3.0) / 2.0, 7) == 0
    # Implementation of diameter() does not work accurately
    # for degenerate cells - sometimes yields NaN
    assert isnan(c1.circumradius())
    assert round(c5.circumradius() - sqrt(3.0) / 2.0, 7) == 0


@skip_in_parallel
def test_cell_h(c0, c1, c5):
    assert round(c0.h() - sqrt(2.0), 7) == 0
    assert round(c1.h() - sqrt(2.0), 7) == 0
    assert round(c5.h() - sqrt(2.0), 7) == 0


@skip_in_parallel
def test_cell_radius_ratio(c0, c1, c5):
    assert round(c0.radius_ratio() - sqrt(3.0) + 1.0, 7) == 0
    assert round(c1.radius_ratio() - 0.0, 7) == 0
    assert round(c5.radius_ratio() - 1.0, 7) == 0


@skip_in_parallel
def xtest_hmin_hmax(mesh1d, mesh2d, mesh3d):
    assert round(mesh1d.hmin() - 0.0, 7) == 0
    assert round(mesh1d.hmax() - 0.25, 7) == 0
    assert round(mesh2d.hmin() - sqrt(2.0), 7) == 0
    assert round(mesh2d.hmax() - 2.0 * sqrt(6.0) / 3.0, 7) == 0
    # nans are not taken into account in hmax and hmin
    assert round(mesh3d.hmin() - sqrt(3.0), 7) == 0
    assert round(mesh3d.hmax() - sqrt(3.0), 7) == 0


@skip_in_parallel
def test_rmin_rmax(mesh1d, mesh2d, mesh3d):
    assert round(mesh1d.rmin() - 0.0, 7) == 0
    assert round(mesh1d.rmax() - 0.125, 7) == 0
    assert round(mesh2d.rmin() - 1.0 / (2.0 + sqrt(2.0)), 7) == 0
    assert round(mesh2d.rmax() - sqrt(6.0) / 6.0, 7) == 0
    assert round(mesh3d.rmin() - 0.0, 7) == 0
    assert round(mesh3d.rmax() - sqrt(3.0) / 6.0, 7) == 0


# - Facilities to run tests on combination of meshes

mesh_factories = [
    (UnitIntervalMesh, (
        MPI.comm_world,
        8,
    )),
    (UnitSquareMesh, (MPI.comm_world, 4, 4)),
    (UnitCubeMesh, (MPI.comm_world, 2, 2, 2)),
    (UnitSquareMesh, (MPI.comm_world, 4, 4, CellType.Type.quadrilateral)),
    (UnitCubeMesh, (MPI.comm_world, 2, 2, 2, CellType.Type.hexahedron)),
    # FIXME: Add mechanism for testing meshes coming from IO
]

mesh_factories_broken_shared_entities = [
    (UnitIntervalMesh, (
        MPI.comm_world,
        8,
    )),
    (UnitSquareMesh, (MPI.comm_world, 4, 4)),
    # FIXME: Problem in test_shared_entities
    (UnitCubeMesh, (MPI.comm_world, 2, 2, 2)),
    (UnitSquareMesh, (MPI.comm_world, 4, 4, CellType.Type.quadrilateral)),
    (UnitCubeMesh, (MPI.comm_world, 2, 2, 2, CellType.Type.hexahedron)),
]

# FIXME: Fix this xfail


def xfail_ghosted_quads_hexes(mesh_factory, ghost_mode):
    """Xfail when mesh_factory on quads/hexes uses
    shared_vertex mode. Needs implementing.
    """
    if mesh_factory in [UnitSquareMesh, UnitCubeMesh]:
        if ghost_mode == cpp.mesh.GhostMode.shared_vertex:
            pytest.xfail(reason="Missing functionality in '{}' with '' "
                         "mode".format(mesh_factory, ghost_mode))


@pytest.mark.parametrize('mesh_factory', mesh_factories_broken_shared_entities)
def test_shared_entities(mesh_factory):
    func, args = mesh_factory
    # xfail_ghosted_quads_hexes(func, ghost_mode)
    mesh = func(*args)
    dim = mesh.topology.dim

    # FIXME: Implement a proper test
    for shared_dim in range(dim + 1):
        # Initialise global indices (if not already)
        mesh.init_global(shared_dim)

        assert isinstance(mesh.topology.shared_entities(shared_dim), dict)
        assert isinstance(
            mesh.topology.global_indices(shared_dim), numpy.ndarray)

        if mesh.topology.have_shared_entities(shared_dim):
            for e in MeshEntities(mesh, shared_dim):
                sharing = e.sharing_processes()
                assert isinstance(sharing, set)
                assert (len(sharing) > 0) == e.is_shared()

        shared_entities = mesh.topology.shared_entities(shared_dim)

        # Check that sum(local-shared) = global count
        rank = MPI.rank(mesh.mpi_comm())
        ct = sum(1 for val in shared_entities.values() if list(val)[0] < rank)
        num_entities_global = MPI.sum(mesh.mpi_comm(),
                                      mesh.num_entities(shared_dim) - ct)

        assert num_entities_global == mesh.num_entities_global(shared_dim)


# Skipping test after removing mesh.order()
@pytest.mark.skip
@pytest.mark.parametrize('mesh_factory', mesh_factories)
def test_mesh_topology_against_fiat(mesh_factory, ghost_mode):
    """Test that mesh cells have topology matching to FIAT reference
    cell they were created from.
    """
    func, args = mesh_factory
    xfail_ghosted_quads_hexes(func, ghost_mode)
    mesh = func(*args)

    # Create FIAT cell
    cell_name = CellType.type2string(mesh.type().cell_type())
    fiat_cell = FIAT.ufc_cell(cell_name)

    # Initialize all mesh entities and connectivities
    mesh.init()

    for cell in Cells(mesh):
        # Get mesh-global (MPI-local) indices of cell vertices
        vertex_global_indices = cell.entities(0)

        # Loop over all dimensions of reference cell topology
        for d, d_topology in fiat_cell.get_topology().items():

            # Get entities of dimension d on the cell
            entities = cell.entities(d)
            if len(entities) == 0:  # Fixup for highest dimension
                entities = (cell.index(), )

            # Loop over all entities of fixed dimension d
            for entity_index, entity_topology in d_topology.items():

                # Check that entity vertices map to cell vertices in right order
                entity = MeshEntity(mesh, d, entities[entity_index])
                entity_vertices = entity.entities(0)
                assert all(vertex_global_indices[numpy.array(entity_topology)]
                           == entity_vertices)


def test_mesh_topology_reference():
    """Check that Mesh.topology returns a reference rather
    than copy"""
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
    assert mesh.topology.id == mesh.topology.id


def test_mesh_topology_lifetime():
    """Check that lifetime of Mesh.topology is bound to
    underlying mesh object"""
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)

    rc = sys.getrefcount(mesh)
    topology = mesh.topology
    assert sys.getrefcount(mesh) == rc + 1
    del topology
    assert sys.getrefcount(mesh) == rc
