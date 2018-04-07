# Copyright (C) 2006 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
import numpy

import sys
import os

import FIAT
from dolfin import *
from dolfin_utils.test import fixture, set_parameters_fixture
from dolfin_utils.test import skip_in_parallel, xfail_in_parallel
from dolfin_utils.test import cd_tempdir


@fixture
def mesh1d():
    # Create 1D mesh with degenerate cell
    mesh1d = UnitIntervalMesh(MPI.comm_world, 4)
    mesh1d.geometry.points[4] = mesh1d.geometry.points[3]
    return mesh1d


@fixture
def mesh2d():
    # Create 2D mesh with one equilateral triangle
    mesh2d = RectangleMesh.create(MPI.comm_world, [Point(0, 0), Point(1, 1)],
                                  [1, 1], CellType.Type.triangle, 'left')
    mesh2d.geometry.points[3] += 0.5*(sqrt(3.0)-1.0)
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
    return RectangleMesh.create(MPI.comm_world, [Point(0, 0), Point(2, 2)], [5, 5], CellType.Type.triangle)


@fixture
def cube():
    return UnitCubeMesh(MPI.comm_world, 3, 3, 3)


@fixture
def box():
    return BoxMesh.create(MPI.comm_world, [Point(0, 0, 0), Point(2, 2, 2)], [2, 2, 5], CellType.Type.tetrahedron)


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
    import ufl

    def _check_ufl_domain(mesh):
        domain = mesh.ufl_domain()
        assert mesh.geometry.dim == domain.geometric_dimension()
        assert mesh.topology.dim == domain.topological_dimension()
        assert mesh.ufl_cell() == domain.ufl_cell()
        assert mesh.id() == domain.ufl_id()

    _check_ufl_domain(interval)
    _check_ufl_domain(square)
    _check_ufl_domain(rectangle)
    _check_ufl_domain(cube)
    _check_ufl_domain(box)


def test_UnitSquareMesh():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(MPI.comm_world, 5, 7)
    assert mesh.num_entities_global(0) == 48
    assert mesh.num_entities_global(2) == 70


def test_UnitSquareMeshDistributed():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(MPI.comm_world, 5, 7)
    assert mesh.num_entities_global(0) == 48
    assert mesh.num_entities_global(2) == 70


def test_UnitSquareMeshLocal():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(MPI.comm_self, 5, 7)
    assert mesh.num_vertices() == 48
    assert mesh.num_cells() == 70


def test_UnitCubeMesh():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 1890


def test_UnitCubeMeshDistributed():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 1890


def test_UnitCubeMeshDistributedLocal():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_self, 5, 7, 9)
    assert mesh.num_vertices() == 480
    assert mesh.num_cells() == 1890


def test_UnitQuadMesh():
    mesh = UnitSquareMesh(MPI.comm_world, 5, 7, CellType.Type.quadrilateral)
    assert mesh.num_entities_global(0) == 48
    assert mesh.num_entities_global(2) == 35


def test_UnitHexMesh():
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9, CellType.Type.hexahedron)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 315


@skip_in_parallel
def test_Assign(mesh, f):
    """Assign value of mesh function."""
    f = f
    f[3] = 10
    v = Vertex(mesh, 3)
    assert f[v] == 10


@skip_in_parallel
def test_Write(cd_tempdir, f):
    """Construct and save a simple meshfunction."""
    f = f
    f[0] = 1
    f[1] = 2
    file = XDMFFile(f.mesh().mpi_comm(), "saved_mesh_function.xdmf")
    file.write(f, XDMFFile.Encoding.ASCII)


@skip_in_parallel
def test_Read(cd_tempdir):
    """Construct and save a simple meshfunction. Then read it back from
    file."""
    # mf = mesh.data().create_mesh_function("mesh_data_function", 2)
    # print "***************", mf
    # mf[0] = 3
    # mf[1] = 4

    # f[0] = 1
    # f[1] = 2
    # file = File("saved_mesh_function.xml")
    # file << f
    # f = MeshFunction('int', mesh, "saved_mesh_function.xml")
    # assert all(f.array() == f.array())


def test_hash():
    h1 = UnitSquareMesh(MPI.comm_world, 4, 4).hash()
    h2 = UnitSquareMesh(MPI.comm_world, 4, 5).hash()
    h3 = UnitSquareMesh(MPI.comm_world, 4, 4).hash()
    assert h1 == h3
    assert h1 != h2


# FIXME: Mesh IO tests should be in io test directory
@skip_in_parallel
def test_MeshXML2D(cd_tempdir):
    """Write and read 2D mesh to/from file"""
    mesh_out = UnitSquareMesh(MPI.comm_world, 3, 3)
    file = XDMFFile(mesh_out.mpi_comm(), "unitsquare.xdmf")
    file.write(mesh_out, XDMFFile.Encoding.ASCII)
    mesh_in = file.read_mesh(MPI.comm_world)
    assert mesh_in.num_vertices() == 16


@skip_in_parallel
def test_MeshXML3D(cd_tempdir):
    """Write and read 3D mesh to/from file"""
    mesh_out = UnitCubeMesh(MPI.comm_world, 3, 3, 3)
    file = XDMFFile(mesh_out.mpi_comm(), "unitcube.xdmf")
    file.write(mesh_out, XDMFFile.Encoding.ASCII)
    mesh_in = file.read_mesh(MPI.comm_world)
    assert mesh_in.num_vertices() == 64


@skip_in_parallel
def xtest_MeshFunction(cd_tempdir):
    """Write and read mesh function to/from file"""
    mesh = UnitSquareMesh(MPI.comm_world, 1, 1)
    f = MeshFunction('int', mesh, 0)
    f[0] = 2
    f[1] = 4
    f[2] = 6
    f[3] = 8
    file = XDMFFile(mesh.mpi_comm(), "meshfunction.xdmf")
    file.write(f, XDMFFile.Encoding.ASCII)
    g = MeshFunction('int', mesh, 0)
    file.read(g)
    for v in vertices(mesh):
        assert f[v] == g[v]


def test_GetGeometricalDimension():
    """Get geometrical dimension of mesh"""
    mesh = UnitSquareMesh(MPI.comm_world, 5, 5)
    assert mesh.geometry.dim == 2


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
    assert round(c0.inradius() - (3.0-sqrt(3.0))/6.0, 7) == 0
    assert round(c1.inradius() - 0.0, 7) == 0
    assert round(c5.inradius() - sqrt(3.0)/6.0, 7) == 0


@skip_in_parallel
def test_cell_circumradius(c0, c1, c5):
    from math import isnan
    assert round(c0.circumradius() - sqrt(3.0)/2.0, 7) == 0
    # Implementation of diameter() does not work accurately
    # for degenerate cells - sometimes yields NaN
    assert isnan(c1.circumradius())
    assert round(c5.circumradius() - sqrt(3.0)/2.0, 7) == 0


@skip_in_parallel
def test_cell_h(c0, c1, c5):
    from math import isnan
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
    assert round(mesh2d.hmax() - 2.0*sqrt(6.0)/3.0, 7) == 0
    # nans are not taken into account in hmax and hmin
    assert round(mesh3d.hmin() - sqrt(3.0), 7) == 0
    assert round(mesh3d.hmax() - sqrt(3.0), 7) == 0


@skip_in_parallel
def test_rmin_rmax(mesh1d, mesh2d, mesh3d):
    assert round(mesh1d.rmin() - 0.0, 7) == 0
    assert round(mesh1d.rmax() - 0.125, 7) == 0
    assert round(mesh2d.rmin() - 1.0/(2.0+sqrt(2.0)), 7) == 0
    assert round(mesh2d.rmax() - sqrt(6.0)/6.0, 7) == 0
    assert round(mesh3d.rmin() - 0.0, 7) == 0
    assert round(mesh3d.rmax() - sqrt(3.0)/6.0, 7) == 0

# - Facilities to run tests on combination of meshes


ghost_mode = set_parameters_fixture("ghost_mode", [
    "none",
    "shared_facet",
    "shared_vertex",
])

mesh_factories = [
    (UnitIntervalMesh, (MPI.comm_world, 8,)),
    (UnitSquareMesh, (MPI.comm_world, 4, 4)),
    (UnitCubeMesh, (MPI.comm_world, 2, 2, 2)),
    (UnitSquareMesh, (MPI.comm_world, 4, 4, CellType.Type.quadrilateral)),
    (UnitCubeMesh, (MPI.comm_world, 2, 2, 2, CellType.Type.hexahedron)),
    # FIXME: Add mechanism for testing meshes coming from IO
]

mesh_factories_broken_shared_entities = [
    (UnitIntervalMesh, (MPI.comm_world, 8,)),
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
        if ghost_mode == 'shared_vertex':
            pytest.xfail(reason="Missing functionality in '{}' with '' "
                         "mode".format(mesh_factory, ghost_mode))


@pytest.mark.parametrize('mesh_factory', mesh_factories_broken_shared_entities)
def test_shared_entities(mesh_factory, ghost_mode):
    func, args = mesh_factory
    xfail_ghosted_quads_hexes(func, ghost_mode)
    mesh = func(*args)
    dim = mesh.topology.dim

    # FIXME: Implement a proper test
    for shared_dim in range(dim + 1):
        # Initialise global indices (if not already)
        mesh.init_global(shared_dim)

        assert isinstance(mesh.topology.shared_entities(shared_dim), dict)
        assert isinstance(mesh.topology.global_indices(shared_dim),
                          numpy.ndarray)

        if mesh.topology.have_shared_entities(shared_dim):
            for e in MeshEntities(mesh, shared_dim):
                sharing = e.sharing_processes()
                assert isinstance(sharing, set)
                assert (len(sharing) > 0) == e.is_shared()

        n_entities = mesh.num_entities(shared_dim)
        n_global_entities = mesh.num_entities_global(shared_dim)
        shared_entities = mesh.topology.shared_entities(shared_dim)

        # Check that sum(local-shared) = global count
        rank = MPI.rank(mesh.mpi_comm())
        ct = sum(1 for val in shared_entities.values() if list(val)[0] < rank)
        num_entities_global = MPI.sum(
            mesh.mpi_comm(), mesh.num_entities(shared_dim) - ct)

        assert num_entities_global == mesh.num_entities_global(shared_dim)


@pytest.mark.parametrize('mesh_factory', mesh_factories)
def test_mesh_topology_against_fiat(mesh_factory, ghost_mode):
    """Test that mesh cells have topology matching to FIAT reference
    cell they were created from.
    """
    func, args = mesh_factory
    xfail_ghosted_quads_hexes(func, ghost_mode)
    mesh = func(*args)
    assert mesh.ordered()
    tdim = mesh.topology.dim

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
                entities = (cell.index(),)

            # Loop over all entities of fixed dimension d
            for entity_index, entity_topology in d_topology.items():

                # Check that entity vertices map to cell vertices in right order
                entity = MeshEntity(mesh, d, entities[entity_index])
                entity_vertices = entity.entities(0)
                assert all(vertex_global_indices[numpy.array(entity_topology)]
                           == entity_vertices)


@pytest.mark.parametrize('mesh_factory', mesh_factories)
def test_mesh_ufc_ordering(mesh_factory, ghost_mode):
    """Test that DOLFIN follows that UFC standard in numbering
    mesh entities. See chapter 5 of UFC manual
    https://fenicsproject.org/pub/documents/ufc/ufc-user-manual/ufc-user-manual.pdf

    In fact, numbering of other mesh entities than vertices is
    not followed.
    """
    func, args = mesh_factory
    xfail_ghosted_quads_hexes(func, ghost_mode)
    mesh = func(*args)
    assert mesh.ordered()
    tdim = mesh.topology.dim

    # Loop over pair of dimensions d, d1 with d>d1
    for d in range(tdim+1):
        for d1 in range(d):

            # NOTE: DOLFIN UFC noncompliance!
            # DOLFIN has increasing indices only for d-0 incidence
            # with any d; UFC convention for d-d1 with d>d1>0 is not
            # respected in DOLFIN
            if d1 != 0:
                continue

            # Initialize d-d1 connectivity and d1 global indices
            mesh.init(d, d1)
            mesh.init_global(d1)
            assert mesh.topology.have_global_indices(d1)

            # Loop over entities of dimension d
            for e in MeshEntities(mesh, d):

                # Get global indices
                subentities_indices = [e1.global_index()
                                       for e1 in EntityRange(e, d1)]
                assert subentities_indices.count(-1) == 0

                # Check that d1-subentities of d-entity have increasing indices
                assert sorted(subentities_indices) == subentities_indices


def test_mesh_topology_reference():
    """Check that Mesh.topology returns a reference rather
    than copy"""
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
    assert mesh.topology.id() == mesh.topology.id()


def test_mesh_topology_lifetime():
    """Check that lifetime of Mesh.topology is bound to
    underlying mesh object"""
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)

    rc = sys.getrefcount(mesh)
    topology = mesh.topology
    assert sys.getrefcount(mesh) == rc + 1
    del topology
    assert sys.getrefcount(mesh) == rc


def test_mesh_connectivity_lifetime():
    """Check that lifetime of MeshConnectivity is bound to
    underlying mesh topology object"""
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
    mesh.init(1, 2)
    topology = mesh.topology

    # Refcount checks on the MeshConnectivity object
    rc = sys.getrefcount(topology)
    connectivity = topology.connectivity(1, 2)
    assert sys.getrefcount(topology) == rc + 1
    del connectivity
    assert sys.getrefcount(topology) == rc

    # Refcount checks on the returned connectivities array
    conn = topology.connectivity(1, 2)
    rc = sys.getrefcount(conn)
    cells = conn(0)
    assert sys.getrefcount(conn) == rc + 1
    del cells
    assert sys.getrefcount(conn) == rc
