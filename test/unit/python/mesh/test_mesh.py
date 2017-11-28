"Unit tests for the mesh library"

# Copyright (C) 2006 Anders Logg
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

from __future__ import print_function, division

import pytest
import numpy
import six
from six.moves import range

from dolfin import *
from dolfin_utils.test import fixture, set_parameters_fixture
from dolfin_utils.test import skip_in_parallel, xfail_in_parallel
from dolfin_utils.test import cd_tempdir
import FIAT

import os


if has_pybind11():
    CellType.Type_quadrilateral = CellType.Type.quadrilateral
    CellType.Type_hexahedron = CellType.Type.hexahedron


@fixture
def mesh1d():
    # Create 1D mesh with degenerate cell
    mesh1d = UnitIntervalMesh(4)
    mesh1d.coordinates()[4] = mesh1d.coordinates()[3]
    return mesh1d


@fixture
def mesh2d():
    # Create 2D mesh with one equilateral triangle
    mesh2d = UnitSquareMesh(1, 1, 'left')
    mesh2d.coordinates()[3] += 0.5*(sqrt(3.0)-1.0)
    return mesh2d


@fixture
def mesh3d():
    # Create 3D mesh with regular tetrahedron and degenerate cells
    mesh3d = UnitCubeMesh(1, 1, 1)
    mesh3d.coordinates()[2][0] = 1.0
    mesh3d.coordinates()[7][1] = 0.0
    return mesh3d


@fixture
def c0(mesh3d):
    # Original tetrahedron from UnitCubeMesh(1, 1, 1)
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
    return UnitIntervalMesh(10)


@fixture
def square():
    return UnitSquareMesh(5, 5)


@fixture
def rectangle():
    return RectangleMesh(Point(0, 0), Point(2, 2), 5, 5)


@fixture
def cube():
    return UnitCubeMesh(3, 3, 3)


@fixture
def box():
    return BoxMesh(Point(0, 0, 0), Point(2, 2, 2), 2, 2, 5)


@fixture
def mesh():
    return UnitSquareMesh(3, 3)


@fixture
def f(mesh):
    return MeshFunction('int', mesh, 0)


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
        assert mesh.geometry().dim() == domain.geometric_dimension()
        assert mesh.topology().dim() == domain.topological_dimension()
        assert mesh.ufl_cell() == domain.ufl_cell()
        assert mesh.id() == domain.ufl_id()

    _check_ufl_domain(interval)
    _check_ufl_domain(square)
    _check_ufl_domain(rectangle)
    _check_ufl_domain(cube)
    _check_ufl_domain(box)


def test_UnitSquareMesh():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(5, 7)
    assert mesh.num_entities_global(0) == 48
    assert mesh.num_entities_global(2) == 70


def test_UnitSquareMeshDistributed():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(mpi_comm_world(), 5, 7)
    assert mesh.num_entities_global(0) == 48
    assert mesh.num_entities_global(2) == 70
    if has_petsc4py() and not has_pybind11():
        import petsc4py
        assert isinstance(mesh.mpi_comm(), petsc4py.PETSc.Comm)
        assert mesh.mpi_comm() == mpi_comm_world()


def test_UnitSquareMeshLocal():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(mpi_comm_self(), 5, 7)
    assert mesh.num_vertices() == 48
    assert mesh.num_cells() == 70
    if has_petsc4py() and not has_pybind11():
        import petsc4py
        assert isinstance(mesh.mpi_comm(), petsc4py.PETSc.Comm)
        assert mesh.mpi_comm() == mpi_comm_self()


def test_UnitCubeMesh():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(5, 7, 9)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 1890


def test_UnitCubeMeshDistributed():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(mpi_comm_world(), 5, 7, 9)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 1890


def test_UnitCubeMeshDistributedLocal():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(mpi_comm_self(), 5, 7, 9)
    assert mesh.num_vertices() == 480
    assert mesh.num_cells() == 1890


def test_UnitQuadMesh():
    mesh = UnitSquareMesh.create(5, 7, CellType.Type_quadrilateral)
    assert mesh.num_entities_global(0) == 48
    assert mesh.num_entities_global(2) == 35


def test_UnitHexMesh():
    mesh = UnitCubeMesh.create(5, 7, 9, CellType.Type_hexahedron)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 315


def test_RefineUnitIntervalMesh():
    """Refine mesh of unit interval."""
    mesh = UnitIntervalMesh(20)
    cell_markers = CellFunction("bool", mesh)
    cell_markers[0] = (MPI.rank(mesh.mpi_comm()) == 0)
    mesh2 = refine(mesh, cell_markers)
    assert mesh2.num_entities_global(0) == 22
    assert mesh2.num_entities_global(1) == 21


def test_RefineUnitSquareMesh():
    """Refine mesh of unit square."""
    mesh = UnitSquareMesh(5, 7)
    mesh = refine(mesh)
    assert mesh.num_entities_global(0) == 165
    assert mesh.num_entities_global(2) == 280


def test_RefineUnitCubeMesh():
    """Refine mesh of unit cube."""
    mesh = UnitCubeMesh(5, 7, 9)
    mesh = refine(mesh)
    assert mesh.num_entities_global(0) == 3135
    assert mesh.num_entities_global(3) == 15120


def test_BoundaryComputation():
    """Compute boundary of mesh."""
    mesh = UnitCubeMesh(2, 2, 2)
    boundary = BoundaryMesh(mesh, "exterior")
    assert boundary.num_entities_global(0) == 26
    assert boundary.num_entities_global(2) == 48


@xfail_in_parallel
def test_BoundaryBoundary():
    """Compute boundary of boundary."""
    mesh = UnitCubeMesh(2, 2, 2)
    b0 = BoundaryMesh(mesh, "exterior")
    b1 = BoundaryMesh(b0, "exterior")
    assert b1.num_vertices() == 0
    assert b1.num_cells() == 0


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
    file = File("saved_mesh_function.xml")
    file << f


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
    h1 = UnitSquareMesh(4, 4).hash()
    h2 = UnitSquareMesh(4, 5).hash()
    h3 = UnitSquareMesh(4, 4).hash()
    assert h1 == h3
    assert h1 != h2


@skip_in_parallel
def test_SubsetIterators(mesh):
    def inside1(x):
        return x[0] <= 0.5

    def inside2(x):
        return x[0] >= 0.5
    sd1 = AutoSubDomain(inside1)
    sd2 = AutoSubDomain(inside2)
    cf = CellFunction('size_t', mesh)
    cf.set_all(0)
    sd1.mark(cf, 1)
    sd2.mark(cf, 2)

    for i in range(3):
        num = 0
        for e in SubsetIterator(cf, i):
            num += 1
        assert num == 6


# FIXME: Mesh IO tests should be in io test directory
@skip_in_parallel
def test_MeshXML2D(cd_tempdir):
    """Write and read 2D mesh to/from file"""
    mesh_out = UnitSquareMesh(3, 3)
    mesh_in = Mesh()
    file = File("unitsquare.xml")
    file << mesh_out
    file >> mesh_in
    assert mesh_in.num_vertices() == 16


@skip_in_parallel
def test_MeshXML3D(cd_tempdir):
    """Write and read 3D mesh to/from file"""
    mesh_out = UnitCubeMesh(3, 3, 3)
    mesh_in = Mesh()
    file = File("unitcube.xml")
    file << mesh_out
    file >> mesh_in
    assert mesh_in.num_vertices() == 64


@skip_in_parallel
def xtest_MeshFunction(cd_tempdir):
    """Write and read mesh function to/from file"""
    mesh = UnitSquareMesh(1, 1)
    f = MeshFunction('int', mesh, 0)
    f[0] = 2
    f[1] = 4
    f[2] = 6
    f[3] = 8
    file = File("meshfunction.xml")
    file << f
    g = MeshFunction('int', mesh, 0)
    file >> g
    for v in vertices(mesh):
        assert f[v] == g[v]


def test_GetGeometricalDimension():
    """Get geometrical dimension of mesh"""
    mesh = UnitSquareMesh(5, 5)
    assert mesh.geometry().dim() == 2


@skip_in_parallel
def test_GetCoordinates():
    """Get coordinates of vertices"""
    mesh = UnitSquareMesh(5, 5)
    assert len(mesh.coordinates()) == 36


def test_GetCells():
    """Get cells of mesh"""
    mesh = UnitSquareMesh(5, 5)
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


def test_basic_cell_orientations():
    "Test that default cell orientations initialize and update as expected."
    mesh = UnitIntervalMesh(12)
    orientations = mesh.cell_orientations()
    print(len(orientations))
    assert len(orientations) == 0

    mesh.init_cell_orientations(Expression(("0.0", "1.0", "0.0"), degree=0))
    orientations = mesh.cell_orientations()
    assert len(orientations) == mesh.num_cells()
    for i in range(mesh.num_cells()):
        assert mesh.cell_orientations()[i] == 0


@skip_in_parallel
def test_cell_orientations():
    "Test that cell orientations update as expected."
    mesh = UnitIntervalMesh(12)
    mesh.init_cell_orientations(Expression(("0.0", "1.0", "0.0"), degree=0))
    for i in range(mesh.num_cells()):
        assert mesh.cell_orientations()[i] == 0

    mesh = UnitSquareMesh(2, 2)
    mesh.init_cell_orientations(Expression(("0.0", "0.0", "1.0"), degree=0))
    reference = numpy.array((0, 1, 0, 1, 0, 1, 0, 1))
    # Only compare against reference in serial (don't know how to
    # compare in parallel)
    for i in range(mesh.num_cells()):
        assert mesh.cell_orientations()[i] == reference[i]

    mesh = BoundaryMesh(UnitSquareMesh(2, 2), "exterior")
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]"), degree=1))
    print(mesh.cell_orientations())


# - Facilities to run tests on combination of meshes

ghost_mode = set_parameters_fixture("ghost_mode", [
    "none",
    "shared_facet",
    "shared_vertex",
])

mesh_factories = [
    (UnitIntervalMesh, (8,)),
    (UnitSquareMesh, (4, 4)),
    (UnitDiscMesh.create, (mpi_comm_world(), 10, 1, 2)),
    (UnitDiscMesh.create, (mpi_comm_world(), 10, 2, 2)),
    (UnitDiscMesh.create, (mpi_comm_world(), 10, 1, 3)),
    (UnitDiscMesh.create, (mpi_comm_world(), 10, 2, 3)),
    (SphericalShellMesh.create, (mpi_comm_world(), 1,)),
    (SphericalShellMesh.create, (mpi_comm_world(), 2,)),
    (UnitCubeMesh, (2, 2, 2)),
    (UnitSquareMesh.create, (4, 4, CellType.Type_quadrilateral)),
    (UnitCubeMesh.create, (2, 2, 2, CellType.Type_hexahedron)),
    # FIXME: Add mechanism for testing meshes coming from IO
]

mesh_factories_broken_shared_entities = [
    (UnitIntervalMesh, (8,)),
    (UnitSquareMesh, (4, 4)),
    # FIXME: Problem in test_shared_entities
    xfail_in_parallel((UnitDiscMesh.create, (mpi_comm_world(), 10, 1, 2))),
    xfail_in_parallel((UnitDiscMesh.create, (mpi_comm_world(), 10, 2, 2))),
    xfail_in_parallel((UnitDiscMesh.create, (mpi_comm_world(), 10, 1, 3))),
    xfail_in_parallel((UnitDiscMesh.create, (mpi_comm_world(), 10, 2, 3))),
    xfail_in_parallel((SphericalShellMesh.create, (mpi_comm_world(), 1,))),
    xfail_in_parallel((SphericalShellMesh.create, (mpi_comm_world(), 2,))),
    (UnitCubeMesh, (2, 2, 2)),
    (UnitSquareMesh.create, (4, 4, CellType.Type_quadrilateral)),
    (UnitCubeMesh.create, (2, 2, 2, CellType.Type_hexahedron)),
]

# FIXME: Fix this xfail
def xfail_ghosted_quads_hexes(mesh_factory, ghost_mode):
    """Xfail when mesh_factory on quads/hexes uses
    shared_vertex mode. Needs implementing.
    """
    if mesh_factory in [UnitSquareMesh.create, UnitCubeMesh.create]:
        if ghost_mode == 'shared_vertex':
            pytest.xfail(reason="Missing functionality in '{}' with '' "
                         "mode".format(mesh_factory, ghost_mode))


@pytest.mark.parametrize('mesh_factory', mesh_factories_broken_shared_entities)
def test_shared_entities(mesh_factory, ghost_mode):
    func, args = mesh_factory
    xfail_ghosted_quads_hexes(func, ghost_mode)
    mesh = func(*args)
    dim = mesh.topology().dim()

    # FIXME: Implement a proper test
    for shared_dim in range(dim + 1):
        # Initialise global indices (if not already)
        mesh.init_global(shared_dim)

        assert isinstance(mesh.topology().shared_entities(shared_dim), dict)
        assert isinstance(mesh.topology().global_indices(shared_dim),
                          numpy.ndarray)

        if mesh.topology().have_shared_entities(shared_dim):
            for e in entities(mesh, shared_dim):
                sharing = e.sharing_processes()
                if not has_pybind11():
                    assert isinstance(sharing, numpy.ndarray)
                    assert (sharing.size > 0) == e.is_shared()
                else:
                    assert isinstance(sharing, set)
                    assert (len(sharing) > 0) == e.is_shared()

        n_entities = mesh.num_entities(shared_dim)
        n_global_entities = mesh.num_entities_global(shared_dim)
        shared_entities = mesh.topology().shared_entities(shared_dim)

        # Check that sum(local-shared) = global count
        rank = MPI.rank(mesh.mpi_comm())
        ct = sum(1 for val in six.itervalues(shared_entities) if list(val)[0] < rank)
        num_entities_global = MPI.sum(mesh.mpi_comm(), mesh.num_entities(shared_dim) - ct)

        assert num_entities_global ==  mesh.num_entities_global(shared_dim)


@pytest.mark.parametrize('mesh_factory', mesh_factories)
def test_mesh_topology_against_fiat(mesh_factory, ghost_mode):
    """Test that mesh cells have topology matching to FIAT reference
    cell they were created from.
    """
    func, args = mesh_factory
    xfail_ghosted_quads_hexes(func, ghost_mode)
    mesh = func(*args)
    assert mesh.ordered()
    tdim = mesh.topology().dim()

    # Create FIAT cell
    cell_name = CellType.type2string(mesh.type().cell_type())
    fiat_cell = FIAT.ufc_cell(cell_name)

    # Initialize all mesh entities and connectivities
    mesh.init()

    for cell in cells(mesh):
        # Get mesh-global (MPI-local) indices of cell vertices
        vertex_global_indices = cell.entities(0)

        # Loop over all dimensions of reference cell topology
        for d, d_topology in six.iteritems(fiat_cell.get_topology()):

            # Get entities of dimension d on the cell
            entities = cell.entities(d)
            if len(entities) == 0:  # Fixup for highest dimension
                entities = (cell.index(),)

            # Loop over all entities of fixed dimension d
            for entity_index, entity_topology in six.iteritems(d_topology):

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
    tdim = mesh.topology().dim()

    # Loop over pair of dimensions d, d1 with d>d1
    for d in range(tdim+1):
        for d1 in range(d):

            # NOTE: DOLFIN UFC noncompliance!
            # DOLFIN has increasing indices only for d-0 incidence
            # with any d; UFC convention for d-d1 with d>d1 is not
            # respected in DOLFIN
            if d1 != 0:
                continue

            # Initialize d-d1 connectivity and d1 global indices
            mesh.init(d, d1)
            mesh.init_global(d1)
            assert mesh.topology().have_global_indices(d1)

            # Loop over entities of dimension d
            for e in entities(mesh, d):

                # Get global indices
                subentities_indices = [e1.global_index() for e1 in entities(e, d1)]
                assert subentities_indices.count(-1) == 0

                # Check that d1-subentities of d-entity have increasing indices
                assert sorted(subentities_indices) == subentities_indices
