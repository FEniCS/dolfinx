# Copyright (C) 2006 Anders Logg
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import math
import os
import sys

import numpy as np
import pytest

import dolfinx
import FIAT
from dolfinx import (MPI, BoxMesh, Mesh, MeshEntity, MeshFunction,
                     RectangleMesh, UnitCubeMesh, UnitIntervalMesh,
                     UnitSquareMesh, cpp, has_kahip)
from dolfinx.cpp.mesh import CellType, Partitioner, is_simplex
from dolfinx.fem import assemble_scalar
from dolfinx.io import XDMFFile
from dolfinx_utils.test.fixtures import tempdir
from dolfinx_utils.test.skips import skip_in_parallel
from ufl import dx

assert (tempdir)


@pytest.fixture
def mesh1d():
    """Create 1D mesh with degenerate cell"""
    mesh1d = UnitIntervalMesh(MPI.comm_world, 4)
    mesh1d.geometry.points[4] = mesh1d.geometry.points[3]
    return mesh1d


@pytest.fixture
def mesh2d():
    """Create 2D mesh with one equilateral triangle"""
    mesh2d = RectangleMesh(
        MPI.comm_world, [np.array([0.0, 0.0, 0.0]),
                         np.array([1., 1., 0.0])], [1, 1],
        CellType.triangle, cpp.mesh.GhostMode.none, 'left')
    mesh2d.geometry.points[3, :2] += 0.5 * (math.sqrt(3.0) - 1.0)
    return mesh2d


@pytest.fixture
def mesh3d():
    """Create 3D mesh with regular tetrahedron and degenerate cells"""
    mesh3d = UnitCubeMesh(MPI.comm_world, 1, 1, 1)
    i1 = np.where((mesh3d.geometry.points
                   == (0, 1, 0)).all(axis=1))[0][0]
    i2 = np.where((mesh3d.geometry.points
                   == (1, 1, 1)).all(axis=1))[0][0]

    mesh3d.geometry.points[i1][0] = 1.0
    mesh3d.geometry.points[i2][1] = 0.0
    return mesh3d


@pytest.fixture
def c0(mesh3d):
    """Original tetrahedron from UnitCubeMesh(MPI.comm_world, 1, 1, 1)"""
    return MeshEntity(mesh3d, mesh3d.topology.dim, 0)


@pytest.fixture
def c1(mesh3d):
    # Degenerate cell
    return MeshEntity(mesh3d, mesh3d.topology.dim, 1)


@pytest.fixture
def c5(mesh3d):
    # Regular tetrahedron with edge sqrt(2)
    return MeshEntity(mesh3d, mesh3d.topology.dim, 5)


@pytest.fixture
def interval():
    return UnitIntervalMesh(MPI.comm_world, 10)


@pytest.fixture
def square():
    return UnitSquareMesh(MPI.comm_world, 5, 5)


@pytest.fixture
def rectangle():
    return RectangleMesh(
        MPI.comm_world, [np.array([0.0, 0.0, 0.0]),
                         np.array([2.0, 2.0, 0.0])], [5, 5],
        CellType.triangle, cpp.mesh.GhostMode.none)


@pytest.fixture
def cube():
    return UnitCubeMesh(MPI.comm_world, 3, 3, 3)


@pytest.fixture
def box():
    return BoxMesh(MPI.comm_world, [np.array([0, 0, 0]),
                                    np.array([2, 2, 2])], [2, 2, 5], CellType.tetrahedron,
                   cpp.mesh.GhostMode.none)


@pytest.fixture
def mesh():
    return UnitSquareMesh(MPI.comm_world, 3, 3)


@pytest.fixture
def f(mesh):
    return MeshFunction('int', mesh, 0, 0)


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

    pygmsh = pytest.importorskip("pygmsh")

    if MPI.rank(MPI.comm_world) == 0:
        geom = pygmsh.opencascade.Geometry()
        geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
        pygmsh_mesh = pygmsh.generate_mesh(geom)
        points, cells = pygmsh_mesh.points, pygmsh_mesh.cells
    else:
        points = np.zeros([0, 3])
        cells = {
            "tetra": np.zeros([0, 4], dtype=np.int64),
            "triangle": np.zeros([0, 3], dtype=np.int64),
            "line": np.zeros([0, 2], dtype=np.int64)
        }

    mesh = Mesh(MPI.comm_world, dolfinx.cpp.mesh.CellType.tetrahedron, points,
                cells['tetra'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 3

    mesh = Mesh(MPI.comm_world,
                dolfinx.cpp.mesh.CellType.triangle, points,
                cells['triangle'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 2

    mesh = Mesh(MPI.comm_world,
                dolfinx.cpp.mesh.CellType.interval, points,
                cells['line'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 1
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 1

    if MPI.rank(MPI.comm_world) == 0:
        print("Generate mesh")
        geom = pygmsh.opencascade.Geometry()
        geom.add_ball([0.0, 0.0, 0.0], 1.0, char_length=0.2)
        pygmsh_mesh = pygmsh.generate_mesh(
            geom, extra_gmsh_arguments=['-order', '2'])
        points, cells = pygmsh_mesh.points, pygmsh_mesh.cells
        print("End Generate mesh", cells.keys())
    else:
        points = np.zeros([0, 3])
        cells = {
            "tetra10": np.zeros([0, 10], dtype=np.int64),
            "triangle6": np.zeros([0, 6], dtype=np.int64),
            "line3": np.zeros([0, 3], dtype=np.int64)
        }

    mesh = Mesh(MPI.comm_world, dolfinx.cpp.mesh.CellType.tetrahedron, points,
                cells['tetra10'], [], cpp.mesh.GhostMode.none)
    assert mesh.degree() == 2
    assert mesh.geometry.dim == 3
    assert mesh.topology.dim == 3

    mesh = Mesh(MPI.comm_world, dolfinx.cpp.mesh.CellType.triangle, points,
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
    assert MPI.sum(mesh.mpi_comm(), mesh.topology.index_map(0).size_local) == 48


def test_UnitSquareMeshLocal():
    """Create mesh of unit square."""
    mesh = UnitSquareMesh(MPI.comm_self, 5, 7)
    assert mesh.num_entities(0) == 48
    assert mesh.num_cells() == 70
    assert mesh.geometry.dim == 2


def test_UnitCubeMeshDistributed():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 1890
    assert mesh.geometry.dim == 3
    assert MPI.sum(mesh.mpi_comm(), mesh.topology.index_map(0).size_local) == 480


def test_UnitCubeMeshLocal():
    """Create mesh of unit cube."""
    mesh = UnitCubeMesh(MPI.comm_self, 5, 7, 9)
    assert mesh.num_entities(0) == 480
    assert mesh.num_cells() == 1890
    assert mesh.geometry.dim == 3


def test_UnitQuadMesh():
    mesh = UnitSquareMesh(MPI.comm_world, 5, 7, CellType.quadrilateral)
    assert mesh.num_entities_global(0) == 48
    assert mesh.num_entities_global(2) == 35
    assert mesh.geometry.dim == 2
    assert MPI.sum(mesh.mpi_comm(), mesh.topology.index_map(0).size_local) == 48


def test_UnitHexMesh():
    mesh = UnitCubeMesh(MPI.comm_world, 5, 7, 9, CellType.hexahedron)
    assert mesh.num_entities_global(0) == 480
    assert mesh.num_entities_global(3) == 315
    assert mesh.geometry.dim == 3
    assert MPI.sum(mesh.mpi_comm(), mesh.topology.index_map(0).size_local) == 480


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
    assert cpp.mesh.inradius(c0.mesh(), [c0.index()]) == pytest.approx((3.0 - math.sqrt(3.0)) / 6.0)
    assert cpp.mesh.inradius(c1.mesh(), [c1.index()]) == pytest.approx(0.0)
    assert cpp.mesh.inradius(c5.mesh(), [c5.index()]) == pytest.approx(math.sqrt(3.0) / 6.0)


@skip_in_parallel
def test_cell_circumradius(c0, c1, c5):
    assert cpp.mesh.circumradius(c0.mesh(), [c0.index()], c0.dim) == pytest.approx(math.sqrt(3.0) / 2.0)
    # Implementation of diameter() does not work accurately
    # for degenerate cells - sometimes yields NaN
    r_c1 = cpp.mesh.circumradius(c1.mesh(), [c1.index()], c1.dim)
    assert math.isnan(r_c1)
    assert cpp.mesh.circumradius(c5.mesh(), [c5.index()], c5.dim) == pytest.approx(math.sqrt(3.0) / 2.0)


@skip_in_parallel
def test_cell_h(c0, c1, c5):
    for c in [c0, c1, c5]:
        assert cpp.mesh.h(c.mesh(), [c.index()], c.dim) == pytest.approx(math.sqrt(2.0))


@skip_in_parallel
def test_cell_radius_ratio(c0, c1, c5):
    assert cpp.mesh.radius_ratio(c0.mesh(), [c0.index()]) == pytest.approx(math.sqrt(3.0) - 1.0)
    assert np.isnan(cpp.mesh.radius_ratio(c1.mesh(), [c1.index()]))
    assert cpp.mesh.radius_ratio(c5.mesh(), [c5.index()]) == pytest.approx(1.0)


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
    (UnitIntervalMesh, (MPI.comm_world, 8)),
    (UnitSquareMesh, (MPI.comm_world, 4, 4)),
    (UnitCubeMesh, (MPI.comm_world, 2, 2, 2)),
    (UnitSquareMesh, (MPI.comm_world, 4, 4, CellType.quadrilateral)),
    (UnitCubeMesh, (MPI.comm_world, 2, 2, 2, CellType.hexahedron)),
    # FIXME: Add mechanism for testing meshes coming from IO
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
    mesh.create_connectivity_all()

    for i in range(mesh.num_cells()):
        cell = MeshEntity(mesh, mesh.topology.dim, i)
        # Get mesh-global (MPI-local) indices of cell vertices
        vertex_global_indices = cell.entities(0)

        # Loop over all dimensions of reference cell topology
        for d, d_topology in fiat_cell.get_topology().items():

            # Get entities of dimension d on the cell
            entities = cell.entities(d)
            if len(entities) == 0:  # Fixup for highest dimension
                entities = (i, )

            # Loop over all entities of fixed dimension d
            for entity_index, entity_topology in d_topology.items():

                # Check that entity vertices map to cell vertices in correct order
                entity = MeshEntity(mesh, d, entities[entity_index])
                vertices_dolfin = np.sort(entity.entities(0))
                vertices_fiat = np.sort(vertex_global_indices[np.array(entity_topology)])
                assert all(vertices_fiat == vertices_dolfin)


def test_mesh_topology_lifetime():
    """Check that lifetime of Mesh.topology is bound to underlying mesh object"""
    mesh = UnitSquareMesh(MPI.comm_world, 4, 4)
    rc = sys.getrefcount(mesh)
    topology = mesh.topology
    assert sys.getrefcount(mesh) == rc + 1
    del topology
    assert sys.getrefcount(mesh) == rc


@pytest.mark.xfail(condition=MPI.size(MPI.comm_world) > 1,
                   reason="Small meshes fail in parallel")
def test_small_mesh():
    mesh3d = UnitCubeMesh(MPI.comm_world, 1, 1, 1)
    gdim = mesh3d.geometry.dim
    assert mesh3d.num_entities_global(gdim) == 6

    mesh2d = UnitSquareMesh(MPI.comm_world, 1, 1)
    gdim = mesh2d.geometry.dim
    assert mesh2d.num_entities_global(gdim) == 2

    mesh1d = UnitIntervalMesh(MPI.comm_world, 2)
    gdim = mesh1d.geometry.dim
    assert mesh1d.num_entities_global(gdim) == 2


def test_topology_surface(cube):
    tdim = cube.topology.dim
    cube.create_connectivity(tdim - 1, tdim)

    surface_vertex_markers = cube.topology.on_boundary(0)
    assert surface_vertex_markers

    cube.create_entities(1)
    cube.create_connectivity(2, 1)
    surface_edge_markers = cube.topology.on_boundary(1)
    assert surface_edge_markers

    surface_facet_markers = cube.topology.on_boundary(2)
    sf_count = np.count_nonzero(np.array(surface_facet_markers))
    n = 3
    assert MPI.sum(cube.mpi_comm(), sf_count) == n * n * 12


@pytest.mark.parametrize("mesh_factory", mesh_factories)
@pytest.mark.parametrize("subset_comm", [MPI.comm_world, new_comm(MPI.comm_world)])
@pytest.mark.parametrize(
    "graph_partitioner",
    [Partitioner.scotch,
     pytest.param(Partitioner.kahip,
                  marks=pytest.mark.skipif(not has_kahip, reason="KaHIP is not available"))])
def test_distribute_mesh(subset_comm, tempdir, mesh_factory, graph_partitioner):
    func, args = mesh_factory
    mesh = func(*args)

    if not is_simplex(mesh.topology.cell_type):
        return

    encoding = XDMFFile.Encoding.HDF5
    ghost_mode = cpp.mesh.GhostMode.none
    filename = os.path.join(tempdir, "mesh.xdmf")
    comm = mesh.mpi_comm()
    parts = comm.size

    with XDMFFile(mesh.mpi_comm(), filename, encoding) as file:
        file.write(mesh)

    # Use the subset_comm to read and partition mesh, then distribute to all
    # available processes
    with XDMFFile(subset_comm, filename) as file:
        cell_type, points, cells, indices = file.read_mesh_data(subset_comm)

    partition_data = cpp.mesh.partition_cells(subset_comm, parts, cell_type,
                                              cells, graph_partitioner)

    dist_mesh = cpp.mesh.build_from_partition(MPI.comm_world, cell_type, points,
                                              cells, indices, ghost_mode,
                                              partition_data)

    assert(mesh.topology.cell_type == dist_mesh.topology.cell_type)
    assert mesh.num_entities_global(0) == dist_mesh.num_entities_global(0)
    dim = dist_mesh.topology.dim
    assert mesh.num_entities_global(dim) == dist_mesh.num_entities_global(dim)


@pytest.mark.parametrize("mesh_factory", mesh_factories)
def test_custom_partition(tempdir, mesh_factory):
    func, args = mesh_factory
    mesh = func(*args)

    if not is_simplex(mesh.topology.cell_type):
        return

    comm = mesh.mpi_comm()
    filename = os.path.join(tempdir, "mesh.xdmf")
    encoding = XDMFFile.Encoding.HDF5
    ghost_mode = cpp.mesh.GhostMode.none

    with XDMFFile(comm, filename, encoding) as file:
        file.write(mesh)

    with XDMFFile(comm, filename) as file:
        cell_type, points, cells, global_indices = file.read_mesh_data(comm)

    # Create a custom partition array
    part = np.zeros(cells.shape[0], dtype=np.int32)
    part[:] = comm.rank

    ghost_procs = cpp.mesh.ghost_cell_mapping(comm, part, cell_type, cells)
    cell_partition = cpp.mesh.PartitionData(part, ghost_procs)
    dist_mesh = cpp.mesh.build_from_partition(comm, cell_type, points,
                                              cells, global_indices,
                                              ghost_mode, cell_partition)

    assert(mesh.topology.cell_type == dist_mesh.topology.cell_type)
    assert mesh.num_entities_global(0) == dist_mesh.num_entities_global(0)
    dim = dist_mesh.topology.dim
    assert mesh.num_entities_global(dim) == dist_mesh.num_entities_global(dim)


def test_coords():
    mesh = UnitCubeMesh(MPI.comm_world, 4, 4, 5)
    d = mesh.coordinate_dofs().entity_points()
    d += 2
    assert np.array_equal(d, mesh.coordinate_dofs().entity_points())


def test_UnitHexMesh_assemble():
    mesh = UnitCubeMesh(MPI.comm_world, 6, 7, 5, CellType.hexahedron)
    vol = assemble_scalar(1 * dx(mesh))
    vol = MPI.sum(mesh.mpi_comm(), vol)
    assert(vol == pytest.approx(1, rel=1e-9))


def test_mesh_order_unchanged_triangle():
    points = [[0, 0], [1, 0], [1, 1]]
    cells = [[0, 1, 2]]
    mesh = Mesh(MPI.comm_world, CellType.triangle, points,
                cells, [], cpp.mesh.GhostMode.none)
    assert (mesh.cells()[0] == cells[0]).all()


def test_mesh_order_unchanged_quadrilateral():
    points = [[0, 0], [1, 0], [0, 1], [1, 1]]
    cells = [[0, 1, 2, 3]]
    mesh = Mesh(MPI.comm_world, CellType.quadrilateral, points,
                cells, [], cpp.mesh.GhostMode.none)
    assert (mesh.cells()[0] == cells[0]).all()


def test_mesh_order_unchanged_tetrahedron():
    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1]]
    cells = [[0, 1, 2, 3]]
    mesh = Mesh(MPI.comm_world, CellType.tetrahedron, points,
                cells, [], cpp.mesh.GhostMode.none)
    assert (mesh.cells()[0] == cells[0]).all()


def test_mesh_order_unchanged_hexahedron():
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
              [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    cells = [[0, 1, 2, 3, 4, 5, 6, 7]]
    mesh = Mesh(MPI.comm_world, CellType.hexahedron, points,
                cells, [], cpp.mesh.GhostMode.none)
    assert (mesh.cells()[0] == cells[0]).all()
