# Copyright (C) 2006 Anders Logg
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import math
import sys

import numpy as np
import pytest

import basix
import ufl
from dolfinx import cpp as _cpp
from dolfinx import graph
from dolfinx import mesh as _mesh
from dolfinx.cpp.mesh import (create_cell_partitioner, entities_to_geometry,
                              is_simplex)
from dolfinx.fem import assemble_scalar, form
from dolfinx.mesh import (CellType, DiagonalType, GhostMode, create_box,
                          create_rectangle, create_submesh, create_unit_cube,
                          create_unit_interval, create_unit_square,
                          locate_entities, locate_entities_boundary,
                          compute_boundary_facets)
from ufl import dx

from mpi4py import MPI


def submesh_topology_test(mesh, submesh, entity_map, vertex_map, entity_dim):
    submesh_cell_imap = submesh.topology.index_map(entity_dim)
    submesh_c_to_v = submesh.topology.connectivity(entity_dim, 0)

    assert(submesh_cell_imap.size_local + submesh_cell_imap.num_ghosts == submesh_c_to_v.num_nodes)

    # Check that creating facets / creating connectivity doesn't cause
    # a segmentation fault
    mesh_tdim = mesh.topology.dim
    submesh.topology.create_entities(mesh_tdim - 1)
    submesh.topology.create_connectivity(mesh_tdim - 1, 0)

    # Some processes might not own or ghost entities
    if len(entity_map) > 0:
        mesh.topology.create_connectivity(entity_dim, 0)
        mesh_e_to_v = mesh.topology.connectivity(entity_dim, 0)
        submesh.topology.create_connectivity(entity_dim, 0)
        submesh_e_to_v = submesh.topology.connectivity(entity_dim, 0)
        for submesh_entity in range(len(entity_map)):
            submesh_entity_vertices = submesh_e_to_v.links(submesh_entity)
            # The submesh is created such that entities is the map from the
            # submesh entity to the mesh entity
            mesh_entity = entity_map[submesh_entity]
            mesh_entity_vertices = mesh_e_to_v.links(mesh_entity)
            for i in range(len(submesh_entity_vertices)):
                assert vertex_map[submesh_entity_vertices[i]] == mesh_entity_vertices[i]
    else:
        assert submesh.topology.index_map(entity_dim).size_local == 0


def submesh_geometry_test(mesh, submesh, entity_map, geom_map, entity_dim):
    submesh_geom_index_map = submesh.geometry.index_map()
    assert submesh_geom_index_map.size_local + submesh_geom_index_map.num_ghosts == submesh.geometry.x.shape[0]

    # Some processes might not own or ghost entities
    if len(entity_map) > 0:
        assert mesh.geometry.dim == submesh.geometry.dim

        e_to_g = entities_to_geometry(mesh, entity_dim, entity_map, False)
        for submesh_entity in range(len(entity_map)):
            submesh_x_dofs = submesh.geometry.dofmap.links(submesh_entity)
            # e_to_g[i] gets the mesh x_dofs of entities[i], which should
            # correspond to the x_dofs of cell i in the submesh
            mesh_x_dofs = e_to_g[submesh_entity]
            for i in range(len(submesh_x_dofs)):
                assert mesh_x_dofs[i] == geom_map[submesh_x_dofs[i]]
                assert np.allclose(mesh.geometry.x[mesh_x_dofs[i]], submesh.geometry.x[submesh_x_dofs[i]])


@pytest.fixture
def mesh1d():
    """Create 1D mesh with degenerate cell"""
    mesh1d = create_unit_interval(MPI.COMM_WORLD, 4)
    i1 = np.where((mesh1d.geometry.x == (0.75, 0, 0)).all(axis=1))[0][0]
    i2 = np.where((mesh1d.geometry.x == (1, 0, 0)).all(axis=1))[0][0]
    mesh1d.geometry.x[i2] = mesh1d.geometry.x[i1]
    return mesh1d


def mesh_1d():
    """Create 1D mesh with degenerate cell"""
    mesh1d = create_unit_interval(MPI.COMM_WORLD, 4)
    i1 = np.where((mesh1d.geometry.x == (0.75, 0, 0)).all(axis=1))[0][0]
    i2 = np.where((mesh1d.geometry.x == (1, 0, 0)).all(axis=1))[0][0]
    mesh1d.geometry.x[i2] = mesh1d.geometry.x[i1]
    return mesh1d


@pytest.fixture
def mesh2d():
    """Create 2D mesh with one equilateral triangle"""
    mesh2d = create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]),
                         np.array([1., 1.])], [1, 1],
        CellType.triangle, GhostMode.none,
        create_cell_partitioner(), DiagonalType.left)
    i1 = np.where((mesh2d.geometry.x
                   == (1, 1, 0)).all(axis=1))[0][0]
    mesh2d.geometry.x[i1, :2] += 0.5 * (math.sqrt(3.0) - 1.0)
    return mesh2d


def mesh_2d():
    """Create 2D mesh with one equilateral triangle"""
    mesh2d = create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]),
                         np.array([1., 1.])], [1, 1],
        CellType.triangle, GhostMode.none,
        create_cell_partitioner(), DiagonalType.left)
    i1 = np.where((mesh2d.geometry.x
                   == (1, 1, 0)).all(axis=1))[0][0]
    mesh2d.geometry.x[i1, :2] += 0.5 * (math.sqrt(3.0) - 1.0)
    return mesh2d


@pytest.fixture
def mesh3d():
    """Create 3D mesh with regular tetrahedron and degenerate cells"""
    mesh3d = create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    i1 = np.where((mesh3d.geometry.x == (0, 1, 0)).all(axis=1))[0][0]
    i2 = np.where((mesh3d.geometry.x == (1, 1, 1)).all(axis=1))[0][0]
    mesh3d.geometry.x[i1][0] = 1.0
    mesh3d.geometry.x[i2][1] = 0.0
    return mesh3d


def mesh_3d():
    """Create 3D mesh with regular tetrahedron and degenerate cells"""
    mesh3d = create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    i1 = np.where((mesh3d.geometry.x == (0, 1, 0)).all(axis=1))[0][0]
    i2 = np.where((mesh3d.geometry.x == (1, 1, 1)).all(axis=1))[0][0]
    mesh3d.geometry.x[i1][0] = 1.0
    mesh3d.geometry.x[i2][1] = 0.0
    return mesh3d


@pytest.fixture
def c0(mesh3d):
    """Original tetrahedron from create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)"""
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
    return create_unit_interval(MPI.COMM_WORLD, 18)


@pytest.fixture
def square():
    return create_unit_square(MPI.COMM_WORLD, 5, 5)


@pytest.fixture
def rectangle():
    return create_rectangle(
        MPI.COMM_WORLD, [np.array([0.0, 0.0]),
                         np.array([2.0, 2.0])], [5, 5],
        CellType.triangle, GhostMode.none)


@pytest.fixture
def cube():
    return create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)


@pytest.fixture
def box():
    return create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]),
                                       np.array([2, 2, 2])], [2, 2, 5], CellType.tetrahedron,
                      GhostMode.none)


@pytest.fixture
def mesh():
    return create_unit_square(MPI.COMM_WORLD, 3, 3)


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


def test_create_unit_squareDistributed():
    """Create mesh of unit square."""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 7)
    assert mesh.topology.index_map(0).size_global == 48
    assert mesh.topology.index_map(2).size_global == 70
    assert mesh.geometry.dim == 2
    assert mesh.comm.allreduce(mesh.topology.index_map(0).size_local, MPI.SUM) == 48


def test_create_unit_squareLocal():
    """Create mesh of unit square."""
    mesh = create_unit_square(MPI.COMM_SELF, 5, 7)
    assert mesh.topology.index_map(0).size_global == 48
    assert mesh.topology.index_map(2).size_global == 70
    assert mesh.geometry.dim == 2


def test_create_unit_cubeDistributed():
    """Create mesh of unit cube."""
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 7, 9)
    assert mesh.topology.index_map(0).size_global == 480
    assert mesh.topology.index_map(3).size_global == 1890
    assert mesh.geometry.dim == 3
    assert mesh.comm.allreduce(mesh.topology.index_map(0).size_local, MPI.SUM) == 480


def test_create_unit_cube_local():
    """Create mesh of unit cube."""
    mesh = create_unit_cube(MPI.COMM_SELF, 5, 7, 9)
    assert mesh.topology.index_map(0).size_global == 480
    assert mesh.topology.index_map(0).size_local == 480
    assert mesh.topology.index_map(3).size_global == 1890
    assert mesh.topology.index_map(3).size_local == 1890
    assert mesh.geometry.dim == 3


def test_create_unit_square_quads():
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 7, CellType.quadrilateral)
    assert mesh.topology.index_map(0).size_global == 48
    assert mesh.topology.index_map(2).size_global == 35
    assert mesh.geometry.dim == 2
    assert mesh.comm.allreduce(mesh.topology.index_map(0).size_local, MPI.SUM) == 48


def test_create_unit_square_hex():
    mesh = create_unit_cube(MPI.COMM_WORLD, 5, 7, 9, CellType.hexahedron)
    assert mesh.topology.index_map(0).size_global == 480
    assert mesh.topology.index_map(3).size_global == 315
    assert mesh.geometry.dim == 3
    assert mesh.comm.allreduce(mesh.topology.index_map(0).size_local, MPI.SUM) == 480


def test_create_box_prism():
    mesh = create_box(MPI.COMM_WORLD, [[0., 0., 0.], [1., 1., 1.]], [2, 3, 4], CellType.prism, GhostMode.none)
    assert mesh.topology.index_map(0).size_global == 60
    assert mesh.topology.index_map(3).size_global == 48


@pytest.mark.skip_in_parallel
def test_get_coordinates():
    """Get coordinates of vertices"""
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    assert len(mesh.geometry.x) == 36


@pytest.mark.skip("Needs to be re-implemented")
@pytest.mark.skip_in_parallel
def test_cell_inradius(c0, c1, c5):
    assert _cpp.mesh.inradius(c0[0], [c0[2]]) == pytest.approx((3.0 - math.sqrt(3.0)) / 6.0)
    assert _cpp.mesh.inradius(c1[0], [c1[2]]) == pytest.approx(0.0)
    assert _cpp.mesh.inradius(c5[0], [c5[2]]) == pytest.approx(math.sqrt(3.0) / 6.0)


@pytest.mark.skip("Needs to be re-implemented")
@pytest.mark.skip_in_parallel
def test_cell_circumradius(c0, c1, c5):
    assert _cpp.mesh.circumradius(c0[0], [c0[2]], c0[1]) == pytest.approx(math.sqrt(3.0) / 2.0)
    # Implementation of diameter() does not work accurately
    # for degenerate cells - sometimes yields NaN
    r_c1 = _cpp.mesh.circumradius(c1[0], [c1[2]], c1[1])
    assert math.isnan(r_c1)
    assert _cpp.mesh.circumradius(c5[0], [c5[2]], c5[1]) == pytest.approx(math.sqrt(3.0) / 2.0)


@pytest.mark.skip_in_parallel
def test_cell_h(c0, c1, c5):
    for c in [c0, c1, c5]:
        assert _cpp.mesh.h(c[0], c[1], [c[2]]) == pytest.approx(math.sqrt(2.0))


@pytest.mark.skip("Needs to be re-implemented")
@pytest.mark.skip_in_parallel
def test_cell_radius_ratio(c0, c1, c5):
    assert _cpp.mesh.radius_ratio(c0[0], c0[2]) == pytest.approx(math.sqrt(3.0) - 1.0)
    assert np.isnan(_cpp.mesh.radius_ratio(c1[0], c1[2]))
    assert _cpp.mesh.radius_ratio(c5[0], c5[2]) == pytest.approx(1.0)


@pytest.fixture(params=['dir1_fixture', 'dir2_fixture'])
def dirname(request):
    return request.getfixturevalue(request.param)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("_mesh,hmin,hmax",
                         [
                             #  (mesh_1d, 0.0, 0.25),
                             (mesh_2d, math.sqrt(2.0), math.sqrt(2.0)),
                             (mesh_3d, math.sqrt(2.0), math.sqrt(2.0)),
                         ])
def test_hmin_hmax(_mesh, hmin, hmax):
    mesh = _mesh()
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    h = _cpp.mesh.h(mesh, tdim, range(num_cells))
    assert h.min() == pytest.approx(hmin)
    assert h.max() == pytest.approx(hmax)


# @pytest.mark.skip_in_parallel
# @pytest.mark.skip("Needs to be re-implemented")
# @pytest.mark.parametrize("mesh,rmin,rmax",
#                          [
#                              (mesh_1d(), 0.0, 0.125),
#                              (mesh_2d(), 1.0 / (2.0 + math.sqrt(2.0)), math.sqrt(6.0) / 6.0),
#                              (mesh_3d(), 0.0, math.sqrt(3.0) / 6.0),
#                          ])
# def test_rmin_rmax(mesh, rmin, rmax):
#     tdim = mesh.topology.dim
#     num_cells = mesh.topology.index_map(tdim).size_local
#     inradius = cpp.mesh.inradius(mesh, range(num_cells))
#     assert inradius.min() == pytest.approx(rmin)
#     assert inradius.max() == pytest.approx(rmax)

# - Facilities to run tests on combination of meshes


mesh_factories = [
    (create_unit_interval, (MPI.COMM_WORLD, 18)),
    (create_unit_square, (MPI.COMM_WORLD, 4, 4)),
    (create_unit_cube, (MPI.COMM_WORLD, 2, 2, 2)),
    (create_unit_square, (MPI.COMM_WORLD, 4, 4, CellType.quadrilateral)),
    (create_unit_cube, (MPI.COMM_WORLD, 2, 2, 2, CellType.hexahedron)),
    # FIXME: Add mechanism for testing meshes coming from IO
]


# FIXME: Fix this xfail
def xfail_ghosted_quads_hexes(mesh_factory, ghost_mode):
    """Xfail when mesh_factory on quads/hexes uses shared_vertex mode. Needs implementing."""
    if mesh_factory in [create_unit_square, create_unit_cube]:
        if ghost_mode == GhostMode.shared_vertex:
            pytest.xfail(reason="Missing functionality in \'{}\' with \'{}\' mode".format(mesh_factory, ghost_mode))


@pytest.mark.parametrize("ghost_mode",
                         [
                             GhostMode.none,
                             GhostMode.shared_facet,
                             GhostMode.shared_vertex,
                         ])
@pytest.mark.parametrize('mesh_factory', mesh_factories)
def xtest_mesh_topology_against_basix(mesh_factory, ghost_mode):
    """Test that mesh cells have topology matching to Basix reference
    cell they were created from.
    """
    func, args = mesh_factory
    xfail_ghosted_quads_hexes(func, ghost_mode)
    mesh = func(*args)
    if not is_simplex(mesh.topology.cell_type):
        return

    # Create basix cell
    cell_name = mesh.topology.cell_type.name
    basix_celltype = getattr(basix.CellType, cell_name)

    map = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map.size_local + map.num_ghosts
    for i in range(num_cells):
        # Get indices of cell vertices
        vertex_global_indices = mesh.topology.connectivity(mesh.topology.dim, 0).links(i)

        # Loop over all dimensions of reference cell topology
        for d, d_topology in enumerate(basix.topology(basix_celltype)):

            # Get entities of dimension d on the cell
            entities = mesh.topology.connectivity(mesh.topology.dim, d).links(i)
            if len(entities) == 0:  # Fixup for highest dimension
                entities = (i, )

            # Loop over all entities of fixed dimension d
            for entity_index, entity_topology in enumerate(d_topology):

                # Check that entity vertices map to cell vertices in correct order
                vertices = mesh.topology.connectivity(d, 0).links(entities[entity_index])
                vertices_dolfin = np.sort(vertices)
                vertices2 = np.sort(vertex_global_indices[np.array(entity_topology)])
                assert all(vertices2 == vertices_dolfin)


def xtest_mesh_topology_lifetime():
    """Check that lifetime of Mesh.topology is bound to underlying mesh object"""
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    rc = sys.getrefcount(mesh)
    topology = mesh.topology
    assert sys.getrefcount(mesh) == rc + 1
    del topology
    assert sys.getrefcount(mesh) == rc


@pytest.mark.skip_in_parallel
def test_small_mesh():
    mesh3d = create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    gdim = mesh3d.geometry.dim
    assert mesh3d.topology.index_map(gdim).size_global == 6

    mesh2d = create_unit_square(MPI.COMM_WORLD, 1, 1)
    gdim = mesh2d.geometry.dim
    assert mesh2d.topology.index_map(gdim).size_global == 2

    # mesh1d = create_unit_interval(MPI.COMM_WORLD, 2)
    # gdim = mesh1d.geometry.dim
    # assert mesh1d.topology.index_map(gdim).size_global == 2


def test_unit_hex_mesh_assemble():
    mesh = create_unit_cube(MPI.COMM_WORLD, 6, 7, 5, CellType.hexahedron)
    vol = assemble_scalar(form(1 * dx(mesh)))
    vol = mesh.comm.allreduce(vol, MPI.SUM)
    assert vol == pytest.approx(1, rel=1e-9)


def boundary_0(x):
    lr = np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    tb = np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    return np.logical_or(lr, tb)


def boundary_1(x):
    return np.logical_or(np.isclose(x[0], 1.0), np.isclose(x[1], 1.0))


def boundary_2(x):
    return np.logical_and(np.isclose(x[1], 1), x[0] >= 0.5)


# TODO Test that submesh of full mesh is a copy of the mesh
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("codim", [0, 1])
@pytest.mark.parametrize("marker", [lambda x: x[0] >= 0.5,
                                    lambda x: x[0] >= -1])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_submesh(d, n, codim, marker, ghost_mode):
    if d == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, n, n,
                                  ghost_mode=ghost_mode)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n,
                                ghost_mode=ghost_mode)

    edim = mesh.topology.dim - codim
    entities = locate_entities(mesh, edim, marker)
    submesh, entity_map, vertex_map, geom_map = create_submesh(mesh, edim, entities)
    submesh_topology_test(mesh, submesh, entity_map, vertex_map, edim)
    submesh_geometry_test(mesh, submesh, entity_map, geom_map, edim)


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("n", [2, 6])
@pytest.mark.parametrize("boundary", [boundary_0,
                                      boundary_1,
                                      boundary_2])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_submesh_boundary(d, n, boundary, ghost_mode):
    if d == 2:
        mesh = create_unit_square(MPI.COMM_WORLD, n, n,
                                  ghost_mode=ghost_mode)
    else:
        mesh = create_unit_cube(MPI.COMM_WORLD, n, n, n,
                                ghost_mode=ghost_mode)
    edim = mesh.topology.dim - 1
    entities = locate_entities_boundary(mesh, edim, boundary)
    submesh, entity_map, vertex_map, geom_map = create_submesh(mesh, edim, entities)
    submesh_topology_test(mesh, submesh, entity_map, vertex_map, edim)
    submesh_geometry_test(mesh, submesh, entity_map, geom_map, edim)


def test_empty_rank_mesh():
    """Construction of mesh where some ranks are empty"""
    comm = MPI.COMM_WORLD
    cell_type = CellType.triangle
    tdim = 2
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.Cell(cell_type.name), 1))

    def partitioner(comm, nparts, local_graph, num_ghost_nodes, ghosting):
        """Leave cells on the curent rank"""
        dest = np.full(len(cells), comm.rank, dtype=np.int32)
        return graph.create_adjacencylist(dest)

    if comm.rank == 0:
        cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        cells = graph.create_adjacencylist(cells)
        x = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    else:
        cells = graph.create_adjacencylist(np.empty((0, 3), dtype=np.int64))
        x = np.empty((0, 2), dtype=np.float64)

    mesh = _mesh.create_mesh(comm, cells, x, domain, GhostMode.none, partitioner)

    topology = mesh.topology

    # Check number of vertices
    vmap = topology.index_map(0)
    assert vmap.size_local == x.shape[0]
    assert vmap.num_ghosts == 0

    # Check number of cells
    cmap = topology.index_map(tdim)
    assert cmap.size_local == cells.num_nodes
    assert cmap.num_ghosts == 0

    # Check number of edges
    topology.create_entities(1)
    emap = topology.index_map(1)

    e_to_v = topology.connectivity(1, 0)

    assert emap.num_ghosts == 0
    if comm.rank == 0:
        assert emap.size_local == 5
        assert e_to_v.num_nodes == 5
        assert len(e_to_v.array) == 10
    else:
        assert emap.size_local == 0
        assert len(e_to_v.array) == 0
        assert e_to_v.num_nodes == 0

    # Test creating and getting permutations doesn't throw an error
    mesh.topology.create_entity_permutations()
    mesh.topology.get_cell_permutation_info()
    mesh.topology.get_facet_permutations()


def test_original_index():
    nx = 7
    mesh = create_unit_cube(MPI.COMM_WORLD, nx, nx, nx, ghost_mode=GhostMode.none)

    s = sum(mesh.topology.original_cell_index)
    s = MPI.COMM_WORLD.allreduce(s, MPI.SUM)
    assert(s == nx**3 * 6 * (nx**3 * 6 - 1) // 2)


def compute_num_boundary_facets(mesh):
    "Compute the total number of boundary facets in the mesh"

    # Create facets and facet cell connectivity
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)
    mesh.topology.create_connectivity(tdim - 1, tdim)

    # Compute number of owned facets on the boundary
    boundary_facet_marker = compute_boundary_facets(mesh.topology)
    num_owned_boundary_facets = np.sum(boundary_facet_marker)

    # Sum the number of boundary facets owned by each process to get the
    # total number in the mesh
    num_boundary_facets = mesh.comm.allreduce(
        num_owned_boundary_facets, op=MPI.SUM)

    return num_boundary_facets


@pytest.mark.parametrize("n", [2, 5])
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_boundary_facets(n, d, ghost_mode):
    "Test that the correct number of boundary facets are computed"
    if d == 2:
        mesh = create_unit_square(
            MPI.COMM_WORLD, n, n, ghost_mode=ghost_mode)
        expected_num_boundary_facets = 4 * n
    else:
        mesh = create_unit_cube(
            MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
        expected_num_boundary_facets = 6 * n**2 * 2

    num_boundary_facets = compute_num_boundary_facets(mesh)

    assert(num_boundary_facets == expected_num_boundary_facets)


@pytest.mark.parametrize("n", [2, 5])
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_submesh_codim_0_boundary_facets(n, d, ghost_mode):
    """Test that the correct number of boundary facets are computed
    for a submesh of codim 0"""
    if d == 2:
        mesh_1 = create_rectangle(
            MPI.COMM_WORLD, ((0.0, 0.0), (2.0, 1.0)), (2 * n, n),
            ghost_mode=ghost_mode)
        expected_num_boundary_facets = 4 * n
    else:
        mesh_1 = create_box(
            MPI.COMM_WORLD, ((0.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
            (2 * n, n, n), ghost_mode=ghost_mode)
        expected_num_boundary_facets = 6 * n**2 * 2

    # Create submesh of half of the rectangle / box mesh to get unit
    # square / cube mesh
    edim = mesh_1.topology.dim
    entities = locate_entities(mesh_1, edim, lambda x: x[0] <= 1.0)
    submesh = create_submesh(mesh_1, edim, entities)[0]

    assert(compute_num_boundary_facets(submesh)
           == expected_num_boundary_facets)


@pytest.mark.parametrize("n", [2, 5])
@pytest.mark.parametrize("ghost_mode", [GhostMode.none,
                                        GhostMode.shared_facet])
def test_submesh_codim_1_boundary_facets(n, ghost_mode):
    """Test that the correct number of boundary facets are computed
    for a submesh of codim 1"""
    mesh = create_unit_cube(
        MPI.COMM_WORLD, n, n, n, ghost_mode=ghost_mode)
    edim = mesh.topology.dim - 1
    entities = locate_entities_boundary(
        mesh, edim, lambda x: np.isclose(x[2], 0.0))
    submesh = create_submesh(mesh, edim, entities)[0]

    num_boundary_facets = compute_num_boundary_facets(submesh)
    expected_num_boundary_facets = 4 * n

    assert(num_boundary_facets == expected_num_boundary_facets)
