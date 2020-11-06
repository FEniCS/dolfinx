# Copyright (C) 2013-2014 Anders Logg
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for BoundingBoxTree"""

import numpy
import pytest
from dolfinx import (BoxMesh, UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh,
                     cpp, geometry)
from dolfinx.geometry import BoundingBoxTree
from dolfinx.mesh import locate_entities_boundary
from dolfinx_utils.test.skips import skip_in_parallel
from mpi4py import MPI

# --- compute_collisions with point ---


def rotation_matrix(axis, angle):
    # See https://en.wikipedia.org/wiki/Rotation_matrix,
    # Subsection: Rotation_matrix_from_axis_and_angle.
    if numpy.isclose(numpy.inner(axis, axis), 1):
        n_axis = axis
    else:
        # Normalize axis
        n_axis = axis / numpy.sqrt(numpy.inner(axis, axis))

    # Define cross product matrix of axis
    axis_x = numpy.array([[0, -n_axis[2], n_axis[1]],
                          [n_axis[2], 0, -n_axis[0]],
                          [-n_axis[1], n_axis[0], 0]])
    id = numpy.cos(angle) * numpy.eye(3)
    outer = (1 - numpy.cos(angle)) * numpy.outer(n_axis, n_axis)
    return numpy.sin(angle) * axis_x + id + outer


def test_empty_tree():
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    bbtree = cpp.geometry.BoundingBoxTree(mesh, mesh.topology.dim, [])
    assert bbtree.num_bboxes() == 0


@skip_in_parallel
def test_compute_collisions_point_1d():

    reference = {1: set([4])}

    p = numpy.array([0.3, 0, 0])
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    for dim in range(1, 2):
        tree = BoundingBoxTree(mesh, mesh.topology.dim)
        entities = geometry.compute_collisions_point(tree, p)
        assert set(entities) == reference[dim]


@skip_in_parallel
@pytest.mark.parametrize("point,cells", [(numpy.array([0.52, 0, 0]), [
    set([8, 9, 10, 11, 12, 13, 14, 15]),
    set([0, 1, 2, 3, 4, 5, 6, 7])]),
    (numpy.array([0.9, 0, 0]), [set([14, 15]), set([0, 1])])])
def test_compute_collisions_tree_1d(point, cells):
    mesh_A = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    mesh_B = UnitIntervalMesh(MPI.COMM_WORLD, 16)

    bgeom = mesh_B.geometry.x
    bgeom += point

    tree_A = BoundingBoxTree(mesh_A, mesh_A.topology.dim)
    tree_B = BoundingBoxTree(mesh_B, mesh_B.topology.dim)
    entities = geometry.compute_collisions(tree_A, tree_B)

    entities_A = set([q[0] for q in entities])
    entities_B = set([q[1] for q in entities])

    assert entities_A == cells[0]
    assert entities_B == cells[1]


@skip_in_parallel
@pytest.mark.parametrize("point,cells", [(numpy.array([0.52, 0.51, 0.0]), [
    [20, 21, 22, 23, 28, 29, 30, 31],
    [0, 1, 2, 3, 8, 9, 10, 11]]),
    (numpy.array([0.9, -0.9, 0.0]), [[6, 7], [24, 25]])])
def test_compute_collisions_tree_2d(point, cells):
    mesh_A = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
    mesh_B = UnitSquareMesh(MPI.COMM_WORLD, 4, 4)
    bgeom = mesh_B.geometry.x
    bgeom += point
    tree_A = BoundingBoxTree(mesh_A, mesh_A.topology.dim)
    tree_B = BoundingBoxTree(mesh_B, mesh_B.topology.dim)
    entities = geometry.compute_collisions(tree_A, tree_B)

    entities_A = set([q[0] for q in entities])
    entities_B = set([q[1] for q in entities])
    assert entities_A == set(cells[0])
    assert entities_B == set(cells[1])


@skip_in_parallel
def test_compute_collisions_tree_3d():

    references = [[
        set([18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47]),
        set([0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29])
    ], [
        set([6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35]),
        set([12, 13, 14, 15, 16, 17, 36, 37, 38, 39, 40, 41])
    ]]

    points = [numpy.array([0.52, 0.51, 0.3]),
              numpy.array([0.9, -0.9, 0.3])]

    for i, point in enumerate(points):

        mesh_A = UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)
        mesh_B = UnitCubeMesh(MPI.COMM_WORLD, 2, 2, 2)

        bgeom = mesh_B.geometry.x
        bgeom += point

        tree_A = BoundingBoxTree(mesh_A, mesh_A.topology.dim)
        tree_B = BoundingBoxTree(mesh_B, mesh_B.topology.dim)
        entities = geometry.compute_collisions(tree_A, tree_B)

        entities_A = set([q[0] for q in entities])
        entities_B = set([q[1] for q in entities])
        assert entities_A == references[i][0]
        assert entities_B == references[i][1]


@skip_in_parallel
def test_compute_closest_entity_1d():
    reference = (0, 1.0)
    p = numpy.array([-1.0, 0, 0])
    mesh = UnitIntervalMesh(MPI.COMM_WORLD, 16)
    tree_mid = geometry.BoundingBoxTree.create_midpoint_tree(mesh)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    entity, distance = geometry.compute_closest_entity(tree, tree_mid, mesh, p)
    assert entity == reference[0]
    assert distance[0] == pytest.approx(reference[1], 1.0e-12)


@skip_in_parallel
def test_compute_closest_entity_2d():
    reference = (1, 1.0)
    p = numpy.array([-1.0, 0.01, 0.0])
    mesh = UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    tree_mid = geometry.BoundingBoxTree.create_midpoint_tree(mesh)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    entity, distance = geometry.compute_closest_entity(tree, tree_mid, mesh, p)
    assert entity == reference[0]
    assert distance[0] == pytest.approx(reference[1], 1.0e-12)


@skip_in_parallel
def test_compute_closest_entity_3d():
    reference = (0, 0.1)
    p = numpy.array([0.1, 0.05, -0.1])
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    tree = BoundingBoxTree(mesh, mesh.topology.dim)
    tree_mid = geometry.BoundingBoxTree.create_midpoint_tree(mesh)
    entity, distance = geometry.compute_closest_entity(tree, tree_mid, mesh, p)
    assert entity == reference[0]
    assert distance[0] == pytest.approx(reference[1], 1.0e-12)


def test_surface_bbtree():
    """
    Test creation of BBTree on subset of entities (surface cells)
    """
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    sf = cpp.mesh.exterior_facet_indices(mesh)
    tdim = mesh.topology.dim
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = [f_to_c.links(f)[0] for f in sf]
    bbtree = cpp.geometry.BoundingBoxTree(mesh, tdim, cells)

    # test collision (should not collide with any)
    p = numpy.array([0.5, 0.5, 0.5])
    assert len(cpp.geometry.compute_collisions_point(bbtree, p)) == 0


def test_sub_bbtree():
    """
    Testing point collision with a BoundingBoxTree of sub entitites
    """
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 4, 4, 4, cell_type=cpp.mesh.CellType.hexahedron)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    def top_surface(x):
        return numpy.isclose(x[2], 1)

    top_facets = locate_entities_boundary(mesh, fdim, top_surface)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = [f_to_c.links(f)[0] for f in top_facets]
    bbtree = cpp.geometry.BoundingBoxTree(mesh, tdim, cells)

    # Compute a BBtree for all processes
    process_bbtree = bbtree.compute_global_tree(mesh.mpi_comm())
    # Compute collisions across processors
    point = numpy.array([0.2, 0.2, 1])
    ranks = cpp.geometry.compute_collisions_point(process_bbtree, point)

    # Compute local collisions
    cells = cpp.geometry.compute_collisions_point(bbtree, point)
    if MPI.COMM_WORLD.rank in ranks:
        assert(len(cells) > 0)
    else:
        assert(len(cells) == 0)
    print(MPI.COMM_WORLD, ranks, cells)


@pytest.mark.parametrize("ct", [cpp.mesh.CellType.hexahedron, cpp.mesh.CellType.tetrahedron])
@pytest.mark.parametrize("N", [7, 13])
def test_sub_bbtree_box(ct, N):
    """
    Test that the bounding box of the stem of the bounding box tree is what we expect
    """
    mesh = UnitCubeMesh(MPI.COMM_WORLD, N, N, N, cell_type=ct)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    axis = 1

    def marker(x):
        return numpy.isclose(x[axis], 1)
    facets = locate_entities_boundary(mesh, fdim, marker)
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    cells = numpy.unique([f_to_c.links(f)[0] for f in facets])
    bbtree = cpp.geometry.BoundingBoxTree(mesh, tdim, cells)
    num_boxes = bbtree.num_bboxes()
    if num_boxes > 0:
        bbox = bbtree.get_bbox(num_boxes - 1)
        assert(numpy.isclose(bbox[0][axis], (N - 1) / N))

    tree = cpp.geometry.BoundingBoxTree(mesh, tdim)
    all_boxes = tree.num_bboxes()
    assert(num_boxes < all_boxes)


@skip_in_parallel
def test_surface_bbtree_collision():
    """
    Compute collision between two meshes, where only one cell of each mesh are colliding
    """
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 2, 4, 4, cpp.mesh.CellType.hexahedron)

    def top_surface(x):
        return numpy.isclose(x[2], 1)
    # Tag top facets of cube
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    top_facets = locate_entities_boundary(mesh, tdim - 1, top_surface)
    top_cells = [f_to_c.links(f)[0] for f in top_facets]

    # Rotate cube
    r_matrix = rotation_matrix([1, 1, 0], numpy.pi / 4)
    mesh.geometry.x[:, :] = numpy.dot(r_matrix, mesh.geometry.x.T).T

    # Compute global bounding box of top surface
    bb = cpp.geometry.BoundingBoxTree(mesh, tdim, top_cells)
    global_bbox = bb.compute_global_tree(mesh.mpi_comm())

    mesh2 = BoxMesh(MPI.COMM_WORLD, [numpy.array([0, 0, 1.2]), numpy.array([1, 1, 2])],
                    [3, 3, 3], cpp.mesh.CellType.hexahedron)

    def bottom_surface(x):
        return numpy.isclose(x[2], 1.2)

    # Tag bottom surface
    tdim = mesh2.topology.dim
    mesh2.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = mesh2.topology.connectivity(tdim - 1, tdim)
    bottom_facets = locate_entities_boundary(mesh2, tdim - 1, bottom_surface)
    bottom_cells = [f_to_c.links(f)[0] for f in bottom_facets]

    # Compute local bounding box of bottom cells
    bb2 = cpp.geometry.BoundingBoxTree(mesh2, tdim, bottom_cells)
    global_bbox2 = bb2.compute_global_tree(mesh.mpi_comm())
    # Compute collisions between local box of mesh 2 and global box of mesh 1
    collisions = cpp.geometry.compute_collisions(global_bbox, bb2)
    # Compute collisions between local box of mesh 1 and global box of mesh 2
    rev_collisions = cpp.geometry.compute_collisions(global_bbox2, bb)

    local_collisions = []
    remote_collisions = {i: [] for i in range(MPI.COMM_WORLD.size)}
    remote_collisions.pop(MPI.COMM_WORLD.rank)
    for collision in collisions:
        if MPI.COMM_WORLD.rank == collision[0]:
            local_collisions.append(collision[1])
        else:
            remote_collisions[collision[0]].append(collision[1])
    rev_local_collisions = []
    rev_remote_collisions = {i: [] for i in range(MPI.COMM_WORLD.size)}
    rev_remote_collisions.pop(MPI.COMM_WORLD.rank)
    for collision in rev_collisions:
        if MPI.COMM_WORLD.rank == collision[0]:
            rev_local_collisions.append(collision[1])
        else:
            rev_remote_collisions[collision[0]].append(collision[1])

    def extract_geometricial_data(mesh, dim, entities):
        """
        For a set of entities in a mesh, return the coordinates of the vertices
        """
        mesh_nodes = []
        geom = mesh.geometry
        g_indices = cpp.mesh.entities_to_geometry(mesh, dim,
                                                  numpy.array(entities, dtype=numpy.int32),
                                                  False)
        for cell in g_indices:
            nodes = numpy.zeros((len(cell), 3), dtype=numpy.float64)
            for j, entity in enumerate(cell):
                nodes[j] = geom.x[entity]
            mesh_nodes.append(nodes)
        return mesh_nodes
    # Find colliding cells on local processor
    actual_collisions = []
    if len(local_collisions) > 0 and (rev_local_collisions):
        mesh_nodes = extract_geometricial_data(mesh, mesh.topology.dim, rev_local_collisions)
        mesh2_nodes = extract_geometricial_data(mesh2, mesh2.topology.dim, local_collisions)
        for entity2, simplex2 in zip(local_collisions, mesh2_nodes):
            for entity1, simplex1 in zip(rev_local_collisions, mesh_nodes):
                distance = cpp.geometry.compute_distance_gjk(simplex1, simplex2)
                if numpy.linalg.norm(distance) < 1e-15:
                    actual_collisions.append([entity1, entity2])
        assert(len(actual_collisions) == 1)

    if len(remote_collisions.keys()) > 0 and len(rev_remote_collisions.keys()) > 0:
        raise NotImplementedError("Test not implemented for parallel problems")
