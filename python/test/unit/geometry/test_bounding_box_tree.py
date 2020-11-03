# Copyright (C) 2013-2014 Anders Logg
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for BoundingBoxTree"""

import numpy
import pytest
from dolfinx import UnitCubeMesh, UnitIntervalMesh, UnitSquareMesh, geometry, cpp
from dolfinx.mesh import locate_entities_boundary
from dolfinx.geometry import BoundingBoxTree
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
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 8, 8, 8)
    sf = cpp.mesh.exterior_facet_indices(mesh)
    tdim = mesh.topology.dim
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = [f_to_c.links(f)[0] for f in sf]
    bbtree = cpp.geometry.BoundingBoxTree(mesh, tdim, cells)
    print(bbtree.num_bboxes())

    # test collision (should not collide with any)
    p = numpy.array([0.5, 0.5, 0.5])
    assert len(cpp.geometry.compute_collisions_point(bbtree, p)) == 0


def test_sub_bbtree():
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 4, 4, 4)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    def top_surface(x):
        return numpy.logical_and(numpy.isclose(x[2], 1), x[0] < 0.5)

    top_facets = locate_entities_boundary(mesh, fdim, top_surface)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = [f_to_c.links(f)[0] for f in top_facets]
    bbtree = cpp.geometry.BoundingBoxTree(mesh, tdim, cells)
    process_bbtree = bbtree.compute_global_tree(mesh.mpi_comm())
    ranks = cpp.geometry.compute_process_collisions(process_bbtree, numpy.array([[0.25, 0.25, 1]]).T)
    print("RANKS", ranks)
# @skip_in_parallel


def test_surface_bbtree2():
    # Rotated unit cube
    mesh = UnitCubeMesh(MPI.COMM_WORLD, 2, 4, 4, cpp.mesh.CellType.hexahedron)

    def top_surface(x):
        return numpy.isclose(x[2], 1)
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    top_facets = locate_entities_boundary(mesh, tdim - 1, top_surface)
    top_cells = [f_to_c.links(f)[0] for f in top_facets]

    r_matrix = rotation_matrix([1, 0, 0], numpy.pi / 6)
    mesh.geometry.x[:, :] = numpy.dot(r_matrix, mesh.geometry.x.T).T
    bb = cpp.geometry.BoundingBoxTree(mesh, tdim, top_cells)

    # Translated unit cube
    mesh2 = UnitCubeMesh(MPI.COMM_WORLD, 4, 2, 4, cpp.mesh.CellType.hexahedron)
    mesh2.name = "MESH2"
    mesh2.geometry.x[:, 2] += 1.25

    def bottom_surface(x):
        return numpy.isclose(x[2], 1.25)
    tdim = mesh2.topology.dim
    mesh2.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = mesh2.topology.connectivity(tdim - 1, tdim)
    bottom_facets = locate_entities_boundary(mesh2, tdim - 1, bottom_surface)
    bottom_cells = [f_to_c.links(f)[0] for f in bottom_facets]

    bb2 = cpp.geometry.BoundingBoxTree(mesh2, tdim, bottom_cells)

    # Compute collisions between sub bounding boxes
    collisions = cpp.geometry.compute_collisions(bb, bb2)

    # Compute collisions for the whole mesh
    full_bb = cpp.geometry.BoundingBoxTree(mesh, tdim)
    full_bb2 = cpp.geometry.BoundingBoxTree(mesh2, tdim)
    collisions_full = cpp.geometry.compute_collisions(full_bb, full_bb2)
    print(collisions)
    print(collisions_full)
    # import dolfinx.io
    # mt = MeshTags(mesh, tdim, numpy.array(top_cells, dtype=numpy.int32), numpy.array(top_cells, dtype=numpy.int32))
    # mt2 = MeshTags(mesh2, tdim, numpy.array(bottom_cells, dtype=numpy.int32),
    #                numpy.array(bottom_cells, dtype=numpy.int32))
    # mt2.name = "mt2"
    # mesh2.name = "mesh2"
    # xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "test.xdmf", "w")
    # xdmf.write_mesh(mesh)
    # xdmf.write_meshtags(mt, geometry_xpath="/Xdmf/Domain/Grid[@Name='mesh']/Geometry")
    # xdmf.write_mesh(mesh2)
    # xdmf.write_meshtags(mt2, geometry_xpath="/Xdmf/Domain/Grid[@Name='mesh2']/Geometry")
    # xdmf.close()
