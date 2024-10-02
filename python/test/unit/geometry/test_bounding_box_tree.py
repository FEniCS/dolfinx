# Copyright (C) 2013-2021 Anders Logg, Jørgen S. Dokken, Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for BoundingBoxTree"""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import cpp as _cpp
from dolfinx.geometry import (
    PointOwnershipData,
    bb_tree,
    compute_closest_entity,
    compute_colliding_cells,
    compute_collisions_points,
    compute_collisions_trees,
    compute_distance_gjk,
    create_midpoint_tree,
    determine_point_ownership,
)
from dolfinx.mesh import (
    CellType,
    compute_midpoints,
    create_box,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
    exterior_facet_indices,
    locate_entities,
    locate_entities_boundary,
)


def extract_geometricial_data(mesh, dim, entities):
    """For a set of entities in a mesh, return the coordinates of the
    vertices"""
    mesh_nodes = []
    geom = mesh.geometry
    g_indices = _cpp.mesh.entities_to_geometry(
        mesh._cpp_object, dim, np.array(entities, dtype=np.int32), False
    )
    for cell in g_indices:
        nodes = np.zeros((len(cell), 3), dtype=np.float64)
        for j, entity in enumerate(cell):
            nodes[j] = geom.x[entity]
        mesh_nodes.append(nodes)
    return mesh_nodes


def expand_bbox(bbox, dtype):
    """Expand min max bbox to convex hull"""
    return np.array(
        [
            [bbox[0][0], bbox[0][1], bbox[0][2]],
            [bbox[0][0], bbox[0][1], bbox[1][2]],
            [bbox[0][0], bbox[1][1], bbox[0][2]],
            [bbox[1][0], bbox[0][1], bbox[0][2]],
            [bbox[1][0], bbox[0][1], bbox[1][2]],
            [bbox[1][0], bbox[1][1], bbox[0][2]],
            [bbox[0][0], bbox[1][1], bbox[1][2]],
            [bbox[1][0], bbox[1][1], bbox[1][2]],
        ],
        dtype=dtype,
    )


def find_colliding_cells(mesh, bbox, dtype):
    """Given a mesh and a bounding box((xmin, ymin, zmin), (xmax, ymax,
    zmax)) find all colliding cells"""

    # Find actual cells using known bounding box tree
    colliding_cells = []
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    x_indices = _cpp.mesh.entities_to_geometry(
        mesh._cpp_object, mesh.topology.dim, np.arange(num_cells, dtype=np.int32), False
    )
    points = mesh.geometry.x
    bounding_box = expand_bbox(bbox, dtype)
    for cell in range(num_cells):
        vertex_coords = points[x_indices[cell]]
        bbox_cell = np.array([vertex_coords[0], vertex_coords[0]])
        # Create bounding box for cell
        for i in range(1, vertex_coords.shape[0]):
            for j in range(3):
                bbox_cell[0, j] = min(bbox_cell[0, j], vertex_coords[i, j])
                bbox_cell[1, j] = max(bbox_cell[1, j], vertex_coords[i, j])
        distance = compute_distance_gjk(expand_bbox(bbox_cell, dtype), bounding_box)
        if np.dot(distance, distance) < 1e-16:
            colliding_cells.append(cell)

    return colliding_cells


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_padded_bbox(padding, dtype):
    """Test collision between two meshes separated by a distance of
    epsilon, and check if padding the mesh creates a possible
    collision"""
    eps = 1e-4
    x0 = np.array([0, 0, 0], dtype=dtype)
    x1 = np.array([1, 1, 1 - eps], dtype=dtype)
    mesh_0 = create_box(MPI.COMM_WORLD, [x0, x1], [1, 1, 2], CellType.hexahedron, dtype=dtype)
    x2 = np.array([0, 0, 1 + eps], dtype=dtype)
    x3 = np.array([1, 1, 2], dtype=dtype)
    mesh_1 = create_box(MPI.COMM_WORLD, [x2, x3], [1, 1, 2], CellType.hexahedron, dtype=dtype)
    if padding:
        pad = eps
    else:
        pad = 0

    bbox_0 = bb_tree(mesh_0, mesh_0.topology.dim, padding=pad)
    bbox_1 = bb_tree(mesh_1, mesh_1.topology.dim, padding=pad)
    collisions = compute_collisions_trees(bbox_0, bbox_1)
    if padding:
        assert len(collisions) == 1
        # Check that the colliding elements are separated by a distance
        # 2*epsilon
        element_0 = extract_geometricial_data(mesh_0, mesh_0.topology.dim, [collisions[0][0]])[0]
        element_1 = extract_geometricial_data(mesh_1, mesh_1.topology.dim, [collisions[0][1]])[0]
        distance = np.linalg.norm(compute_distance_gjk(element_0, element_1))
        assert np.isclose(distance, 2 * eps, rtol=1.0e-5, atol=1.0e-7)
    else:
        assert len(collisions) == 0


def rotation_matrix(axis, angle):
    # See https://en.wikipedia.org/wiki/Rotation_matrix,
    # Subsection: Rotation_matrix_from_axis_and_angle.
    if np.isclose(np.inner(axis, axis), 1):
        n_axis = axis
    else:
        # Normalize axis
        n_axis = axis / np.sqrt(np.inner(axis, axis))

    # Define cross product matrix of axis
    axis_x = np.array(
        [[0, -n_axis[2], n_axis[1]], [n_axis[2], 0, -n_axis[0]], [-n_axis[1], n_axis[0], 0]]
    )
    id = np.cos(angle) * np.eye(3)
    outer = (1 - np.cos(angle)) * np.outer(n_axis, n_axis)
    return np.sin(angle) * axis_x + id + outer


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_tree(dtype):
    mesh = create_unit_interval(MPI.COMM_WORLD, 16, dtype=dtype)
    bbtree = bb_tree(mesh, mesh.topology.dim, 0.0, np.array([], dtype=dtype))
    assert bbtree.num_bboxes == 0


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_collisions_point_1d(dtype):
    N = 16
    p = np.array([0.3, 0, 0], dtype=dtype)
    mesh = create_unit_interval(MPI.COMM_WORLD, N, dtype=dtype)
    dx = 1 / N
    cell_index = int(p[0] // dx)
    # Vertices of cell we should collide with
    vertices = np.array([[dx * cell_index, 0, 0], [dx * (cell_index + 1), 0, 0]], dtype=dtype)

    # Compute collision
    tdim = mesh.topology.dim
    tree = bb_tree(mesh, tdim, 0.0)
    entities = compute_collisions_points(tree, p)
    assert len(entities.array) == 1

    # Get the vertices of the geometry
    geom_entities = _cpp.mesh.entities_to_geometry(mesh._cpp_object, tdim, entities.array, False)[0]
    x = mesh.geometry.x
    cell_vertices = x[geom_entities]
    # Check that we get the cell with correct vertices
    assert np.allclose(cell_vertices, vertices)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("point", [np.array([0.52, 0, 0]), np.array([0.9, 0, 0])])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_collisions_tree_1d(point, dtype):
    mesh_A = create_unit_interval(MPI.COMM_WORLD, 16, dtype=dtype)

    def locator_A(x):
        return x[0] >= point[0]

    # Locate all vertices of mesh A that should collide
    vertices_A = _cpp.mesh.locate_entities(mesh_A._cpp_object, 0, locator_A)
    mesh_A.topology.create_connectivity(0, mesh_A.topology.dim)
    v_to_c = mesh_A.topology.connectivity(0, mesh_A.topology.dim)

    # Find all cells connected to vertex in the collision bounding box
    cells_A = np.sort(np.unique(np.hstack([v_to_c.links(vertex) for vertex in vertices_A])))

    mesh_B = create_unit_interval(MPI.COMM_WORLD, 16, dtype=dtype)
    bgeom = mesh_B.geometry.x
    bgeom += point

    def locator_B(x):
        return x[0] <= 1

    # Locate all vertices of mesh B that should collide
    vertices_B = _cpp.mesh.locate_entities(mesh_B._cpp_object, 0, locator_B)
    mesh_B.topology.create_connectivity(0, mesh_B.topology.dim)
    v_to_c = mesh_B.topology.connectivity(0, mesh_B.topology.dim)

    # Find all cells connected to vertex in the collision bounding box
    cells_B = np.sort(np.unique(np.hstack([v_to_c.links(vertex) for vertex in vertices_B])))

    # Find colliding entities using bounding box trees
    tree_A = bb_tree(mesh_A, mesh_A.topology.dim, 0.0)
    tree_B = bb_tree(mesh_B, mesh_B.topology.dim, 0.0)
    entities = compute_collisions_trees(tree_A, tree_B)
    entities_A = np.sort(np.unique([q[0] for q in entities]))
    entities_B = np.sort(np.unique([q[1] for q in entities]))
    assert np.allclose(entities_A, cells_A)
    assert np.allclose(entities_B, cells_B)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("point", [np.array([0.52, 0.51, 0.0]), np.array([0.9, -0.9, 0.0])])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_collisions_tree_2d(point, dtype):
    mesh_A = create_unit_square(MPI.COMM_WORLD, 3, 3, dtype=dtype)
    mesh_B = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=dtype)
    bgeom = mesh_B.geometry.x
    bgeom += point
    tree_A = bb_tree(mesh_A, mesh_A.topology.dim, 0.0)
    tree_B = bb_tree(mesh_B, mesh_B.topology.dim, 0.0)
    entities = compute_collisions_trees(tree_A, tree_B)

    entities_A = np.sort(np.unique([q[0] for q in entities]))
    entities_B = np.sort(np.unique([q[1] for q in entities]))
    cells_A = find_colliding_cells(mesh_A, tree_B.get_bbox(tree_B.num_bboxes - 1), dtype)
    cells_B = find_colliding_cells(mesh_B, tree_A.get_bbox(tree_A.num_bboxes - 1), dtype)
    assert np.allclose(entities_A, cells_A)
    assert np.allclose(entities_B, cells_B)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("point", [np.array([0.52, 0.51, 0.3]), np.array([0.9, -0.9, 0.3])])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_collisions_tree_3d(point, dtype):
    mesh_A = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, dtype=dtype)
    mesh_B = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, dtype=dtype)

    bgeom = mesh_B.geometry.x
    bgeom += point

    tree_A = bb_tree(mesh_A, mesh_A.topology.dim, 0.0)
    tree_B = bb_tree(mesh_B, mesh_B.topology.dim, 0.0)
    entities = compute_collisions_trees(tree_A, tree_B)
    entities_A = np.sort(np.unique([q[0] for q in entities]))
    entities_B = np.sort(np.unique([q[1] for q in entities]))
    cells_A = find_colliding_cells(mesh_A, tree_B.get_bbox(tree_B.num_bboxes - 1), dtype)
    cells_B = find_colliding_cells(mesh_B, tree_A.get_bbox(tree_A.num_bboxes - 1), dtype)
    assert np.allclose(entities_A, cells_A)
    assert np.allclose(entities_B, cells_B)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_closest_entity_1d(dim, dtype):
    ref_distance = 0.75
    N = 16
    points = np.array([[-ref_distance, 0, 0], [2 / N, 2 * ref_distance, 0]], dtype=dtype)
    mesh = create_unit_interval(MPI.COMM_WORLD, N, dtype=dtype)
    tree = bb_tree(mesh, dim, 0.0)
    num_entities_local = (
        mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
    )
    entities = np.arange(num_entities_local, dtype=np.int32)
    midpoint_tree = create_midpoint_tree(mesh, dim, entities)
    closest_entities = compute_closest_entity(tree, midpoint_tree, mesh, points)

    # Find which entity is colliding with known closest point on mesh
    p_c = np.array([[0, 0, 0], [2 / N, 0, 0]], dtype=dtype)
    colliding_entity_bboxes = compute_collisions_points(tree, p_c)

    # Refine search by checking for actual collision if the entities are
    # cells
    if dim == mesh.topology.dim:
        colliding_cells = compute_colliding_cells(mesh, colliding_entity_bboxes, p_c)
        for i in range(points.shape[0]):
            # If colliding entity is on process
            if colliding_cells.links(i).size > 0:
                assert np.isin(closest_entities[i], colliding_cells.links(i))
    else:
        for i in range(points.shape[0]):
            # Only check closest entity if any bounding box on the
            # process intersects with the point
            if colliding_entity_bboxes.links(i).size > 0:
                assert np.isin(closest_entities[i], colliding_entity_bboxes.links(i))


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_closest_entity_2d(dim, dtype):
    points = np.array([-1.0, -0.01, 0.0], dtype=dtype)
    mesh = create_unit_square(MPI.COMM_WORLD, 15, 15, dtype=dtype)
    mesh.topology.create_entities(dim)
    tree = bb_tree(mesh, dim, 0.0)
    num_entities_local = (
        mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
    )
    entities = np.arange(num_entities_local, dtype=np.int32)
    midpoint_tree = create_midpoint_tree(mesh, dim, entities)

    # Find which entity is colliding with known closest point on mesh
    closest_entities = compute_closest_entity(tree, midpoint_tree, mesh, points)

    # Find which entity is colliding with known closest point on mesh
    p_c = np.array([0, 0, 0], dtype=dtype)
    colliding_entity_bboxes = compute_collisions_points(tree, p_c)

    # Refine search by checking for actual collision if the entities are
    # cells
    if dim == mesh.topology.dim:
        colliding_cells = compute_colliding_cells(mesh, colliding_entity_bboxes, p_c).array
        if len(colliding_cells) > 0:
            assert np.isin(closest_entities[0], colliding_cells)
    else:
        if len(colliding_entity_bboxes.links(0)) > 0:
            assert np.isin(closest_entities[0], colliding_entity_bboxes.links(0))


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_closest_entity_3d(dim, dtype):
    points = np.array([[0.9, 0, 1.135]], dtype=dtype)
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, dtype=dtype)
    mesh.topology.create_entities(dim)

    tree = bb_tree(mesh, dim, 0.0)
    num_entities_local = (
        mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
    )
    entities = np.arange(num_entities_local, dtype=np.int32)
    midpoint_tree = create_midpoint_tree(mesh, dim, entities)
    closest_entities = compute_closest_entity(tree, midpoint_tree, mesh, points)

    # Find which entity is colliding with known closest point on mesh
    p_c = np.array([0.9, 0, 1], dtype=dtype)
    colliding_entity_bboxes = compute_collisions_points(tree, p_c)

    # Refine search by checking for actual collision if the entities are
    # cells
    if dim == mesh.topology.dim:
        colliding_cells = compute_colliding_cells(mesh, colliding_entity_bboxes, p_c).array
        if len(colliding_cells) > 0:
            assert np.isin(closest_entities[0], colliding_cells)
    else:
        if len(colliding_entity_bboxes.links(0)) > 0:
            assert np.isin(closest_entities[0], colliding_entity_bboxes.links(0))


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_closest_sub_entity(dim, dtype):
    """Compute distance from subset of cells in a mesh to a point inside the mesh"""
    ref_distance = 0.31
    xc, yc, zc = 0.5, 0.5, 0.5
    points = np.array([xc + ref_distance, yc, zc], dtype=dtype)
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, dtype=dtype)
    mesh.topology.create_entities(dim)
    left_entities = locate_entities(mesh, dim, lambda x: x[0] <= xc)
    tree = bb_tree(mesh, dim, 0.0, left_entities)
    midpoint_tree = create_midpoint_tree(mesh, dim, left_entities)
    closest_entities = compute_closest_entity(tree, midpoint_tree, mesh, points)

    # Find which entity is colliding with known closest point on mesh
    p_c = np.array([xc, yc, zc], dtype=dtype)
    colliding_entity_bboxes = compute_collisions_points(tree, p_c)

    # Refine search by checking for actual collision if the entities are
    # cells
    if dim == mesh.topology.dim:
        colliding_cells = compute_colliding_cells(mesh, colliding_entity_bboxes, p_c).array
        if len(colliding_cells) > 0:
            assert np.isin(closest_entities[0], colliding_cells)
    else:
        if len(colliding_entity_bboxes.links(0)) > 0:
            assert np.isin(closest_entities[0], colliding_entity_bboxes.links(0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_surface_bbtree(dtype):
    """Test creation of BBTree on subset of entities(surface cells)"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, dtype=dtype)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    sf = exterior_facet_indices(mesh.topology)
    tdim = mesh.topology.dim
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = np.array([f_to_c.links(f)[0] for f in sf], dtype=np.int32)
    bbtree = bb_tree(mesh, tdim, 0.0, cells)

    # test collision (should not collide with any)
    p = np.array([0.5, 0.5, 0.5])
    assert len(compute_collisions_points(bbtree, p).array) == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sub_bbtree_codim1(dtype):
    """Testing point collision with a BoundingBoxTree of sub entities"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, cell_type=CellType.hexahedron, dtype=dtype)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    top_facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 1))
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = np.array([f_to_c.links(f)[0] for f in top_facets], dtype=np.int32)
    bbtree = bb_tree(mesh, tdim, 0.0, cells)

    # Compute a BBtree for all processes
    process_bbtree = bbtree.create_global_tree(mesh.comm)

    # Find possible ranks for this point
    point = np.array([0.2, 0.2, 1.0], dtype=dtype)
    ranks = compute_collisions_points(process_bbtree, point)

    # Compute local collisions
    cells = compute_collisions_points(bbtree, point)
    if MPI.COMM_WORLD.rank in ranks.array:
        assert len(cells.links(0)) > 0
    else:
        assert len(cells.links(0)) == 0


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD, MPI.COMM_SELF])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_serial_global_bb_tree(dtype, comm):
    # Test if global bb tree with only one node returns the correct collision
    mesh = create_unit_cube(comm, 4, 5, 3)

    # First point should not be in any tree
    # Second point should always be in the global tree, but only in
    # entity tree with a serial mesh
    x = np.array([[2.0, 2.0, 3.0], [0.3, 0.2, 0.1]], dtype=dtype)

    tree = bb_tree(mesh, mesh.topology.dim, 0.0)
    global_tree = tree.create_global_tree(mesh.comm)

    tree_col = compute_collisions_points(tree, x)
    global_tree_col = compute_collisions_points(global_tree, x)
    assert len(tree_col.links(0)) == 0 and len(global_tree_col.links(0)) == 0
    assert len(global_tree_col.links(1)) > 0
    # Only guaranteed local tree collision if mesh is on one process
    if comm.size == 1:
        assert len(tree_col.links(1)) > 0


@pytest.mark.parametrize("ct", [CellType.hexahedron, CellType.tetrahedron])
@pytest.mark.parametrize("N", [7, 13])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sub_bbtree_box(ct, N, dtype):
    """Test that the bounding box of the stem of the bounding box tree is what we expect"""
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, cell_type=ct, dtype=dtype)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 1.0))
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    cells = np.int32(np.unique([f_to_c.links(f)[0] for f in facets]))
    bbtree = bb_tree(mesh, tdim, 0.0, cells)
    num_boxes = bbtree.num_bboxes
    if num_boxes > 0:
        bbox = bbtree.get_bbox(num_boxes - 1)
        assert np.isclose(bbox[0][1], (N - 1) / N)
    tree = bb_tree(mesh, tdim, 0.0)
    assert num_boxes < tree.num_bboxes


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_surface_bbtree_collision(dtype):
    """Compute collision between two meshes, where only one cell of each mesh are colliding"""
    tdim = 3
    mesh1 = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron, dtype=dtype)
    mesh2 = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron, dtype=dtype)
    mesh2.geometry.x[:, :] += np.array([0.9, 0.9, 0.9])

    mesh1.topology.create_connectivity(mesh1.topology.dim - 1, mesh1.topology.dim)
    sf = exterior_facet_indices(mesh1.topology)
    f_to_c = mesh1.topology.connectivity(tdim - 1, tdim)

    # Compute unique set of cells (some will be counted multiple times)
    cells = np.array(list(set([f_to_c.links(f)[0] for f in sf])), dtype=np.int32)
    bbtree1 = bb_tree(mesh1, tdim, 0.0, cells)

    mesh2.topology.create_connectivity(mesh2.topology.dim - 1, mesh2.topology.dim)
    sf = exterior_facet_indices(mesh2.topology)
    f_to_c = mesh2.topology.connectivity(tdim - 1, tdim)
    cells = np.array(list(set([f_to_c.links(f)[0] for f in sf])), dtype=np.int32)
    bbtree2 = bb_tree(mesh2, tdim, 0.0, cells)

    collisions = compute_collisions_trees(bbtree1, bbtree2)
    assert len(collisions) == 1


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_determine_point_ownership(dim, affine, dtype):
    """Find point owners (ranks and cells) using bounding box trees + global communication
    and compare to point ownership data results."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_dtype = MPI.DOUBLE if dtype == np.float64 else MPI.FLOAT

    tdim = dim
    num_cells_side = 4
    if tdim == 2:
        ct = CellType.triangle if affine else CellType.quadrilateral
        mesh = create_unit_square(MPI.COMM_WORLD, num_cells_side, num_cells_side, ct, dtype=dtype)
    else:
        ct = CellType.tetrahedron if affine else CellType.hexahedron
        mesh = create_unit_cube(
            MPI.COMM_WORLD,
            num_cells_side,
            num_cells_side,
            num_cells_side,
            ct,
            dtype=dtype,
        )
    cell_map = mesh.topology.index_map(tdim)

    tree = bb_tree(mesh, mesh.topology.dim, 0.0, np.arange(cell_map.size_local))
    num_global_cells = num_cells_side**tdim
    if affine:
        num_global_cells *= 2 * (3 ** (tdim - 2))
    local_midpoints = compute_midpoints(
        mesh, tdim, np.arange(mesh.topology.index_map(tdim).size_local)
    )
    midpoints_per_rank = np.zeros(comm.size, dtype=np.int32)
    midpoints_offsets = np.zeros(comm.size, dtype=np.int32)
    comm.Allgather(np.array([local_midpoints.shape[0]], dtype=np.int32), midpoints_per_rank)
    midpoints_offsets[1:] = np.cumsum(midpoints_per_rank[:-1])
    all_midpoints = np.zeros((num_global_cells, 3), dtype=dtype)
    comm.Allgatherv(
        local_midpoints, [all_midpoints, midpoints_per_rank * 3, midpoints_offsets * 3, mpi_dtype]
    )
    # Find potential owner cells
    tree_col = compute_collisions_points(tree, all_midpoints)

    mesh.topology.create_connectivity(tdim - 1, 0)
    mesh.topology.create_connectivity(0, tdim)
    cfc = mesh.topology.connectivity(tdim, tdim - 1)
    fpc = mesh.topology.connectivity(tdim - 1, 0)

    # Narrow it down to a single owner cell
    def is_inside(mesh, icell, point):
        fdim = tdim - 1
        is_inside = True
        cpoints = mesh.geometry.x[mesh.geometry.dofmap[icell, :]]  # cell points
        ccentroid = np.average(cpoints, axis=0)  # cell centroid
        for ifacet in cfc.links(icell):
            fpoints_indices = _cpp.mesh.entities_to_geometry(
                mesh._cpp_object,
                0,
                fpc.links(ifacet),
                False,
            )
            fpoints_indices = fpoints_indices.reshape(fpoints_indices.size)
            fpoints = mesh.geometry.x[fpoints_indices]
            fcentroid = np.average(fpoints, axis=0)  # facet centroid
            # Compute facet normal pointing to outside of owner cell
            normal = np.zeros(3, dtype=dtype)
            facet_vector1 = fpoints[1, :] - fpoints[0, :]
            if fdim == 1:
                normal[0] = -facet_vector1[1]
                normal[1] = +facet_vector1[0]
            elif fdim == 2:
                facet_vector2 = fpoints[2, :] - fpoints[0, :]
                normal = np.cross(facet_vector1, facet_vector2)
            else:
                raise ValueError("Unexpected facet dimension.")
            normal /= np.linalg.norm(normal)
            # Re-align if pointing to inside the parent cell
            normal = -normal if (np.dot((ccentroid - fcentroid), normal) > 0) else normal
            # Test the point
            signed_distance = np.dot((point - fcentroid), normal)
            if signed_distance > 1e-9:
                is_inside = False
                break
        return is_inside

    processwise_owners = np.zeros(2 * num_global_cells, dtype=np.int32)
    owners = np.empty_like(processwise_owners)
    for ipoint in range(num_global_cells):
        potential_owners = tree_col.links(ipoint)
        owner_cells = []
        for cell in potential_owners:
            if is_inside(mesh, cell, all_midpoints[ipoint, :]):
                owner_cells.append(cell)
        if owner_cells:
            assert len(owner_cells) == 1
            processwise_owners[2 * ipoint] = rank
            processwise_owners[2 * ipoint + 1] = owner_cells[0]

    # Since ghost cells are left out and the points considered are midpoints
    # of cells, they are only contained in a single process / cell
    # The value at a given index is null if it doesn't correspond
    # to the current process.
    # We can sum the processwise arrays to obtain a global array
    comm.Allreduce(processwise_owners, owners, op=MPI.SUM)
    owner_ranks = owners[[2 * i for i in range(num_global_cells)]]
    owner_cells = owners[[2 * i + 1 for i in range(num_global_cells)]]

    # Reorganize ownership data (point, index, rank, cell) into dictionary
    ownership_data = {}
    for ipoint in range(num_global_cells):
        ownership_data[tuple(all_midpoints[ipoint])] = (
            ipoint,
            owner_ranks[ipoint],
            owner_cells[ipoint],
        )

    def check_po(po: PointOwnershipData, src_points, ownership_data, global_dest_owners):
        """
        Check point ownership data

        po: PointOwnershipData object to check
        src_points: Points sent by process
        ownership_data: {point:(global_index,rank,cell}
        global_dest_owners: Rank who sent each point
        """
        # Check src_owner: Check owner ranks of sent points
        src_owner = po.src_owner()
        for ipoint in range(src_points.shape[0]):
            assert ownership_data[tuple(src_points[ipoint])][1] == src_owner[ipoint]

        dest_points = po.dest_points()
        dest_owners = po.dest_owner()
        dest_cells = po.dest_cells()

        # Check dest_points: All points that should have been found have been found
        dest_points_indices = list(range(dest_points.shape[0]))
        for point, data in ownership_data.items():
            (iglobal, processor, _) = data
            if processor == rank:
                found = False
                point = np.array(point, dtype=dtype)
                for jpoint in dest_points_indices:
                    found = np.allclose(point, dest_points[jpoint])
                    if found:
                        break
                assert found
                dest_points_indices.remove(jpoint)

        # Check dest_owners and dest_cells
        # dest_owners: Ranks that asked about the points we own
        # dest_cells: Local index of cell that contains the points we own
        for ipoint in range(dest_points.shape[0]):
            iglobal = ownership_data[tuple(dest_points[ipoint])][0]
            c = ownership_data[tuple(dest_points[ipoint])][2]
            assert dest_owners[ipoint] == global_dest_owners[iglobal]
            assert dest_cells[ipoint] == c

    def set_local_range(array):
        N = array.shape[0]
        n = N // comm.size
        r = N % comm.size
        # First r processes has one extra value
        if rank < r:
            (start, stop) = [rank * (n + 1), (rank + 1) * (n + 1)]
        else:
            (start, stop) = [rank * n + r, (rank + 1) * n + r]
        return array[start:stop], start, stop

    def compute_global_owners(N, start, stop):
        """Compute array of ranks who own each point"""
        mask_points_owned = np.zeros(N, np.int32)
        global_owners = np.empty_like(mask_points_owned)
        mask_points_owned[start:stop] = rank
        comm.Allreduce(mask_points_owned, global_owners, op=MPI.SUM)
        return global_owners

    # All cells
    points, start, stop = set_local_range(all_midpoints)
    owners = compute_global_owners(np.int64(all_midpoints.shape[0]), start, stop)
    all_cells = np.arange(cell_map.size_local, dtype=dtype)
    po = determine_point_ownership(mesh, points, 0.0, all_cells)

    check_po(po, points, ownership_data, owners)

    # Left half
    num_left_cells = np.rint(num_global_cells / 2).astype(np.int32)
    left_midpoints = np.zeros((num_left_cells, 3), dtype=dtype)
    counter = 0
    indices_left = []
    for ipoint in range(num_global_cells):
        if all_midpoints[ipoint, 0] <= 0.5:
            left_midpoints[counter] = all_midpoints[ipoint]
            indices_left.append(ipoint)
            counter += 1
    points, start, stop = set_local_range(left_midpoints)
    owners = compute_global_owners(np.int64(all_midpoints.shape[0]), start, stop)
    left_cells = locate_entities(mesh, tdim, lambda x: x[0] <= 0.5)
    left_cells = np.array(
        [cell for cell in left_cells if cell < cell_map.size_local], dtype=np.int32
    )  # Filter out ghost cells
    lpo = determine_point_ownership(mesh, points, 0.0, left_cells)

    left_ownership_data = {}
    for idx, ipoint in enumerate(indices_left):
        left_ownership_data[tuple(all_midpoints[ipoint])] = (
            idx,
            owner_ranks[ipoint],
            owner_cells[ipoint],
        )
    check_po(lpo, points, left_ownership_data, owners)
