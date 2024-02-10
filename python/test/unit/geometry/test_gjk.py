# Copyright (C) 2020 Jørgen S. Dokken and Chris N. Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import ufl
from basix.ufl import element
from dolfinx import geometry
from dolfinx.geometry import compute_distance_gjk
from dolfinx.mesh import create_mesh


def distance_point_to_line_3D(P1, P2, point):
    """Distance from point to line"""
    return np.linalg.norm(np.cross(P2 - P1, P1 - point)) / np.linalg.norm(P2 - P1)


def distance_point_to_plane_3D(P1, P2, P3, point):
    """Distance from point to plane"""
    return np.abs(
        np.dot(np.cross(P2 - P1, P3 - P1) / np.linalg.norm(np.cross(P2 - P1, P3 - P1)), point - P2)
    )


@pytest.mark.parametrize("delta", [0.1, 1e-12, 0, -2])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_line_point_distance(delta, dtype):
    line = np.array([[0.1, 0.2, 0.3], [0.5, 0.8, 0.7]], dtype=dtype)
    point_on_line = line[0] + 0.27 * (line[1] - line[0])
    normal = np.cross(line[0], line[1])
    point = point_on_line + delta * normal
    d = compute_distance_gjk(line, point)
    distance = np.linalg.norm(d)
    actual_distance = distance_point_to_line_3D(line[0], line[1], point)
    assert np.isclose(distance, actual_distance, atol=1e-8)


@pytest.mark.parametrize("delta", [0.1, 1e-12, 0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_line_line_distance(delta, dtype):
    line = np.array([[-0.5, -0.7, -0.3], [1, 2, 3]], dtype=dtype)
    point_on_line = line[0] + 0.38 * (line[1] - line[0])
    normal = np.cross(line[0], line[1])
    point = point_on_line + delta * normal
    line_2 = np.array([point, [2, 5, 6]], dtype=dtype)
    distance = np.linalg.norm(compute_distance_gjk(line, line_2))
    actual_distance = distance_point_to_line_3D(line[0], line[1], line_2[0])
    assert np.isclose(distance, actual_distance, atol=1e-7)


@pytest.mark.parametrize("delta", [0.1 ** (3 * i) for i in range(6)])
@pytest.mark.parametrize("dtype", [np.float64])
def test_tri_distance(delta, dtype):
    tri_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=dtype)
    tri_2 = np.array([[1, delta, 0], [3, 1.2, 0], [1, 1, 0]], dtype=dtype)
    P1 = tri_1[2]
    P2 = tri_1[1]
    point = tri_2[0]
    actual_distance = distance_point_to_line_3D(P1, P2, point)
    distance = np.linalg.norm(compute_distance_gjk(tri_1, tri_2))
    assert np.isclose(distance, actual_distance, atol=1e-6)


@pytest.mark.parametrize("delta", [0.1 * 0.1 ** (3 * i) for i in range(6)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_quad_distance2d(delta, dtype):
    quad_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=dtype)
    quad_2 = np.array([[0, 1 + delta, 0], [2, 2, 0], [2, 4, 0], [4, 4, 0]], dtype=dtype)
    P1 = quad_1[2]
    P2 = quad_1[3]
    point = quad_2[0]
    actual_distance = distance_point_to_line_3D(P1, P2, point)
    distance = np.linalg.norm(compute_distance_gjk(quad_1, quad_2))
    assert np.isclose(distance, actual_distance, atol=1e-8)


@pytest.mark.parametrize("delta", [1 * 0.5 ** (3 * i) for i in range(7)])
def test_tetra_distance_3d(delta):
    tetra_1 = np.array([[0, 0, 0.2], [1, 0, 0.1], [0, 1, 0.3], [0, 0, 1]], dtype=np.float64)
    tetra_2 = np.array([[0, 0, -3], [1, 0, -3], [0, 1, -3], [0.5, 0.3, -delta]], dtype=np.float64)
    actual_distance = distance_point_to_plane_3D(tetra_1[0], tetra_1[1], tetra_1[2], tetra_2[3])
    distance = np.linalg.norm(compute_distance_gjk(tetra_1, tetra_2))
    assert np.isclose(distance, actual_distance, atol=1e-15)


@pytest.mark.parametrize("delta", [(-1) ** i * np.sqrt(2) * 0.1 ** (3 * i) for i in range(6)])
def test_tetra_collision_3d(delta):
    tetra_1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tetra_2 = np.array([[0, 0, -3], [1, 0, -3], [0, 1, -3], [0.5, 0.3, -delta]], dtype=np.float64)
    actual_distance = distance_point_to_plane_3D(tetra_1[0], tetra_1[1], tetra_1[2], tetra_2[3])
    distance = np.linalg.norm(compute_distance_gjk(tetra_1, tetra_2))
    if delta < 0:
        assert np.isclose(distance, 0, atol=1e-15)
    else:
        assert np.isclose(distance, actual_distance, atol=1e-15)


@pytest.mark.parametrize("delta", [0, -0.1, -0.49, -0.51])
def test_hex_collision_3d(delta):
    hex_1 = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=np.float64,
    )
    P0 = np.array([1.5 + delta, 1.5 + delta, 0.5], dtype=np.float64)
    P1 = np.array([2, 2, 1], dtype=np.float64)
    P2 = np.array([2, 1.25, 0.25], dtype=np.float64)
    P3 = P1 + P2 - P0
    quad_1 = np.array([P0, P1, P2, P3], dtype=np.float64)
    n = np.cross(quad_1[1] - quad_1[0], quad_1[2] - quad_1[0]) / np.linalg.norm(
        np.cross(quad_1[1] - quad_1[0], quad_1[2] - quad_1[0])
    )
    quad_2 = quad_1 + n
    hex_2 = np.zeros((8, 3), dtype=np.float64)
    hex_2[:4, :] = quad_1
    hex_2[4:, :] = quad_2
    actual_distance = np.linalg.norm(np.array([1, 1, P0[2]], dtype=np.float64) - hex_2[0])
    distance = np.linalg.norm(compute_distance_gjk(hex_1, hex_2))
    if P0[0] < 1:
        assert np.isclose(distance, 0, atol=1e-15)
    else:
        assert np.isclose(distance, actual_distance, atol=1e-15)


@pytest.mark.parametrize("delta", [1e8, 1.0, 1e-6, 1e-12])
@pytest.mark.parametrize("scale", [1000.0, 1.0, 1e-4])
@pytest.mark.parametrize("dtype", [np.float64])
def test_cube_distance(delta, scale, dtype):
    cubes = [
        scale
        * np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [-1, 1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, 1],
                [1, 1, 1],
            ],
            dtype=dtype,
        )
    ]

    # Rotate cube 45 degrees around z, so that an edge faces along
    # x-axis (vertical)
    r = Rotation.from_euler("z", 45, degrees=True)
    cubes.append(r.apply(cubes[0]))

    # Rotate cube around y, so that a corner faces along the x-axis
    r = Rotation.from_euler("y", np.arctan2(1.0, np.sqrt(2)))
    cubes.append(r.apply(cubes[1]))

    # Rotate cube 45 degrees around y, so that an edge faces along
    # x-axis (horizontal)
    r = Rotation.from_euler("y", 45, degrees=True)
    cubes.append(r.apply(cubes[0]))

    # Rotate scene through an arbitrary angle
    r = Rotation.from_euler("xyz", [22, 13, -47], degrees=True)

    for c0 in range(4):
        for c1 in range(4):
            dx = cubes[c0][:, 0].max() - cubes[c1][:, 0].min()
            cube0 = cubes[c0]
            # Separate cubes along x-axis by distance delta
            cube1 = cubes[c1] + np.array([dx + delta, 0, 0])
            c0rot = r.apply(cube0)
            c1rot = r.apply(cube1)
            distance = np.linalg.norm(compute_distance_gjk(c0rot, c1rot))
            assert np.isclose(distance, delta)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_collision_2nd_order_triangle(dtype):
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.65, 0.65], [0.0, 0.5], [0.5, 0.0]], dtype=dtype
    )
    cells = np.array([[0, 1, 2, 3, 4, 5]])
    domain = ufl.Mesh(element("Lagrange", "triangle", 2, shape=(2,), dtype=dtype))
    mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)

    # Sample points along an interior line of the domain. The last point
    # is outside the simplex made by the vertices.
    sample_points = np.array([[0.1, 0.3, 0.0], [0.2, 0.5, 0.0], [0.6, 0.6, 0.0]])

    # Create boundingboxtree
    tree = geometry.bb_tree(mesh, mesh.geometry.dim)
    cell_candidates = geometry.compute_collisions_points(tree, sample_points)
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, sample_points)
    # Check for collision
    for i in range(colliding_cells.num_nodes):
        assert len(colliding_cells.links(i)) == 1

    # Check if there is a point on the linear approximation of the
    # curved facet
    def line_through_points(p0, p1):
        return lambda x: (p1[1] - p0[1]) / (p1[0] - p0[0]) * (x - p0[0]) + p0[1]

    line_func = line_through_points(points[2], points[3])
    point = np.array([0.2, line_func(0.2), 0])

    # Point inside 2nd order geometry, outside linear approximation
    # Useful for debugging on a later stage
    # point = np.array([0.25, 0.89320760, 0])
    distance = geometry.squared_distance(mesh, mesh.topology.dim - 1, np.array([2]), point)
    assert np.isclose(distance, 0)
