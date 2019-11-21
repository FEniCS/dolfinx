# Copyright (C) 2009-2019 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the fem interface"""

from random import shuffle

import numpy as np
import pytest
from dolfin_utils.test.skips import skip_in_parallel

from dolfin import MPI, FunctionSpace, cpp, fem, Mesh, FacetNormal, Function
from dolfin.cpp.mesh import CellType
from ufl import inner, grad, ds

xfail = pytest.mark.xfail(strict=True)


@pytest.mark.parametrize('space_type', [
    ("P", 1), ("P", 2), ("P", 3), ("P", 4),
    ("N1curl", 1),
    ("RT", 1), ("RT", 2), ("RT", 3), ("RT", 4),
    ("BDM", 1),
    ("N2curl", 1),
])
def test_triangle_dof_ordering(space_type):
    """Checks that dofs on shared triangle edges match up"""
    # Create a triangle mesh
    if MPI.rank(MPI.comm_world) == 0:
        N = 6
        # Create a grid of points [0, 0.5, ..., 9.5]**2, then order them
        # in a random order
        temp_points = np.array([[x / 2, y / 2] for x in range(N) for y in range(N)])
        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        # Make triangle cells using the randomly ordered points
        cells = []
        for x in range(N - 1):
            for y in range(N - 1):
                a = N * y + x
                # Adds two triangle cells:
                # a+N -- a+N+1
                #  |   / |
                #  |  /  |
                #  | /   |
                #  a --- a+1
                for cell in [[a, a + 1, a + N + 1], [a, a + N + 1, a + N]]:
                    cells.append([order[i] for i in cell])

        # On process 0, input mesh data and distribute to other
        # processes
        mesh = cpp.mesh.build_distributed_mesh(MPI.comm_world, CellType.triangle, points,
                                               np.array(cells), [], cpp.mesh.GhostMode.none,
                                               cpp.mesh.Partitioner.scotch)
    else:
        # On other processes, accept distributed data
        mesh = cpp.mesh.build_distributed_mesh(MPI.comm_world, CellType.triangle, np.ndarray((0, 2)),
                                               np.ndarray((0, 3)), [], cpp.mesh.GhostMode.none,
                                               cpp.mesh.Partitioner.scotch)

    V = FunctionSpace(mesh, space_type)
    dofmap = V.dofmap

    edges = {}

    # Get coordinates of dofs and edges and check that they are the same
    # for each global dof number
    X = V.element.dof_reference_coordinates()
    coord_dofs = mesh.coordinate_dofs().entity_points()
    x_g = mesh.geometry.points
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    for cell_n, cell in enumerate(coord_dofs):
        dofs = dofmap.cell_dofs(cell_n)

        x_coord_new = np.zeros([3, 2])
        for v in range(3):
            x_coord_new[v] = x_g[coord_dofs[cell_n, v], :2]
        x = X.copy()
        cmap.push_forward(x, X, x_coord_new)

        edge_dofs_local = []
        for i in range(3):
            edge_dofs_local += list(dofmap.dof_layout.entity_dofs(1, i))
        edge_dofs = [dofs[i] for i in edge_dofs_local]
        for i, j in zip(edge_dofs, x[edge_dofs_local]):
            if i in edges:
                assert np.allclose(j, edges[i])
            else:
                edges[i] = j


@pytest.mark.parametrize('space_type', [
    ("P", 1), ("P", 2), ("P", 3), ("P", 4),
    ("N1curl", 1), ("N1curl", 2),
    ("RT", 1), ("RT", 2), ("RT", 3), ("RT", 4),
    ("BDM", 1),
    ("N2curl", 1),
])
def test_tetrahedron_dof_ordering(space_type):
    """Checks that dofs on shared tetrahedron edges and faces match up"""
    if MPI.rank(MPI.comm_world) == 0:
        # Create simple tetrahedron mesh
        N = 3
        temp_points = np.array([[x / 2, y / 2, z / 2] for x in range(N) for y in range(N) for z in range(N)])

        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        cells = []
        for x in range(N - 1):
            for y in range(N - 1):
                for z in range(N - 1):
                    a = N ** 2 * z + N * y + x
                    for c in [[a + N, a + N ** 2 + 1, a, a + 1],
                              [a + N, a + N ** 2 + 1, a + 1, a + N + 1],
                              [a + N, a + N ** 2 + 1, a + N + 1, a + N ** 2 + N + 1],
                              [a + N, a + N ** 2 + 1, a + N ** 2 + N + 1, a + N ** 2 + N],
                              [a + N, a + N ** 2 + 1, a + N ** 2 + N, a + N ** 2],
                              [a + N, a + N ** 2 + 1, a + N ** 2, a]]:
                        cell = [order[i] for i in c]
                        cells.append(cell)
        # On process 0, input mesh data and distribute to other
        # processes
        mesh = cpp.mesh.build_distributed_mesh(MPI.comm_world, CellType.tetrahedron, points,
                                               np.array(cells), [], cpp.mesh.GhostMode.none,
                                               cpp.mesh.Partitioner.scotch)
    else:
        # On other processes, accept distributed data
        mesh = cpp.mesh.build_distributed_mesh(MPI.comm_world, CellType.tetrahedron, np.ndarray((0, 3)),
                                               np.ndarray((0, 4)), [], cpp.mesh.GhostMode.none,
                                               cpp.mesh.Partitioner.scotch)

    V = FunctionSpace(mesh, space_type)
    dofmap = V.dofmap

    edges = {}
    faces = {}

    # Get coordinates of dofs and edges and check that they are the same
    # for each global dof number
    X = V.element.dof_reference_coordinates()
    coord_dofs = mesh.coordinate_dofs().entity_points()
    x_g = mesh.geometry.points
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    for cell_n, cell in enumerate(coord_dofs):
        dofs = dofmap.cell_dofs(cell_n)

        x_coord_new = np.zeros([4, 3])
        for v in range(4):
            x_coord_new[v] = x_g[coord_dofs[cell_n, v]]
        x = X.copy()
        cmap.push_forward(x, X, x_coord_new)

        edge_dofs_local = []
        for i in range(6):
            edge_dofs_local += list(dofmap.dof_layout.entity_dofs(1, i))
        edge_dofs = [dofs[i] for i in edge_dofs_local]
        for i, j in zip(edge_dofs, x[edge_dofs_local]):
            if i in edges:
                assert np.allclose(j, edges[i])
            else:
                edges[i] = j

        face_dofs_local = []
        for i in range(4):
            face_dofs_local += list(dofmap.dof_layout.entity_dofs(2, i))
        face_dofs = [dofs[i] for i in face_dofs_local]
        for i, j in zip(face_dofs, x[face_dofs_local]):
            if i in faces:
                assert np.allclose(j, faces[i])
            else:
                faces[i] = j


@pytest.mark.parametrize('space_type', [
    ("P", 1), ("P", 2), ("P", 3), ("P", 4),
])
def test_quadrilateral_dof_ordering(space_type):
    """Checks that dofs on shared quadrilateral edges match up"""
    if MPI.rank(MPI.comm_world) == 0:
        # Create a quadrilateral mesh
        N = 10
        temp_points = np.array([[x / 2, y / 2] for x in range(N) for y in range(N)])

        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        cells = []
        for x in range(N - 1):
            for y in range(N - 1):
                a = N * y + x
                cell = [order[i] for i in [a, a + 1, a + N, a + N + 1]]
                cells.append(cell)

        # On process 0, input mesh data and distribute to other
        # processes
        mesh = cpp.mesh.build_distributed_mesh(MPI.comm_world, CellType.quadrilateral, points,
                                               np.array(cells), [], cpp.mesh.GhostMode.none,
                                               cpp.mesh.Partitioner.scotch)
    else:
        # On other processes, accept distributed data
        mesh = cpp.mesh.build_distributed_mesh(MPI.comm_world, CellType.quadrilateral, np.ndarray((0, 2)),
                                               np.ndarray((0, 4)), [], cpp.mesh.GhostMode.none,
                                               cpp.mesh.Partitioner.scotch)

    V = FunctionSpace(mesh, space_type)
    dofmap = V.dofmap

    edges = {}

    # Get coordinates of dofs and edges and check that they are the same
    # for each global dof number
    X = V.element.dof_reference_coordinates()
    coord_dofs = mesh.coordinate_dofs().entity_points()
    x_g = mesh.geometry.points
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    for cell_n, cell in enumerate(coord_dofs):
        dofs = dofmap.cell_dofs(cell_n)

        x_coord_new = np.zeros([4, 2])
        for v in range(4):
            x_coord_new[v] = x_g[coord_dofs[cell_n, v], :2]
        x = X.copy()
        cmap.push_forward(x, X, x_coord_new)

        edge_dofs_local = []
        for i in range(4):
            edge_dofs_local += list(dofmap.dof_layout.entity_dofs(1, i))
        edge_dofs = [dofs[i] for i in edge_dofs_local]
        for i, j in zip(edge_dofs, x[edge_dofs_local]):
            if i in edges:
                assert np.allclose(j, edges[i])
            else:
                edges[i] = j


@pytest.mark.parametrize('space_type', [
    ("P", 1), ("P", 2), ("P", 3), ("P", 4),
])
def test_hexahedron_dof_ordering(space_type):
    """Checks that dofs on shared hexahedron edges match up"""
    if MPI.rank(MPI.comm_world) == 0:
        # Create a hexahedron mesh
        N = 5
        temp_points = np.array([[x / 2, y / 2, z / 2] for x in range(N) for y in range(N) for z in range(N)])

        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        cells = []
        for x in range(N - 1):
            for y in range(N - 1):
                for z in range(N - 1):
                    a = N ** 2 * z + N * y + x
                    cell = [order[i] for i in [a, a + 1, a + N, a + N + 1,
                                               a + N ** 2, a + 1 + N ** 2, a + N + N ** 2,
                                               a + N + 1 + N ** 2]]
                    cells.append(cell)

        # On process 0, input mesh data and distribute to other
        # processes
        mesh = cpp.mesh.build_distributed_mesh(MPI.comm_world, CellType.hexahedron, points,
                                               np.array(cells), [], cpp.mesh.GhostMode.none,
                                               cpp.mesh.Partitioner.scotch)
    else:
        # On other processes, accept distributed data
        mesh = cpp.mesh.build_distributed_mesh(MPI.comm_world, CellType.hexahedron, np.ndarray((0, 3)),
                                               np.ndarray((0, 8)), [], cpp.mesh.GhostMode.none,
                                               cpp.mesh.Partitioner.scotch)

    V = FunctionSpace(mesh, space_type)
    dofmap = V.dofmap

    edges = {}
    faces = {}

    # Get coordinates of dofs and edges and check that they are the same
    # for each global dof number
    X = V.element.dof_reference_coordinates()
    coord_dofs = mesh.coordinate_dofs().entity_points()
    x_g = mesh.geometry.points
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    for cell_n, cell in enumerate(coord_dofs):
        dofs = dofmap.cell_dofs(cell_n)

        x_coord_new = np.zeros([8, 3])
        for v in range(8):
            x_coord_new[v] = x_g[coord_dofs[cell_n, v]]
        x = X.copy()
        cmap.push_forward(x, X, x_coord_new)

        edge_dofs_local = []
        for i in range(12):
            edge_dofs_local += list(dofmap.dof_layout.entity_dofs(1, i))
        edge_dofs = [dofs[i] for i in edge_dofs_local]
        for i, j in zip(edge_dofs, x[edge_dofs_local]):
            if i in edges:
                assert np.allclose(j, edges[i])
            else:
                edges[i] = j

        face_dofs_local = []
        for i in range(6):
            face_dofs_local += list(dofmap.dof_layout.entity_dofs(2, i))
        face_dofs = [dofs[i] for i in face_dofs_local]
        for i, j in zip(face_dofs, x[face_dofs_local]):
            if i in faces:
                assert np.allclose(j, faces[i])
            else:
                faces[i] = j


@skip_in_parallel
@pytest.mark.parametrize('cell_type', [CellType.triangle, CellType.tetrahedron,
                                       CellType.quadrilateral, CellType.hexahedron])
def test_facet_normals(cell_type):
    """Test that FacetNormal is outward facing"""
    if cell_type == CellType.triangle:
        points = np.array([[0., 0.], [1., 0.], [0., 1.]])
        num_facets = 3
    elif cell_type == CellType.tetrahedron:
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        num_facets = 4
    elif cell_type == CellType.quadrilateral:
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        num_facets = 4
    elif cell_type == CellType.hexahedron:
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [1., 1., 0.], [0., 0., 1.], [1., 0., 1.],
                           [0., 1., 1.], [1., 1., 1.]])
        num_facets = 6
    num_points = len(points)

    order = list(range(num_points))
    for count in range(10):
        # Randomly number the points and create the mest
        shuffle(order)
        ordered_points = np.zeros(points.shape)
        for i, j in enumerate(order):
            ordered_points[j] = points[i]
        cells = np.array([order])
        mesh = Mesh(MPI.comm_world, cell_type, ordered_points, cells,
                    [], cpp.mesh.GhostMode.none)
        mesh.geometry.coord_mapping = fem.create_coordinate_map(mesh)
        V = FunctionSpace(mesh, ("Lagrange", 2))
        normal = FacetNormal(mesh)

        # For each facet, check that the inner product of the normal and
        # the vector that has a positive normal component on only that facet
        # is positive
        for i in range(num_facets):
            v = Function(V)
            if cell_type == CellType.triangle:
                co = points[i]
                v.interpolate(lambda x: (x[0] * (x[0] / 2 - co[0])
                                         + x[1] * (x[1] / 2 - co[1])))
            elif cell_type == CellType.tetrahedron:
                co = points[i]
                v.interpolate(lambda x: (x[0] * (x[0] / 2 - co[0])
                                         + x[1] * (x[1] / 2 - co[1])
                                         + x[2] * (x[2] / 2 - co[2])))
            elif cell_type == CellType.quadrilateral:
                c = i // 2
                d = i % 2
                v.interpolate(lambda x: x[c] ** 2 / 2 - d * x[c])
            elif cell_type == CellType.hexahedron:
                c = i // 2
                d = i % 2
                v.interpolate(lambda x: x[c] ** 2 / 2 - d * x[c])

            a = inner(grad(v), normal) * ds
            result = fem.assemble_scalar(a)
            assert result > 0
