# Copyright (C) 2009-2019 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for dofmap construction"""

from random import shuffle

import numpy as np
import pytest
from mpi4py import MPI

import ufl
from dolfinx import FunctionSpace, fem
from dolfinx.mesh import create_mesh

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

    # Create a mesh
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    if MPI.COMM_WORLD.rank == 0:
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
        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    else:
        mesh = create_mesh(MPI.COMM_WORLD, np.ndarray((0, 3)), np.ndarray((0, 2)), domain)

    V = FunctionSpace(mesh, space_type)
    dofmap = V.dofmap

    edges = {}

    # Get coordinates of dofs and edges and check that they are the same
    # for each global dof number
    X = V.element.dof_reference_coordinates()
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    for cell_n in range(coord_dofs.num_nodes):
        dofs = dofmap.cell_dofs(cell_n)

        x_coord_new = np.zeros([3, 2])
        for v in range(3):
            x_coord_new[v] = x_g[coord_dofs.links(cell_n)[v], :2]
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

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "tetrahedron", 1))
    if MPI.COMM_WORLD.rank == 0:
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
        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    else:
        mesh = create_mesh(MPI.COMM_WORLD, np.ndarray((0, 4)), np.ndarray((0, 3)), domain)

    V = FunctionSpace(mesh, space_type)
    dofmap = V.dofmap

    edges = {}
    faces = {}

    # Get coordinates of dofs and edges and check that they are the same
    # for each global dof number
    X = V.element.dof_reference_coordinates()
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    for cell_n in range(coord_dofs.num_nodes):
        dofs = dofmap.cell_dofs(cell_n)

        x_coord_new = np.zeros([4, 3])
        for v in range(4):
            x_coord_new[v] = x_g[coord_dofs.links(cell_n)[v]]
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


@pytest.mark.parametrize('space_type', [("P", 1), ("P", 2), ("P", 3), ("P", 4), ])
def test_quadrilateral_dof_ordering(space_type):
    """Checks that dofs on shared quadrilateral edges match up"""

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    if MPI.COMM_WORLD.rank == 0:
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
        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    else:
        mesh = create_mesh(MPI.COMM_WORLD, np.ndarray((0, 4)), np.ndarray((0, 2)), domain)

    V = FunctionSpace(mesh, space_type)
    dofmap = V.dofmap

    edges = {}

    # Get coordinates of dofs and edges and check that they are the same
    # for each global dof number
    X = V.element.dof_reference_coordinates()
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    for cell_n in range(coord_dofs.num_nodes):
        dofs = dofmap.cell_dofs(cell_n)

        x_coord_new = np.zeros([4, 2])
        for v in range(4):
            x_coord_new[v] = x_g[coord_dofs.links(cell_n)[v], :2]
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


@pytest.mark.parametrize('space_type', [("P", 1), ("P", 2), ("P", 3), ("P", 4), ])
def test_hexahedron_dof_ordering(space_type):
    """Checks that dofs on shared hexahedron edges match up"""

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "hexahedron", 1))
    if MPI.COMM_WORLD.rank == 0:
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
        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
    else:
        mesh = create_mesh(MPI.COMM_WORLD, np.ndarray((0, 8)), np.ndarray((0, 3)), domain)

    V = FunctionSpace(mesh, space_type)
    dofmap = V.dofmap

    edges = {}
    faces = {}

    # Get coordinates of dofs and edges and check that they are the same
    # for each global dof number
    X = V.element.dof_reference_coordinates()
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cmap = fem.create_coordinate_map(mesh.ufl_domain())
    for cell_n in range(coord_dofs.num_nodes):
        dofs = dofmap.cell_dofs(cell_n)

        x_coord_new = np.zeros([8, 3])
        for v in range(8):
            x_coord_new[v] = x_g[coord_dofs.links(cell_n)[v]]
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
