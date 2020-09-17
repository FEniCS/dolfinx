# Copyright (C) 2009-2020 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for dofmap construction"""

from random import shuffle, choice

import numpy as np
import pytest
import ufl
from dolfinx import FunctionSpace, fem, Function, VectorFunctionSpace
from dolfinx.mesh import create_mesh
from mpi4py import MPI
from dolfinx_utils.test.skips import skip_in_parallel


@pytest.mark.parametrize('space_type', [
    ("P", 1), ("P", 2), ("P", 3), ("P", 4),
    ("N1curl", 1),
    ("RT", 1), ("RT", 2), ("RT", 3), ("RT", 4),
    ("BDM", 1),
    ("N2curl", 1),
])
def test_triangle_dof_positions(space_type):
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
def test_tetrahedron_dof_positions(space_type):
    """Checks that dofs on shared tetrahedron edges and faces match up"""

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "tetrahedron", 1))
    if MPI.COMM_WORLD.rank == 0:
        N = 6
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


@pytest.mark.parametrize('space_type', [
    ("P", 1), ("P", 2), ("P", 3), ("P", 4)
])
def test_quadrilateral_dof_positions(space_type):
    """Checks that dofs on shared quadrilateral edges match up"""

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    if MPI.COMM_WORLD.rank == 0:
        # Create a quadrilateral mesh
        N = 6
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


@pytest.mark.parametrize('space_type', [
    ("P", 1), ("P", 2), ("P", 3), ("P", 4),
])
def test_hexahedron_dof_positions(space_type):
    """Checks that dofs on shared hexahedron edges match up"""

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "hexahedron", 1))
    if MPI.COMM_WORLD.rank == 0:
        N = 6
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


@skip_in_parallel
@pytest.mark.parametrize('space_type', ["P", "N1curl", "RT", "BDM", "N2curl"])
@pytest.mark.parametrize('space_order', range(1, 4))
def test_triangle_evaluation(space_type, space_order):
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    temp_points = np.array([[-1., -1.], [0., 0.], [1., 0.], [0., 1.]])

    for repeat in range(10):
        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        cells = []
        for cell in [[0, 1, 3], [1, 2, 3]]:
            # Randomly number the cell
            cell_order = list(range(3))
            shuffle(cell_order)
            cells.append([order[cell[i]] for i in cell_order])
        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = FunctionSpace(mesh, (space_type, space_order))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        N = 6
        eval_points = np.array([[0., i / N, 0.] for i in range(N + 1)])
        for d in dofs:
            v = Function(V)
            v.vector[:] = [1 if i == d else 0 for i in range(V.dim)]
            values0 = v.eval(eval_points, [0 for i in eval_points])
            values1 = v.eval(eval_points, [1 for i in eval_points])
            if len(eval_points) == 1:
                values0 = [values0]
                values1 = [values1]
            if space_type in ["RT", "BDM"]:
                # Hdiv
                for i, j in zip(values0, values1):
                    assert np.isclose(i[0], j[0])
            elif space_type in ["N1curl", "N2curl"]:
                # Hcurl
                for i, j in zip(values0, values1):
                    assert np.allclose(i[1], j[1])
            else:
                assert np.allclose(values0, values1)


@skip_in_parallel
@pytest.mark.parametrize('space_type', ["Q", "RTCE", "RTCF"])
@pytest.mark.parametrize('space_order', range(1, 4))
def test_quadrilateral_evaluation(space_type, space_order):
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    temp_points = np.array([[-1., -1.], [0., 0.], [1., 0.],
                            [-1., 1.], [0., 1.], [2., 2.]])

    for repeat in range(10):
        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        connections = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}

        cells = []
        for cell in [[0, 1, 3, 4], [1, 2, 4, 5]]:
            # Randomly number the cell
            start = choice(range(4))
            cell_order = [start]
            for i in range(2):
                diff = choice([i for i in connections[start] if i not in cell_order]) - cell_order[0]
                cell_order += [c + diff for c in cell_order]
            cells.append([order[cell[i]] for i in cell_order])

        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = FunctionSpace(mesh, (space_type, space_order))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        N = 6
        eval_points = np.array([[0., i / N, 0.] for i in range(N + 1)])
        for d in dofs:
            v = Function(V)
            v.vector[:] = [1 if i == d else 0 for i in range(V.dim)]
            values0 = v.eval(eval_points, [0 for i in eval_points])
            values1 = v.eval(eval_points, [1 for i in eval_points])
            if len(eval_points) == 1:
                values0 = [values0]
                values1 = [values1]
            if space_type == "RTCF":
                # Hdiv
                for i, j in zip(values0, values1):
                    assert np.isclose(i[0], j[0])
            elif space_type == "RTCE":
                # Hcurl
                for i, j in zip(values0, values1):
                    assert np.allclose(i[1], j[1])
            else:
                assert np.allclose(values0, values1)


@skip_in_parallel
@pytest.mark.parametrize('space_type', ["P", "N1curl", "RT", "BDM", "N2curl"])
@pytest.mark.parametrize('space_order', range(1, 4))
def test_tetrahedron_evaluation(space_type, space_order):
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "tetrahedron", 1))
    temp_points = np.array([[-1., 0., -1.], [0., 0., 0.], [1., 0., 1.],
                            [0., 1., 0.], [0., 0., 1.]])

    for repeat in range(10):
        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        cells = []
        for cell in [[0, 1, 3, 4], [1, 2, 3, 4]]:
            # Randomly number the cell
            cell_order = list(range(4))
            shuffle(cell_order)
            cells.append([order[cell[i]] for i in cell_order])

        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = FunctionSpace(mesh, (space_type, space_order))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        N = 6
        eval_points = np.array([[0., i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)])
        for d in dofs:
            v = Function(V)
            v.vector[:] = [1 if i == d else 0 for i in range(V.dim)]
            values0 = v.eval(eval_points, [0 for i in eval_points])
            values1 = v.eval(eval_points, [1 for i in eval_points])
            if len(eval_points) == 1:
                values0 = [values0]
                values1 = [values1]
            if space_type in ["RT", "BDM"]:
                # Hdiv
                for i, j in zip(values0, values1):
                    assert np.isclose(i[0], j[0])
            elif space_type in ["N1curl", "N2curl"]:
                # Hcurl
                for i, j in zip(values0, values1):
                    assert np.allclose(i[1:], j[1:])
            else:
                assert np.allclose(values0, values1)


@skip_in_parallel
@pytest.mark.parametrize('space_type', ["Q", "NCE", "NCF"])
@pytest.mark.parametrize('space_order', range(1, 4))
def test_hexahedron_evaluation(space_type, space_order):

    if space_type == "NCF" and space_order >= 3:
        print("Eval in this space not supported yet")
        return
    if space_type == "NCE" and space_order >= 2:
        print("Eval in this space not supported yet")
        return

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "hexahedron", 1))
    temp_points = np.array([[-1., 0., -1.], [0., 0., 0.], [1., 0., 1.],
                            [-1., 1., 1.], [0., 1., 0.], [1., 1., 1.],
                            [-1., 0., 0.], [0., 0., 1.], [1., 0., 2.],
                            [-1., 1., 2.], [0., 1., 1.], [1., 1., 2.]])

    for repeat in range(10):
        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        connections = {0: [1, 2, 4], 1: [0, 3, 5], 2: [0, 3, 6], 3: [1, 2, 7],
                       4: [0, 5, 6], 5: [1, 4, 7], 6: [2, 4, 7], 7: [3, 5, 6]}

        cells = []
        for cell in [[0, 1, 3, 4, 6, 7, 9, 10], [1, 2, 4, 5, 7, 8, 10, 11]]:
            # Randomly number the cell
            start = choice(range(8))
            cell_order = [start]
            for i in range(3):
                diff = choice([i for i in connections[start] if i not in cell_order]) - cell_order[0]
                cell_order += [c + diff for c in cell_order]
            cells.append([order[cell[i]] for i in cell_order])

        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = FunctionSpace(mesh, (space_type, space_order))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        N = 6
        eval_points = np.array([[0., i / N, j / N] for i in range(N + 1) for j in range(N + 1)])
        for d in dofs:
            v = Function(V)
            v.vector[:] = [1 if i == d else 0 for i in range(V.dim)]
            values0 = v.eval(eval_points, [0 for i in eval_points])
            values1 = v.eval(eval_points, [1 for i in eval_points])
            if len(eval_points) == 1:
                values0 = [values0]
                values1 = [values1]
            if space_type == "NCF":
                # Hdiv
                for i, j in zip(values0, values1):
                    assert np.isclose(i[0], j[0])
            elif space_type == "NCE":
                # Hcurl
                for i, j in zip(values0, values1):
                    assert np.allclose(i[1:], j[1:])
            else:
                assert np.allclose(values0, values1)


# TODO: Fix jump integrals in FFC by passing in full info for both cells, then re-enable these tests
@skip_in_parallel
@pytest.mark.parametrize('space_type', ["P", "N1curl", "RT", "BDM", "N2curl"])
@pytest.mark.parametrize('space_order', range(1, 4))
def xtest_triangle_integral(space_type, space_order):
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    temp_points = np.array([[-1., -1.], [0., 0.], [1., 0.], [0., 1.]])

    for repeat in range(10):
        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        cells = []
        for cell in [[0, 1, 3], [1, 2, 3]]:
            # Randomly number the cell
            cell_order = list(range(3))
            shuffle(cell_order)
            cells.append([order[cell[i]] for i in cell_order])

        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = FunctionSpace(mesh, (space_type, space_order))
        Vvec = VectorFunctionSpace(mesh, ("P", 1))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        for d in dofs:
            v = Function(V)
            v.vector[:] = [1 if i == d else 0 for i in range(V.dim)]
            if space_type in ["RT", "BDM"]:
                # Hdiv
                def normal(x):
                    values = np.zeros((2, x.shape[1]))
                    values[0] = [1 for i in values[0]]
                    return values

                n = Function(Vvec)
                n.interpolate(normal)
                form = ufl.inner(ufl.jump(v), n) * ufl.dS
            elif space_type in ["N1curl", "N2curl"]:
                # Hcurl
                def tangent(x):
                    values = np.zeros((2, x.shape[1]))
                    values[1] = [1 for i in values[1]]
                    return values

                t = Function(Vvec)
                t.interpolate(tangent)
                form = ufl.inner(ufl.jump(v), t) * ufl.dS
            else:
                form = ufl.jump(v) * ufl.dS

            value = fem.assemble_scalar(form)
            assert np.isclose(value, 0)


# TODO: Fix jump integrals in FFC by passing in full info for both cells, then re-enable these tests
@skip_in_parallel
@pytest.mark.parametrize('space_type', ["Q", "RTCE", "RTCF"])
@pytest.mark.parametrize('space_order', range(1, 4))
def xtest_quadrilateral_integral(space_type, space_order):
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "quadrilateral", 1))
    temp_points = np.array([[-1., -1.], [0., 0.], [1., 0.],
                            [-1., 1.], [0., 1.], [2., 2.]])

    for repeat in range(10):
        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        connections = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}

        cells = []
        for cell in [[0, 1, 3, 4], [1, 2, 4, 5]]:
            # Randomly number the cell
            start = choice(range(4))
            cell_order = [start]
            for i in range(2):
                diff = choice([i for i in connections[start] if i not in cell_order]) - cell_order[0]
                cell_order += [c + diff for c in cell_order]
            cells.append([order[cell[i]] for i in cell_order])

        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = FunctionSpace(mesh, (space_type, space_order))
        Vvec = VectorFunctionSpace(mesh, ("P", 1))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        for d in dofs:
            v = Function(V)
            v.vector[:] = [1 if i == d else 0 for i in range(V.dim)]
            if space_type in ["RTCF"]:
                # Hdiv
                def normal(x):
                    values = np.zeros((2, x.shape[1]))
                    values[0] = [1 for i in values[0]]
                    return values

                n = Function(Vvec)
                n.interpolate(normal)
                form = ufl.inner(ufl.jump(v), n) * ufl.dS
            elif space_type in ["RTCE"]:
                # Hcurl
                def tangent(x):
                    values = np.zeros((2, x.shape[1]))
                    values[1] = [1 for i in values[1]]
                    return values

                t = Function(Vvec)
                t.interpolate(tangent)
                form = ufl.inner(ufl.jump(v), t) * ufl.dS
            else:
                form = ufl.jump(v) * ufl.dS

            value = fem.assemble_scalar(form)
            assert np.isclose(value, 0)


# TODO: Fix jump integrals in FFC by passing in full info for both cells, then re-enable these tests
@skip_in_parallel
@pytest.mark.parametrize('space_type', ["P", "N1curl", "RT", "BDM", "N2curl"])
@pytest.mark.parametrize('space_order', range(1, 4))
def xtest_tetrahedron_integral(space_type, space_order):
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "tetrahedron", 1))
    temp_points = np.array([[-1., 0., -1.], [0., 0., 0.], [1., 0., 1.],
                            [0., 1., 0.], [0., 0., 1.]])

    for repeat in range(10):
        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        cells = []
        for cell in [[0, 1, 3, 4], [1, 2, 3, 4]]:
            # Randomly number the cell
            cell_order = list(range(4))
            shuffle(cell_order)
            cells.append([order[cell[i]] for i in cell_order])

        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = FunctionSpace(mesh, (space_type, space_order))
        Vvec = VectorFunctionSpace(mesh, ("P", 1))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        for d in dofs:
            v = Function(V)
            v.vector[:] = [1 if i == d else 0 for i in range(V.dim)]
            if space_type in ["RT", "BDM"]:
                # Hdiv
                def normal(x):
                    values = np.zeros((3, x.shape[1]))
                    values[0] = [1 for i in values[0]]
                    return values

                n = Function(Vvec)
                n.interpolate(normal)
                form = ufl.inner(ufl.jump(v), n) * ufl.dS
            elif space_type in ["N1curl", "N2curl"]:
                # Hcurl
                def tangent1(x):
                    values = np.zeros((3, x.shape[1]))
                    values[1] = [1 for i in values[1]]
                    return values

                def tangent2(x):
                    values = np.zeros((3, x.shape[1]))
                    values[2] = [1 for i in values[2]]
                    return values

                t1 = Function(Vvec)
                t1.interpolate(tangent1)
                t2 = Function(Vvec)
                t2.interpolate(tangent1)
                form = ufl.inner(ufl.jump(v), t1) * ufl.dS
                form += ufl.inner(ufl.jump(v), t2) * ufl.dS
            else:
                form = ufl.jump(v) * ufl.dS

            value = fem.assemble_scalar(form)
            assert np.isclose(value, 0)


# TODO: Fix jump integrals in FFC by passing in full info for both cells, then re-enable these tests
@skip_in_parallel
@pytest.mark.parametrize('space_type', ["Q", "NCE", "NCF"])
@pytest.mark.parametrize('space_order', range(1, 4))
def xtest_hexahedron_integral(space_type, space_order):
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "hexahedron", 1))
    temp_points = np.array([[-1., 0., -1.], [0., 0., 0.], [1., 0., 1.],
                            [-1., 1., 1.], [0., 1., 0.], [1., 1., 1.],
                            [-1., 0., 0.], [0., 0., 1.], [1., 0., 2.],
                            [-1., 1., 2.], [0., 1., 1.], [1., 1., 2.]])

    for repeat in range(10):
        order = [i for i, j in enumerate(temp_points)]
        shuffle(order)
        points = np.zeros(temp_points.shape)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        connections = {0: [1, 2, 4], 1: [0, 3, 5], 2: [0, 3, 6], 3: [1, 2, 7],
                       4: [0, 5, 6], 5: [1, 4, 7], 6: [2, 4, 7], 7: [3, 5, 6]}

        cells = []
        for cell in [[0, 1, 3, 4, 6, 7, 9, 10], [1, 2, 4, 5, 7, 8, 10, 11]]:
            # Randomly number the cell
            start = choice(range(8))
            cell_order = [start]
            for i in range(3):
                diff = choice([i for i in connections[start] if i not in cell_order]) - cell_order[0]
                cell_order += [c + diff for c in cell_order]
            cells.append([order[cell[i]] for i in cell_order])

        mesh = create_mesh(MPI.COMM_WORLD, cells, points, domain)
        V = FunctionSpace(mesh, (space_type, space_order))
        Vvec = VectorFunctionSpace(mesh, ("P", 1))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        for d in dofs:
            v = Function(V)
            v.vector[:] = [1 if i == d else 0 for i in range(V.dim)]
            if space_type in ["NCF"]:
                # Hdiv
                def normal(x):
                    values = np.zeros((3, x.shape[1]))
                    values[0] = [1 for i in values[0]]
                    return values

                n = Function(Vvec)
                n.interpolate(normal)
                form = ufl.inner(ufl.jump(v), n) * ufl.dS
            elif space_type in ["NCE"]:
                # Hcurl
                def tangent1(x):
                    values = np.zeros((3, x.shape[1]))
                    values[1] = [1 for i in values[1]]
                    return values

                def tangent2(x):
                    values = np.zeros((3, x.shape[1]))
                    values[2] = [1 for i in values[2]]
                    return values

                t1 = Function(Vvec)
                t1.interpolate(tangent1)
                t2 = Function(Vvec)
                t2.interpolate(tangent1)
                form = ufl.inner(ufl.jump(v), t1) * ufl.dS
                form += ufl.inner(ufl.jump(v), t2) * ufl.dS
            else:
                form = ufl.jump(v) * ufl.dS

            value = fem.assemble_scalar(form)
            assert np.isclose(value, 0)
