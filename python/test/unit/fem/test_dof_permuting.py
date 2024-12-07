# Copyright (C) 2009-2024 Garth N. Wells, Matthew W. Scroggs and Jorgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for dofmap construction"""

import random

from mpi4py import MPI

import numpy as np
import pytest

import ufl
from basix.ufl import element
from dolfinx import default_real_type
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from dolfinx.mesh import create_mesh, create_unit_cube


def randomly_ordered_mesh(cell_type):
    """Create a randomly ordered mesh to use in the test."""
    random.seed(6)

    if cell_type == "triangle" or cell_type == "quadrilateral":
        gdim = 2
    elif cell_type == "tetrahedron" or cell_type == "hexahedron":
        gdim = 3

    domain = ufl.Mesh(element("Lagrange", cell_type, 1, shape=(gdim,), dtype=default_real_type))
    # Create a mesh
    if MPI.COMM_WORLD.rank == 0:
        N = 6
        if cell_type == "triangle" or cell_type == "quadrilateral":
            temp_points = np.array([[x / 2, y / 2] for y in range(N) for x in range(N)])
        elif cell_type == "tetrahedron" or cell_type == "hexahedron":
            temp_points = np.array(
                [[x / 2, y / 2, z / 2] for z in range(N) for y in range(N) for x in range(N)]
            )

        order = [i for i, j in enumerate(temp_points)]
        random.shuffle(order)
        points = np.zeros(temp_points.shape, dtype=default_real_type)
        for i, j in enumerate(order):
            points[j] = temp_points[i]

        if cell_type == "triangle":
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

        elif cell_type == "quadrilateral":
            cells = []
            for x in range(N - 1):
                for y in range(N - 1):
                    a = N * y + x
                    cell = [order[i] for i in [a, a + 1, a + N, a + N + 1]]
                    cells.append(cell)

        elif cell_type == "tetrahedron":
            cells = []
            for x in range(N - 1):
                for y in range(N - 1):
                    for z in range(N - 1):
                        a = N**2 * z + N * y + x
                        for c in [
                            [a + N, a + N**2 + 1, a, a + 1],
                            [a + N, a + N**2 + 1, a + 1, a + N + 1],
                            [a + N, a + N**2 + 1, a + N + 1, a + N**2 + N + 1],
                            [a + N, a + N**2 + 1, a + N**2 + N + 1, a + N**2 + N],
                            [a + N, a + N**2 + 1, a + N**2 + N, a + N**2],
                            [a + N, a + N**2 + 1, a + N**2, a],
                        ]:
                            cell = [order[i] for i in c]
                            cells.append(cell)

        elif cell_type == "hexahedron":
            cells = []
            for x in range(N - 1):
                for y in range(N - 1):
                    for z in range(N - 1):
                        a = N**2 * z + N * y + x
                        cell = [
                            order[i]
                            for i in [
                                a,
                                a + 1,
                                a + N,
                                a + N + 1,
                                a + N**2,
                                a + 1 + N**2,
                                a + N + N**2,
                                a + N + 1 + N**2,
                            ]
                        ]
                        cells.append(cell)

        # On process 0, input mesh data and distribute to other
        # processes
        return create_mesh(MPI.COMM_WORLD, cells, points, domain)
    else:
        if cell_type == "triangle":
            return create_mesh(
                MPI.COMM_WORLD,
                np.ndarray((0, 3)),
                np.ndarray((0, 2), dtype=default_real_type),
                domain,
            )
        elif cell_type == "quadrilateral":
            return create_mesh(
                MPI.COMM_WORLD,
                np.ndarray((0, 4)),
                np.ndarray((0, 2), dtype=default_real_type),
                domain,
            )
        elif cell_type == "tetrahedron":
            return create_mesh(
                MPI.COMM_WORLD,
                np.ndarray((0, 4)),
                np.ndarray((0, 3), dtype=default_real_type),
                domain,
            )
        elif cell_type == "hexahedron":
            return create_mesh(
                MPI.COMM_WORLD,
                np.ndarray((0, 8)),
                np.ndarray((0, 3), dtype=default_real_type),
                domain,
            )


@pytest.mark.parametrize("space_type", [("P", 1), ("P", 2), ("P", 3), ("P", 4)])
@pytest.mark.parametrize("cell_type", ["triangle", "tetrahedron", "quadrilateral", "hexahedron"])
def test_dof_positions(cell_type, space_type):
    """Checks that dofs on shared triangle edges match up"""
    mesh = randomly_ordered_mesh(cell_type)

    if cell_type == "triangle":
        entities_per_cell = [3, 3, 1]
    elif cell_type == "quadrilateral":
        entities_per_cell = [4, 4, 1]
    elif cell_type == "tetrahedron":
        entities_per_cell = [4, 6, 4, 1]
    elif cell_type == "hexahedron":
        entities_per_cell = [8, 12, 6, 1]

    # Get coordinates of dofs and edges and check that they are the same
    # for each global dof number
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    cmap = mesh.geometry.cmap
    tdim = mesh.topology.dim

    V = functionspace(mesh, space_type)
    entities = {i: {} for i in range(1, tdim)}
    for cell in range(coord_dofs.shape[0]):
        # Push coordinates forward
        X = V.element.interpolation_points()
        xg = x_g[coord_dofs[cell], :tdim]
        x = cmap.push_forward(X, xg)

        dofs = V.dofmap.cell_dofs(cell)

        for entity_dim in range(1, tdim):
            entity_dofs_local = []
            for i in range(entities_per_cell[entity_dim]):
                entity_dofs_local += list(V.dofmap.dof_layout.entity_dofs(entity_dim, i))
            entity_dofs = [dofs[i] for i in entity_dofs_local]
            for i, j in zip(entity_dofs, x[entity_dofs_local]):
                if i in entities[entity_dim]:
                    assert np.allclose(j, entities[entity_dim][i], atol=1e-06)
                else:
                    entities[entity_dim][i] = j


def random_evaluation_mesh(cell_type):
    random.seed(6)

    if cell_type == "triangle" or cell_type == "quadrilateral":
        gdim = 2
    elif cell_type == "tetrahedron" or cell_type == "hexahedron":
        gdim = 3

    domain = ufl.Mesh(element("Lagrange", cell_type, 1, shape=(gdim,), dtype=default_real_type))
    if cell_type == "triangle":
        temp_points = np.array(
            [[-1.0, -1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=default_real_type
        )
        temp_cells = [[0, 1, 3], [1, 2, 3]]
    elif cell_type == "quadrilateral":
        temp_points = np.array(
            [[-1.0, -1.0], [0.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [2.0, 2.0]],
            dtype=default_real_type,
        )
        temp_cells = [[0, 1, 3, 4], [1, 2, 4, 5]]
    elif cell_type == "tetrahedron":
        temp_points = np.array(
            [[-1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=default_real_type,
        )
        temp_cells = [[0, 1, 3, 4], [1, 2, 3, 4]]
    elif cell_type == "hexahedron":
        temp_points = np.array(
            [
                [-1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [-1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [-1.0, 1.0, 2.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0],
            ],
            dtype=default_real_type,
        )
        temp_cells = [[0, 1, 3, 4, 6, 7, 9, 10], [1, 2, 4, 5, 7, 8, 10, 11]]

    order = [i for i, j in enumerate(temp_points)]
    random.shuffle(order)
    points = np.zeros(temp_points.shape, dtype=default_real_type)
    for i, j in enumerate(order):
        points[j] = temp_points[i]

    cells = []
    for cell in temp_cells:
        # Randomly number the cell
        if cell_type == "triangle":
            cell_order = list(range(3))
            random.shuffle(cell_order)
        elif cell_type == "quadrilateral":
            connections = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
            start = random.choice(range(4))
            cell_order = [start]
            for i in range(2):
                diff = (
                    random.choice([i for i in connections[start] if i not in cell_order])
                    - cell_order[0]
                )
                cell_order += [c + diff for c in cell_order]
        elif cell_type == "tetrahedron":
            cell_order = list(range(4))
            random.shuffle(cell_order)
        elif cell_type == "hexahedron":
            connections = {
                0: [1, 2, 4],
                1: [0, 3, 5],
                2: [0, 3, 6],
                3: [1, 2, 7],
                4: [0, 5, 6],
                5: [1, 4, 7],
                6: [2, 4, 7],
                7: [3, 5, 6],
            }
            start = random.choice(range(8))
            cell_order = [start]
            for i in range(3):
                diff = (
                    random.choice([i for i in connections[start] if i not in cell_order])
                    - cell_order[0]
                )
                cell_order += [c + diff for c in cell_order]

        cells.append([order[cell[i]] for i in cell_order])
    return create_mesh(MPI.COMM_WORLD, np.array(cells), points, domain)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    "cell_type,space_type",
    [(c, s) for c in ["triangle", "tetrahedron"] for s in ["P", "N1curl", "RT", "BDM", "N2curl"]]
    + [("quadrilateral", s) for s in ["Q", "S", "RTCE", "RTCF", "BDMCE", "BDMCF"]]
    + [("hexahedron", s) for s in ["Q", "S", "NCE", "NCF", "AAE", "AAF"]],
)
@pytest.mark.parametrize("space_order", range(1, 4))
def test_evaluation(cell_type, space_type, space_order):
    if cell_type == "hexahedron" and space_order > 3:
        pytest.skip("Skipping expensive test on hexahedron")

    random.seed(4)
    for repeat in range(10):
        mesh = random_evaluation_mesh(cell_type)
        V = functionspace(mesh, (space_type, space_order))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        N = 5
        if cell_type == "tetrahedron":
            eval_points = np.array(
                [[0.0, i / N, j / N] for i in range(N + 1) for j in range(N + 1 - i)],
                dtype=default_real_type,
            )
        elif cell_type == "hexahedron":
            eval_points = np.array(
                [[0.0, i / N, j / N] for i in range(N + 1) for j in range(N + 1)],
                dtype=default_real_type,
            )
        else:
            eval_points = np.array(
                [[0.0, i / N, 0.0] for i in range(N + 1)], dtype=default_real_type
            )

        for d in dofs:
            v = Function(V)
            v.x.array[:] = [1 if i == d else 0 for i in range(v.x.index_map.size_local)]
            values0 = v.eval(eval_points, [0 for i in eval_points])
            values1 = v.eval(eval_points, [1 for i in eval_points])
            if len(eval_points) == 1:
                values0 = [values0]
                values1 = [values1]
            if space_type in ["RT", "BDM", "RTCF", "NCF", "BDMCF", "AAF"]:
                # Hdiv
                for i, j in zip(values0, values1):
                    assert np.isclose(i[0], j[0], rtol=1.0e-5, atol=1.0e-3)
            elif space_type in ["N1curl", "N2curl", "RTCE", "NCE", "BDMCE", "AAE"]:
                # Hcurl
                for i, j in zip(values0, values1):
                    assert np.allclose(i[1:], j[1:], rtol=1.0e-4, atol=1.0e-2)
            else:
                assert np.allclose(values0, values1, rtol=1.0e-6, atol=1.0e-4)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize(
    "cell_type,space_type",
    [(c, s) for c in ["triangle", "tetrahedron"] for s in ["P", "N1curl", "RT", "BDM", "N2curl"]]
    + [("quadrilateral", s) for s in ["Q", "S", "RTCE", "RTCF", "BDMCE", "BDMCF"]]
    + [("hexahedron", s) for s in ["Q", "S", "NCE", "NCF", "AAE", "AAF"]],
)
@pytest.mark.parametrize("space_order", range(1, 4))
def test_integral(cell_type, space_type, space_order):
    if cell_type == "hexahedron" and space_order >= 3:
        pytest.skip("Skipping expensive test on hexahedron")

    random.seed(4)
    for repeat in range(10):
        mesh = random_evaluation_mesh(cell_type)
        V = functionspace(mesh, (space_type, space_order))
        gdim = mesh.geometry.dim
        Vvec = functionspace(mesh, ("P", 1, (gdim,)))
        dofs = [i for i in V.dofmap.cell_dofs(0) if i in V.dofmap.cell_dofs(1)]

        tdim = mesh.topology.dim
        for d in dofs:
            v = Function(V)
            v.x.array[:] = [1 if i == d else 0 for i, _ in enumerate(v.x.array[:])]
            if space_type in ["RT", "BDM", "RTCF", "NCF", "BDMCF", "AAF"]:
                # Hdiv
                def normal(x):
                    values = np.zeros((tdim, x.shape[1]))
                    values[0] = [1 for i in values[0]]
                    return values

                n = Function(Vvec)
                n.interpolate(normal)
                _form = ufl.inner(ufl.jump(v), n) * ufl.dS
            elif space_type in ["N1curl", "N2curl", "RTCE", "NCE", "BDMCE", "AAE"]:
                # Hcurl
                def tangent(x):
                    values = np.zeros((tdim, x.shape[1]))
                    values[1] = [1 for i in values[1]]
                    return values

                t = Function(Vvec)
                t.interpolate(tangent)
                _form = ufl.inner(ufl.jump(v), t) * ufl.dS
                if tdim == 3:

                    def tangent2(x):
                        values = np.zeros((3, x.shape[1]))
                        values[2] = [1 for i in values[2]]
                        return values

                    t2 = Function(Vvec)
                    t2.interpolate(tangent2)
                    _form += ufl.inner(ufl.jump(v), t2) * ufl.dS
            else:
                _form = ufl.jump(v) * ufl.dS

            value = assemble_scalar(form(_form))
            assert np.isclose(value, 0.0, rtol=1.0e-6, atol=1.0e-6)


@pytest.mark.parametrize(
    "data_types",
    [
        (np.float32, np.float32),
        (np.float64, np.float64),
        (np.float32, np.complex64),
        (np.float64, np.complex128),
    ],
)
@pytest.mark.parametrize("space_order", range(3, 5))
def test_permutation_wrappers(space_order, data_types):
    s_type, d_type = data_types
    domain = create_unit_cube(MPI.COMM_WORLD, 5, 3, 4, dtype=s_type)
    V = functionspace(domain, ("N1curl", space_order))
    u = Function(V, name="u", dtype=d_type)
    u.interpolate(lambda x: np.array([x[0], x[1], x[2]]))

    arr = u.x.array[V.dofmap.list]

    domain.topology.create_entity_permutations()
    cell_perm = domain.topology.get_cell_permutation_info()
    org_data = arr.copy()
    V.element.Tt_apply(arr.reshape(-1), cell_perm, 1)
    V.element.Tt_inv_apply(arr.reshape(-1), cell_perm, 1)
    eps = 100 * np.finfo(s_type).eps
    np.testing.assert_allclose(org_data.reshape(-1), arr.reshape(-1), atol=eps)
