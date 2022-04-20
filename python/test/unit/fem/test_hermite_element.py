import basix
from basix.ufl_wrapper import BasixElement

import numpy as np
import pytest
import random

import ufl
from dolfinx.fem import (Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form,
                         locate_dofs_topological)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               set_bc)
from dolfinx.mesh import CellType, compute_boundary_facets, create_mesh
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dS, grad, inner, jump

from mpi4py import MPI
from petsc4py import PETSc


def two_unit_cells(cell_type, agree=False, random_order=True, return_order=False):
    if cell_type == CellType.interval:
        points = np.array([[0.], [1.], [-1.]])
        if agree:
            cells = [[0, 1], [2, 0]]
        else:
            cells = [[0, 1], [0, 2]]
    if cell_type == CellType.triangle:
        # Define equilateral triangles with area 1
        root = 3 ** 0.25  # 4th root of 3
        points = np.array([[0., 0.], [2 / root, 0.],
                           [1 / root, root], [1 / root, -root]])
        if agree:
            cells = [[0, 1, 2], [0, 3, 1]]
        else:
            cells = [[0, 1, 2], [1, 0, 3]]
    elif cell_type == CellType.tetrahedron:
        # Define regular tetrahedra with volume 1
        s = 2 ** 0.5 * 3 ** (1 / 3)  # side length
        points = np.array([[0., 0., 0.], [s, 0., 0.],
                           [s / 2, s * np.sqrt(3) / 2, 0.],
                           [s / 2, s / 2 / np.sqrt(3), s * np.sqrt(2 / 3)],
                           [s / 2, s / 2 / np.sqrt(3), -s * np.sqrt(2 / 3)]])
        if agree:
            cells = [[0, 1, 2, 3], [0, 1, 4, 2]]
        else:
            cells = [[0, 1, 2, 3], [0, 2, 1, 4]]
    elif cell_type == CellType.quadrilateral:
        # Define unit quadrilaterals (area 1)
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0., -1.], [1., -1.]])
        if agree:
            cells = [[0, 1, 2, 3], [4, 5, 0, 1]]
        else:
            cells = [[0, 1, 2, 3], [5, 1, 4, 0]]
    elif cell_type == CellType.hexahedron:
        # Define unit hexahedra (volume 1)
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                           [1., 1., 0.], [0., 0., 1.], [1., 0., 1.],
                           [0., 1., 1.], [1., 1., 1.], [0., 0., -1.],
                           [1., 0., -1.], [0., 1., -1.], [1., 1., -1.]])
        if agree:
            cells = [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 1, 2, 3]]
        else:
            cells = [[0, 1, 2, 3, 4, 5, 6, 7], [9, 11, 8, 10, 1, 3, 0, 2]]
    num_points = len(points)

    # Randomly number the points and create the mesh
    order = list(range(num_points))
    if random_order:
        random.shuffle(order)
    ordered_points = np.zeros(points.shape)
    for i, j in enumerate(order):
        ordered_points[j] = points[i]
    ordered_cells = np.array([[order[i] for i in c] for c in cells])

    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell_type.name, 1))
    mesh = create_mesh(MPI.COMM_WORLD, ordered_cells, ordered_points, domain)
    if return_order:
        return mesh, order
    return mesh


@pytest.mark.skip_in_parallel
def test_hermite_interval():
    e = basix.create_element(basix.ElementFamily.Hermite, basix.CellType.interval, 3)
    ufl_element = BasixElement(e)

    root = 3 ** 0.25
    pts = [[2 / root * i / 10, 0, 0] for i in range(11)]

    for i in range(3):
        mesh = two_unit_cells(CellType.interval)
        V = FunctionSpace(mesh, ufl_element)

        f = Function(V)
        f.vector[:] = np.random.rand(f.vector[:].shape[0])

        # Assert that the function is continuous between cells
        eval0 = f.eval(pts, [0 for p in pts])
        eval1 = f.eval(pts, [0 for p in pts])
        assert np.allclose(eval0, eval1)

        # Test that the jump in the derivative is 0
        for d in [0, 1]:
            djump = jump(f.dx(d))
            a = inner(djump, djump) * dS
            a = form(a)
            value = assemble_scalar(a)
            assert np.absolute(value) < 1e-14


@pytest.mark.skip_in_parallel
def test_hermite_triangle():
    e = basix.create_element(basix.ElementFamily.Hermite, basix.CellType.triangle, 3)
    ufl_element = BasixElement(e)

    root = 3 ** 0.25
    pts = [[2 / root * i / 10, 0, 0] for i in range(11)]

    for i in range(10):
        mesh = two_unit_cells(CellType.triangle)
        V = FunctionSpace(mesh, ufl_element)

        f = Function(V)
        f.vector[:] = np.random.rand(f.vector[:].shape[0])

        # Assert that the function is continuous between cells
        eval0 = f.eval(pts, [0 for p in pts])
        eval1 = f.eval(pts, [0 for p in pts])
        assert np.allclose(eval0, eval1)

        # Test that the jump in the derivative is 0
        for d in [0, 1]:
            djump = jump(f.dx(d))
            a = inner(djump, djump) * dS
            a = form(a)
            value = assemble_scalar(a)
            assert np.absolute(value) < 1e-14
