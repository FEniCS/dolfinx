# Copyright (C) 2019 Michal Habera
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


import cffi
import dolfinx
import numba
import numpy as np
import ufl
import basix
from mpi4py import MPI
from petsc4py import PETSc


def test_rank0():
    """Test evaluation of UFL expression.

    This test evaluates gradient of P2 function at vertices of reference
    triangle. Because these points coincide with positions of point evaluation
    degrees-of-freedom of vector P1 space, values could be used to interpolate
    the expression into this space.

    This test also shows simple Numba assembler which accepts the donor P2
    function ``f`` as a coefficient and tabulates vector P1 function into
    tensor ``b``.

    For a donor function f(x, y) = x^2 + 2*y^2 result is compared with the
    exact gradient grad f(x, y) = [2*x, 4*y].
    """
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    P2 = dolfinx.FunctionSpace(mesh, ("P", 2))
    vP1 = dolfinx.VectorFunctionSpace(mesh, ("P", 1))

    f = dolfinx.Function(P2)

    def expr1(x):
        return x[0] ** 2 + 2.0 * x[1] ** 2

    f.interpolate(expr1)

    ufl_expr = ufl.grad(f)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    compiled_expr = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), (ufl_expr, points))

    ffi = cffi.FFI()

    @numba.njit
    def assemble_expression(b, kernel, mesh, dofmap, coeff, coeff_dofmap):
        pos, x_dofmap, x = mesh
        geometry = np.zeros((3, 2))
        w = np.zeros(6, dtype=PETSc.ScalarType)
        constants = np.zeros(1, dtype=PETSc.ScalarType)
        b_local = np.zeros(6, dtype=PETSc.ScalarType)

        for i, cell in enumerate(pos[:-1]):
            num_vertices = pos[i + 1] - pos[i]
            c = x_dofmap[cell:cell + num_vertices]
            for j in range(3):
                for k in range(2):
                    geometry[j, k] = x[c[j], k]

            for j in range(6):
                w[j] = coeff[coeff_dofmap[i * 6 + j]]

            b_local.fill(0.0)
            kernel(ffi.from_buffer(b_local),
                   ffi.from_buffer(w),
                   ffi.from_buffer(constants),
                   ffi.from_buffer(geometry))
            for j in range(3):
                for k in range(2):
                    b[2 * dofmap[i * 3 + j] + k] = b_local[2 * j + k]

    # Prepare mesh and dofmap data
    pos = mesh.geometry.dofmap.offsets
    x_dofs = mesh.geometry.dofmap.array
    x = mesh.geometry.x
    coeff_dofmap = P2.dofmap.list.array
    dofmap = vP1.dofmap.list.array

    # Data structure for the result
    b = dolfinx.Function(vP1)

    assemble_expression(b.vector.array, compiled_expr.tabulate_expression,
                        (pos, x_dofs, x), dofmap, f.vector.array, coeff_dofmap)

    def grad_expr1(x):
        values = np.empty((2, x.shape[1]))
        values[0] = 2.0 * x[0]
        values[1] = 4.0 * x[1]

        return values

    b2 = dolfinx.Function(vP1)
    b2.interpolate(grad_expr1)

    assert np.isclose((b2.vector - b.vector).norm(), 0.0)


def test_simple_evaluation():
    """Test evaluation of UFL Expression.

    This test evaluates a UFL Expression on cells of the mesh and compares the
    result with an analytical expression.

    For a function f(x, y) = 3*(x^2 + 2*y^2) the result is compared with the
    exact gradient:

        grad f(x, y) = 3*[2*x, 4*y].

    (x^2 + 2*y^2) is first interpolated into a P2 finite element space. The
    scaling by a constant factor of 3 and the gradient is calculated using code
    generated by FFCX. The analytical solution is found by evaluating the
    spatial coordinates as an Expression using UFL/FFCX and passing the result
    to a numpy function that calculates the exact gradient.
    """
    mesh = dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
    P2 = dolfinx.FunctionSpace(mesh, ("P", 2))

    # NOTE: The scaling by a constant factor of 3.0 to get f(x, y) is
    # implemented within the UFL Expression. This is to check that the
    # Constants are being set up correctly.
    def exact_expr(x):
        return x[0] ** 2 + 2.0 * x[1] ** 2

    # Unused, but remains for clarity.
    def f(x):
        return 3 * (x[0] ** 2 + 2.0 * x[1] ** 2)

    def exact_grad_f(x):
        values = np.zeros_like(x)
        values[:, 0::2] = 2 * x[:, 0::2]
        values[:, 1::2] = 4 * x[:, 1::2]
        values *= 3.0
        return values

    expr = dolfinx.Function(P2)
    expr.interpolate(exact_expr)

    ufl_grad_f = dolfinx.Constant(mesh, 3.0) * ufl.grad(expr)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    grad_f_expr = dolfinx.Expression(ufl_grad_f, points)
    assert grad_f_expr.num_points == points.shape[0]
    assert grad_f_expr.value_size == 2

    # NOTE: Cell numbering is process local.
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    grad_f_evaluated = grad_f_expr.eval(cells)
    assert grad_f_evaluated.shape[0] == cells.shape[0]
    assert grad_f_evaluated.shape[1] == grad_f_expr.value_size * grad_f_expr.num_points

    # Evaluate points in global space
    ufl_x = ufl.SpatialCoordinate(mesh)
    x_expr = dolfinx.Expression(ufl_x, points)
    assert x_expr.num_points == points.shape[0]
    assert x_expr.value_size == 2
    x_evaluated = x_expr.eval(cells)
    assert x_evaluated.shape[0] == cells.shape[0]
    assert x_evaluated.shape[1] == x_expr.num_points * x_expr.value_size

    # Evaluate exact gradient using global points
    grad_f_exact = exact_grad_f(x_evaluated)

    assert(np.allclose(grad_f_evaluated, grad_f_exact))


def test_assembly_into_quadrature_function():
    """Test assembly into a Quadrature function.

    This test evaluates a UFL Expression into a Quadrature function space by
    evaluating the Expression on all cells of the mesh, and then inserting the
    evaluated values into a PETSc Vector constructed from a matching Quadrature
    function space.

    Concretely, we consider the evaluation of:

        e = B*(K(T)))**2 * grad(T)

    where

        K = 1/(A + B*T)

    where A and B are Constants and T is a Coefficient on a P2 finite element
    space with T = x + 2*y.

    The result is compared with interpolating the analytical expression of e
    directly into the Quadrature space.

    In parallel, each process evaluates the Expression on both local cells and
    ghost cells so that no parallel communication is required after insertion
    into the vector.
    """
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 6)

    quadrature_degree = 2
    quadrature_points = basix.make_quadrature(basix.CellType.triangle, quadrature_degree)
    Q_element = ufl.VectorElement("Quadrature", ufl.triangle, quadrature_degree, quad_scheme="default")
    Q = dolfinx.FunctionSpace(mesh, Q_element)

    def T_exact(x):
        return x[0] + 2.0 * x[1]

    P2 = dolfinx.FunctionSpace(mesh, ("P", 2))
    T = dolfinx.Function(P2)
    T.interpolate(T_exact)
    A = dolfinx.Constant(mesh, 1.0)
    B = dolfinx.Constant(mesh, 2.0)

    K = 1.0 / (A + B * T)
    e = B * K**2 * ufl.grad(T)

    e_expr = dolfinx.Expression(e, quadrature_points)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    e_eval = e_expr.eval(cells)

    # Assemble into Function
    e_Q = dolfinx.Function(Q)
    with e_Q.vector.localForm() as e_Q_local:
        e_Q_local.setBlockSize(e_Q.function_space.dofmap.bs)
        e_Q_local.setValuesBlocked(Q.dofmap.list.array, e_eval, addv=PETSc.InsertMode.INSERT)

    def e_exact(x):
        T = x[0] + 2.0 * x[1]
        K = 1.0 / (A.value + B.value * T)

        grad_T = np.zeros((2, x.shape[1]))
        grad_T[0, :] = 1.0
        grad_T[1, :] = 2.0

        e = B.value * K**2 * grad_T
        return e

    e_exact_Q = dolfinx.Function(Q)
    e_exact_Q.interpolate(e_exact)

    assert(np.isclose((e_exact_Q.vector - e_Q.vector).norm(), 0.0))
