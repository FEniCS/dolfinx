# Copyright (C) 2019 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from mpi4py import MPI

import numpy as np
import pytest

import basix
import dolfinx.cpp
import ufl
from basix.ufl import quadrature_element
from dolfinx import fem, la
from dolfinx.fem import Constant, Expression, Function, form, functionspace
from dolfinx.mesh import create_unit_square

dolfinx.cpp.common.init_logging(["-v"])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_rank0(dtype):
    """Test evaluation of UFL expression.

    This test evaluates gradient of P2 function at interpolation points
    of vector dP1 element.

    For a donor function f(x, y) = x^2 + 2*y^2 result is compared with the
    exact gradient grad f(x, y) = [2*x, 4*y].

    """
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=dtype(0).real.dtype)
    gdim = mesh.geometry.dim
    P2 = functionspace(mesh, ("P", 2))
    vdP1 = functionspace(mesh, ("DG", 1, (gdim,)))

    f = Function(P2, dtype=dtype)
    f.interpolate(lambda x: x[0] ** 2 + 2.0 * x[1] ** 2)

    ufl_expr = ufl.grad(f)
    points = vdP1.element.interpolation_points()

    compiled_expr = Expression(ufl_expr, points, dtype=dtype)
    num_cells = mesh.topology.index_map(2).size_local
    array_evaluated = compiled_expr.eval(mesh, np.arange(num_cells, dtype=np.int32))

    def scatter(vec, array_evaluated, dofmap):
        for i in range(num_cells):
            for j in range(3):
                for k in range(2):
                    vec[2 * dofmap[i * 3 + j] + k] = array_evaluated[i, 2 * j + k]

    # Data structure for the result
    b = Function(vdP1, dtype=dtype)
    dofmap = vdP1.dofmap.list.flatten()
    scatter(b.x.array, array_evaluated, dofmap)
    b.x.scatter_forward()

    b2 = Function(vdP1, dtype=dtype)
    b2.interpolate(lambda x: np.vstack((2.0 * x[0], 4.0 * x[1])))

    assert np.allclose(b2.x.array, b.x.array, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_rank1_hdiv(dtype):
    """Test rank-1 Expression, i.e. Expression containing Argument
    (TrialFunction).

    Test compiles linear interpolation operator RT_2 ->
    vector DG_2 and assembles it into global matrix A. Input space RT_2
    is chosen because it requires dof permutations.

    """
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, dtype=dtype(0).real.dtype)
    gdim = mesh.geometry.dim
    vdP1 = functionspace(mesh, ("DG", 2, (gdim,)))
    RT1 = functionspace(mesh, ("RT", 2))
    f = ufl.TrialFunction(RT1)

    points = vdP1.element.interpolation_points()
    compiled_expr = Expression(f, points, dtype=dtype)
    num_cells = mesh.topology.index_map(2).size_local
    array_evaluated = compiled_expr.eval(mesh, np.arange(num_cells, dtype=np.int32))

    def scatter(A, array_evaluated, dofmap0, dofmap1):
        for i in range(num_cells):
            rows = dofmap0[i, :]
            cols = dofmap1[i, :]
            A_local = array_evaluated[i, :].reshape(len(rows), len(cols))
            for i, row in enumerate(rows):
                for j, col in enumerate(cols):
                    A[row, col] = A_local[i, j]

    dofmap_col = RT1.dofmap.list
    dofmap_row = vdP1.dofmap.list
    dofmap_row_unrolled = (2 * np.repeat(dofmap_row, 2).reshape(-1, 2) + np.arange(2)).flatten()
    dofmap_row = dofmap_row_unrolled.reshape(-1, 12)

    a = form(ufl.inner(f, ufl.TestFunction(vdP1)) * ufl.dx, dtype=dtype)
    A = fem.create_matrix(a, block_mode=la.BlockMode.expanded)
    As = A.to_scipy(ghosted=True)
    scatter(As, array_evaluated, dofmap_row, dofmap_col)
    A.scatter_reverse()

    gvec = la.vector(A.index_map(1), bs=A.block_size[1], dtype=dtype)
    g = Function(RT1, gvec, name="g", dtype=dtype)

    # Interpolate a numpy expression into RT1
    g.interpolate(lambda x: np.row_stack((np.sin(x[0]), np.cos(x[1]))))

    # Interpolate RT1 into vdP1 (non-compiled interpolation)
    h = Function(vdP1, dtype=dtype)
    h.interpolate(g)

    # Wrap A as SciPy sparse matrix, owned rows only
    A1 = A.to_scipy(ghosted=False)

    # Interpolate RT1 into vdP1 (compiled, mat-vec interpolation)
    h2 = Function(vdP1, dtype=dtype)
    h2.x.array[: A1.shape[0]] += A1 @ g.x.array
    h2.x.scatter_forward()
    assert np.linalg.norm(h2.x.array - h.x.array) == pytest.approx(0.0, abs=1.0e-4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_simple_evaluation(dtype):
    """Test evaluation of UFL Expression.

    This test evaluates a UFL Expression on cells of the mesh and
    compares the result with an analytical expression.

    For a function f(x, y) = 3*(x^2 + 2*y^2) the result is compared with
    the exact gradient:

        grad f(x, y) = 3*[2*x, 4*y].

    (x^2 + 2*y^2) is first interpolated into a P2 finite element space.
    The scaling by a constant factor of 3 and the gradient is calculated
    using code generated by FFCx. The analytical solution is found by
    evaluating the spatial coordinates as an Expression using UFL/FFCx
    and passing the result to a numpy function that calculates the exact
    gradient.

    """
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 3, dtype=xtype)
    P2 = functionspace(mesh, ("P", 2))

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

    expr = Function(P2, dtype=dtype)
    expr.interpolate(exact_expr)

    ufl_grad_f = Constant(mesh, dtype(3.0)) * ufl.grad(expr)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    grad_f_expr = Expression(ufl_grad_f, points, dtype=dtype)
    assert grad_f_expr.X().shape[0] == points.shape[0]
    assert grad_f_expr.value_size == 2

    # # NOTE: Cell numbering is process local
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    grad_f_evaluated = grad_f_expr.eval(mesh, cells)
    assert grad_f_evaluated.shape[0] == cells.shape[0]
    assert grad_f_evaluated.shape[1] == grad_f_expr.value_size * grad_f_expr.X().shape[0]

    # Evaluate points in global space
    ufl_x = ufl.SpatialCoordinate(mesh)
    x_expr = Expression(ufl_x, points, dtype=xtype)
    assert x_expr.X().shape[0] == points.shape[0]
    assert x_expr.value_size == 2
    x_evaluated = x_expr.eval(mesh, cells)
    assert x_evaluated.shape[0] == cells.shape[0]
    assert x_evaluated.shape[1] == x_expr.X().shape[0] * x_expr.value_size

    # Evaluate exact gradient using global points
    grad_f_exact = exact_grad_f(x_evaluated)
    assert np.allclose(grad_f_evaluated, grad_f_exact, rtol=1.0e-5, atol=1.0e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_assembly_into_quadrature_function(dtype):
    """Test assembly into a Quadrature function.

    This test evaluates a UFL Expression into a Quadrature function
    space by evaluating the Expression on all cells of the mesh, and
    then inserting the evaluated values into a Vector constructed from a
    matching Quadrature function space.

    Concretely, we consider the evaluation of:

        e = B*(K(T)))**2 * grad(T)

    where

        K = 1/(A + B*T)

    where A and B are Constants and T is a Coefficient on a P2 finite
    element space with T = x + 2*y.

    The result is compared with interpolating the analytical expression
    of e directly into the Quadrature space.

    In parallel, each process evaluates the Expression on both local
    cells and ghost cells so that no parallel communication is required
    after insertion into the vector.

    """
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 6, dtype=xtype)

    quadrature_degree = 2
    quadrature_points, _ = basix.make_quadrature(basix.CellType.triangle, quadrature_degree)
    quadrature_points = quadrature_points.astype(xtype)
    Q_element = quadrature_element("triangle", (2,), degree=quadrature_degree, scheme="default")
    Q = functionspace(mesh, Q_element)
    P2 = functionspace(mesh, ("P", 2))

    T = Function(P2, dtype=dtype)
    T.interpolate(lambda x: x[0] + 2.0 * x[1])
    A = Constant(mesh, dtype(1.0))
    B = Constant(mesh, dtype(2.0))

    K = 1.0 / (A + B * T)
    e = B * K**2 * ufl.grad(T)

    e_expr = Expression(e, quadrature_points, dtype=dtype)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    e_eval = e_expr.eval(mesh, cells)

    # # Assemble into Function
    e_Q = Function(Q, dtype=dtype)
    e_Q_local = e_Q.x.array
    bs = e_Q.function_space.dofmap.bs
    dofs = np.empty((bs * Q.dofmap.list.flatten().size,), dtype=np.int32)
    for i in range(bs):
        dofs[i::2] = bs * Q.dofmap.list.flatten() + i
    e_Q_local[dofs] = e_eval.flatten()

    def e_exact(x):
        T = x[0] + 2.0 * x[1]
        K = 1.0 / (A.value + B.value * T)

        grad_T = np.zeros((2, x.shape[1]))
        grad_T[0, :] = 1.0
        grad_T[1, :] = 2.0

        e = B.value * K**2 * grad_T
        return e

    # # FIXME: Below is only for testing purposes,
    # # never to be used in user code!
    # # TODO: Replace when interpolation into Quadrature element works.
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    tdim = mesh.topology.dim
    Q_dofs = Q.dofmap.list

    bs = Q.dofmap.bs
    Q_dofs_unrolled = bs * np.repeat(Q_dofs, bs).reshape(-1, bs) + np.arange(bs)
    Q_dofs_unrolled = Q_dofs_unrolled.reshape(-1, bs * quadrature_points.shape[0]).astype(
        Q_dofs.dtype
    )
    local = e_Q.x.array
    e_exact_eval = np.zeros_like(local)
    for cell in range(num_cells):
        xg = x_g[coord_dofs[cell], :tdim]
        x = mesh.geometry.cmap.push_forward(quadrature_points, xg)
        e_exact_eval[Q_dofs_unrolled[cell]] = e_exact(x.T).T.flatten()
    assert np.allclose(local, e_exact_eval)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_expression_eval_cells_subset(dtype):
    xtype = dtype(0).real.dtype
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 4, dtype=xtype)
    V = dolfinx.fem.functionspace(mesh, ("DG", 0))

    cells_imap = mesh.topology.index_map(mesh.topology.dim)
    all_cells = np.arange(cells_imap.size_local + cells_imap.num_ghosts, dtype=np.int32)
    cells_to_dofs = np.array([V.dofmap.cell_dofs(i)[0] for i in all_cells], dtype=np.int32)
    dofs_to_cells = np.argsort(cells_to_dofs)

    u = dolfinx.fem.Function(V, dtype=dtype)
    u.x.array[:] = dofs_to_cells
    u.x.scatter_forward()
    e = dolfinx.fem.Expression(u, V.element.interpolation_points())

    # Test eval on single cell
    for c in range(cells_imap.size_local):
        u_ = e.eval(mesh, np.array([c], dtype=np.int32))
        assert np.allclose(u_, float(c))

    # Test eval on unordered cells
    cells = np.arange(cells_imap.size_local - 1, -1, -1, dtype=np.int32)
    u_ = e.eval(mesh, cells).flatten()
    assert np.allclose(u_, cells)

    # Test eval on unordered and non sequential cells
    cells = np.arange(cells_imap.size_local - 1, -1, -2, dtype=np.int32)
    u_ = e.eval(mesh, cells)
    assert np.allclose(u_.ravel(), cells)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_expression_comm(dtype):
    xtype = dtype(0).real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=xtype)
    v = Constant(mesh, dtype(1))
    u = Function(functionspace(mesh, ("Lagrange", 1)), dtype=dtype)
    Expression(v, u.function_space.element.interpolation_points(), comm=MPI.COMM_WORLD)
    Expression(v, u.function_space.element.interpolation_points(), comm=MPI.COMM_SELF)
