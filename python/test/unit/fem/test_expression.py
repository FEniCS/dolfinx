# Copyright (C) 2019 Michal Habera
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ctypes
import ctypes.util

import cffi
import numba
import numpy as np
import numpy.typing

import basix
import dolfinx.cpp
import ufl
from dolfinx.cpp.la.petsc import create_matrix
from dolfinx.fem.petsc import load_petsc_lib
from dolfinx.fem import (Constant, Expression, Function, FunctionSpace,
                         VectorFunctionSpace, create_sparsity_pattern, form)
from dolfinx.mesh import create_unit_square

from mpi4py import MPI
from petsc4py import PETSc

dolfinx.cpp.common.init_logging(["-v"])

# Get PETSc int and scalar types
if np.dtype(PETSc.ScalarType).kind == 'c':
    complex = True
else:
    complex = False

scalar_size = np.dtype(PETSc.ScalarType).itemsize
index_size = np.dtype(PETSc.IntType).itemsize

if index_size == 8:
    c_int_t = "int64_t"
    ctypes_index: np.typing.DTypeLike = ctypes.c_int64
elif index_size == 4:
    c_int_t = "int32_t"
    ctypes_index = ctypes.c_int32
else:
    raise RuntimeError("Cannot translate PETSc index size into a C type, index_size: {}.".format(index_size))

if complex and scalar_size == 16:
    c_scalar_t = "double _Complex"
    numba_scalar_t = numba.types.complex128
elif complex and scalar_size == 8:
    c_scalar_t = "float _Complex"
    numba_scalar_t = numba.types.complex64
elif not complex and scalar_size == 8:
    c_scalar_t = "double"
    numba_scalar_t = numba.types.float64
elif not complex and scalar_size == 4:
    c_scalar_t = "float"
    numba_scalar_t = numba.types.float32
else:
    raise RuntimeError(
        "Cannot translate PETSc scalar type to a C type, complex: {} size: {}.".format(complex, scalar_size))


ffi = cffi.FFI()
# Get MatSetValuesLocal from PETSc available via cffi in ABI mode
ffi.cdef("""int MatSetValuesLocal(void* mat, {0} nrow, const {0}* irow,
                                {0} ncol, const {0}* icol, const {1}* y, int addv);
int MatZeroRowsLocal(void* mat, {0} nrow, const {0}* irow, {1} diag);
int MatAssemblyBegin(void* mat, int mode);
int MatAssemblyEnd(void* mat, int mode);
""".format(c_int_t, c_scalar_t))

petsc_lib_cffi = load_petsc_lib(ffi.dlopen)
MatSetValues = petsc_lib_cffi.MatSetValuesLocal


def test_rank0():
    """Test evaluation of UFL expression.
    This test evaluates gradient of P2 function at interpolation points
    of vector dP1 element.
    For a donor function f(x, y) = x^2 + 2*y^2 result is compared with the
    exact gradient grad f(x, y) = [2*x, 4*y].
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5)
    P2 = FunctionSpace(mesh, ("P", 2))
    vdP1 = VectorFunctionSpace(mesh, ("DG", 1))

    f = Function(P2)

    def expr1(x):
        return x[0] ** 2 + 2.0 * x[1] ** 2

    f.interpolate(expr1)

    ufl_expr = ufl.grad(f)
    points = vdP1.element.interpolation_points()

    compiled_expr = Expression(ufl_expr, points)
    num_cells = mesh.topology.index_map(2).size_local
    array_evaluated = compiled_expr.eval(np.arange(num_cells, dtype=np.int32))

    @numba.njit
    def scatter(vec, array_evaluated, dofmap):
        for i in range(num_cells):
            for j in range(3):
                for k in range(2):
                    vec[2 * dofmap[i * 3 + j] + k] = array_evaluated[i, 2 * j + k]

    # Data structure for the result
    b = Function(vdP1)

    dofmap = vdP1.dofmap.list.array
    scatter(b.x.array, array_evaluated, dofmap)
    b.x.scatter_forward()

    def grad_expr1(x):
        return np.vstack((2.0 * x[0], 4.0 * x[1]))

    b2 = Function(vdP1)
    b2.interpolate(grad_expr1)

    assert np.allclose(b2.x.array, b.x.array)


def test_rank1_hdiv():
    """Test rank-1 Expression, i.e. Expression containing Argument (TrialFunction)
    Test compiles linear interpolation operator RT_2 -> vector DG_2 and assembles it into
    global matrix A. Input space RT_2 is chosen because it requires dof permutations.
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)
    vdP1 = VectorFunctionSpace(mesh, ("DG", 2))
    RT1 = FunctionSpace(mesh, ("RT", 2))

    f = ufl.TrialFunction(RT1)

    points = vdP1.element.interpolation_points()
    compiled_expr = Expression(f, points)

    num_cells = mesh.topology.index_map(2).size_local
    array_evaluated = compiled_expr.eval(np.arange(num_cells, dtype=np.int32))

    @numba.njit
    def scatter(A, array_evaluated, dofmap0, dofmap1):
        for i in range(num_cells):
            rows = dofmap0[i, :]
            cols = dofmap1[i, :]
            A_local = array_evaluated[i, :]
            MatSetValues(A, 12, rows.ctypes, 8, cols.ctypes, A_local.ctypes, 1)

    a = form(ufl.inner(f, ufl.TestFunction(vdP1)) * ufl.dx)
    sparsity_pattern = create_sparsity_pattern(a)
    sparsity_pattern.assemble()
    A = create_matrix(MPI.COMM_WORLD, sparsity_pattern)

    dofmap_col = RT1.dofmap.list.array.reshape(-1, 8).astype(np.dtype(PETSc.IntType))
    dofmap_row = vdP1.dofmap.list.array

    dofmap_row_unrolled = (2 * np.repeat(dofmap_row, 2).reshape(-1, 2)
                           + np.arange(2)).flatten()
    dofmap_row = dofmap_row_unrolled.reshape(-1, 12).astype(np.dtype(PETSc.IntType))
    scatter(A.handle, array_evaluated, dofmap_row, dofmap_col)
    A.assemble()

    g = Function(RT1, name="g")

    def expr1(x):
        return np.row_stack((np.sin(x[0]), np.cos(x[1])))

    # Interpolate a numpy expression into RT1
    g.interpolate(expr1)

    # Interpolate RT1 into vdP1 (non-compiled interpolation)
    h = Function(vdP1)
    h.interpolate(g)

    # Interpolate RT1 into vdP1 (compiled, mat-vec interpolation)
    h2 = Function(vdP1)
    h2.vector.axpy(1.0, A * g.vector)

    assert np.isclose((h2.vector - h.vector).norm(), 0.0)


def test_simple_evaluation():
    """Test evaluation of UFL Expression.

    This test evaluates a UFL Expression on cells of the mesh and compares the
    result with an analytical expression.

    For a function f(x, y) = 3*(x^2 + 2*y^2) the result is compared with the
    exact gradient:

        grad f(x, y) = 3*[2*x, 4*y].

    (x^2 + 2*y^2) is first interpolated into a P2 finite element space. The
    scaling by a constant factor of 3 and the gradient is calculated using code
    generated by FFCx. The analytical solution is found by evaluating the
    spatial coordinates as an Expression using UFL/FFCx and passing the result
    to a numpy function that calculates the exact gradient.
    """
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 3)
    P2 = FunctionSpace(mesh, ("P", 2))

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

    expr = Function(P2)
    expr.interpolate(exact_expr)

    ufl_grad_f = Constant(mesh, PETSc.ScalarType(3.0)) * ufl.grad(expr)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    grad_f_expr = Expression(ufl_grad_f, points)
    assert grad_f_expr.X().shape[0] == points.shape[0]
    assert grad_f_expr.value_size == 2

    # NOTE: Cell numbering is process local.
    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    grad_f_evaluated = grad_f_expr.eval(cells)
    assert grad_f_evaluated.shape[0] == cells.shape[0]
    assert grad_f_evaluated.shape[1] == grad_f_expr.value_size * grad_f_expr.X().shape[0]

    # Evaluate points in global space
    ufl_x = ufl.SpatialCoordinate(mesh)
    x_expr = Expression(ufl_x, points)
    assert x_expr.X().shape[0] == points.shape[0]
    assert x_expr.value_size == 2
    x_evaluated = x_expr.eval(cells)
    assert x_evaluated.shape[0] == cells.shape[0]
    assert x_evaluated.shape[1] == x_expr.X().shape[0] * x_expr.value_size

    # Evaluate exact gradient using global points
    grad_f_exact = exact_grad_f(x_evaluated)

    assert np.allclose(grad_f_evaluated, grad_f_exact)


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
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 6)

    quadrature_degree = 2
    quadrature_points, wts = basix.make_quadrature(basix.CellType.triangle, quadrature_degree)
    Q_element = ufl.VectorElement("Quadrature", ufl.triangle, quadrature_degree, quad_scheme="default")
    Q = FunctionSpace(mesh, Q_element)
    P2 = FunctionSpace(mesh, ("P", 2))

    T = Function(P2)
    T.interpolate(lambda x: x[0] + 2.0 * x[1])
    A = Constant(mesh, PETSc.ScalarType(1.0))
    B = Constant(mesh, PETSc.ScalarType(2.0))

    K = 1.0 / (A + B * T)
    e = B * K**2 * ufl.grad(T)

    e_expr = Expression(e, quadrature_points)

    map_c = mesh.topology.index_map(mesh.topology.dim)
    num_cells = map_c.size_local + map_c.num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)

    e_eval = e_expr.eval(cells)

    # Assemble into Function
    e_Q = Function(Q)
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

    # FIXME: Below is only for testing purposes,
    # never to be used in user code!
    #
    # Replace when interpolation into Quadrature element works.
    coord_dofs = mesh.geometry.dofmap
    x_g = mesh.geometry.x
    tdim = mesh.topology.dim
    Q_dofs = Q.dofmap.list.array.reshape(num_cells, quadrature_points.shape[0])
    bs = Q.dofmap.bs

    Q_dofs_unrolled = bs * np.repeat(Q_dofs, bs).reshape(-1, bs) + np.arange(bs)
    Q_dofs_unrolled = Q_dofs_unrolled.reshape(-1, bs * quadrature_points.shape[0]).astype(Q_dofs.dtype)

    with e_Q.vector.localForm() as local:
        e_exact_eval = np.zeros_like(local.array)
        for cell in range(num_cells):
            xg = x_g[coord_dofs.links(cell), :tdim]
            x = mesh.geometry.cmap.push_forward(quadrature_points, xg)
            e_exact_eval[Q_dofs_unrolled[cell]] = e_exact(x.T).T.flatten()
        assert np.allclose(local.array, e_exact_eval)


def test_expression_eval_cells_subset():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 4)
    V = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))

    cells_imap = mesh.topology.index_map(mesh.topology.dim)
    all_cells = np.arange(
        cells_imap.size_local + cells_imap.num_ghosts, dtype=np.int32)
    cells_to_dofs = np.fromiter(
        map(V.dofmap.cell_dofs, all_cells), dtype=np.int32)
    dofs_to_cells = np.argsort(cells_to_dofs)

    u = dolfinx.fem.Function(V)
    u.x.array[:] = dofs_to_cells
    u.x.scatter_forward()
    e = dolfinx.fem.Expression(u, V.element.interpolation_points())

    # Test eval on single cell
    for c in range(cells_imap.size_local):
        u_ = e.eval(np.array([c], dtype=np.int32))
        assert np.allclose(u_, float(c))

    # Test eval on unordered cells
    cells = np.arange(cells_imap.size_local, dtype=np.int32)[::-1]
    u_ = e.eval(cells).flatten()
    assert np.allclose(u_, cells)

    # Test eval on unordered and non sequential cells
    cells = np.arange(cells_imap.size_local, dtype=np.int32)[::-2]
    u_ = e.eval(cells)
    assert np.allclose(u_.ravel(), cells)
