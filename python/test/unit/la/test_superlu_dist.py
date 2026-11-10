# Copyright (C) 2026 Jack S. Hale, Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for SuperLU_DIST."""

from mpi4py import MPI

import numpy as np
import pytest

import dolfinx
from dolfinx.common import index_map
from dolfinx.fem import (
    Expression,
    Function,
    apply_lifting,
    assemble_matrix,
    assemble_scalar,
    assemble_vector,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.la import InsertMode, matrix_csr, vector
from dolfinx.mesh import create_unit_square, exterior_facet_indices
from ufl import SpatialCoordinate, TestFunction, TrialFunction, as_vector, div, dx, grad, inner


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_solver(dtype) -> None:
    """Manufactured Poisson and screened problem with exact solution u = x[1]**3.

    Test includes various checks that SuperLU_DIST Factor option works correctly
    for efficient re-use of permutations and factors.
    """
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    mesh_dtype = dtype().real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=mesh_dtype)
    V = functionspace(mesh, ("Lagrange", 4))
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(grad(u), grad(v)) * dx
    a = form(a, dtype=dtype)

    # Exact solution
    def u_ex(x):
        return x[1] ** 3

    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ex(x)))
    L = inner(f, v) * dx
    L = form(L, dtype=dtype)

    u_bc = Function(V, dtype=dtype)
    u_bc.interpolate(u_ex)

    # Create Dirichlet boundary condition
    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bndry_facets = exterior_facet_indices(mesh.topology)
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    bc = dirichletbc(u_bc, bdofs)

    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(InsertMode.add)
    bc.set(b.array)

    a = form(a, dtype=dtype)
    A = assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()

    def check_error(u_ex, uh):
        M = (u_ex(x) - uh) ** 2 * dx
        M = form(M, dtype=dtype)
        error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
        eps = np.sqrt(np.finfo(dtype).eps)
        assert np.isclose(error, 0.0, atol=eps)

    def solve_and_check(solver, b):
        uh = Function(V, dtype=dtype)
        error_code = solver.solve(b, uh.x)
        assert error_code == 0
        uh.x.scatter_forward()
        return uh

    # Standard solve
    A_superlu = superlu_dist_matrix(A)
    solver_1 = superlu_dist_solver(A_superlu)
    solver_1.set_option("SymmetricMode", "YES")
    uh_1 = solve_and_check(solver_1, b)
    check_error(u_ex, uh_1)

    # Check second solve re-using permutations from previous solve
    solver_1.set_option("Fact", "FACTORED")
    solve_and_check(solver_1, b)
    uh_2 = solve_and_check(solver_1, b)
    check_error(u_ex, uh_2)

    # Check can do same solve with MatrixCSR A again.
    A_superlu_2 = superlu_dist_matrix(A)
    solver_2 = superlu_dist_solver(A_superlu_2)
    solver_2.set_option("SymmetricMode", "YES")
    uh_3 = solve_and_check(solver_2, b)
    check_error(u_ex, uh_3)

    # Assemble a new operator and re-use row permutation from solve with old
    # operator.
    gamma = 1e-1
    a_2 = (inner(grad(u), grad(v)) + gamma * inner(u, v)) * dx
    a_2 = form(a_2, dtype=dtype)

    f_2 = -div(grad(u_ex(x))) + gamma * u_ex(x)
    L_2 = inner(f_2, v) * dx
    L_2 = form(L_2, dtype=dtype)

    b_2 = assemble_vector(L_2)
    apply_lifting(b_2.array, [a_2], bcs=[[bc]])
    b_2.scatter_reverse(InsertMode.add)
    bc.set(b_2.array)

    A_3 = assemble_matrix(a_2, bcs=[bc])
    A_3.scatter_reverse()

    A_superlu_3 = superlu_dist_matrix(A_3)
    solver_2.set_A(A_superlu_3, "SamePattern")
    uh_2 = solve_and_check(solver_2, b_2)
    check_error(u_ex, uh_2)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float32,
        np.float64,
        pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
    ],
)
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_solver_blocked(dtype) -> None:
    """Vector Poisson problem on a vector Lagrange space (block size 2)."""
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    mesh_dtype = dtype().real.dtype
    mesh = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=mesh_dtype)
    V = functionspace(mesh, ("Lagrange", 3, (2,)))
    u, v = TrialFunction(V), TestFunction(V)

    a = form(inner(grad(u), grad(v)) * dx, dtype=dtype)

    def u_ex(x):
        return np.vstack((x[1] ** 3, x[0] ** 3))

    x = SpatialCoordinate(mesh)
    u_ex_ufl = as_vector((x[1] ** 3, x[0] ** 3))
    f = -div(grad(u_ex_ufl))
    L = form(inner(f, v) * dx, dtype=dtype)

    u_bc = Function(V, dtype=dtype)
    u_bc.interpolate(u_ex)

    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bndry_facets = exterior_facet_indices(mesh.topology)
    bdofs = locate_dofs_topological(V, facetdim, bndry_facets)
    bc = dirichletbc(u_bc, bdofs)

    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(InsertMode.add)
    bc.set(b.array)

    A = assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()
    assert A.block_size == [2, 2]

    A_superlu = superlu_dist_matrix(A)
    solver = superlu_dist_solver(A_superlu)
    solver.set_option("SymmetricMode", "YES")
    uh = Function(V, dtype=dtype)
    error_code = solver.solve(b, uh.x)
    assert error_code == 0
    uh.x.scatter_forward()

    M = form(inner(u_ex_ufl - uh, u_ex_ufl - uh) * dx, dtype=dtype)
    error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
    eps = np.sqrt(np.finfo(dtype).eps)
    assert np.isclose(error, 0.0, atol=eps)


def _solve_block_system(a01, a10, f, exact, dtype):
    """Solve [[a00, a01], [a10, a11]] with SuperLU_DIST and check the error.

    The diagonal blocks are Helmholtz operators on the spaces of the
    off-diagonal forms; ``a10`` may be ``None``.
    """
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    V, W = (arg.ufl_function_space() for arg in a01.arguments())
    u, v = TrialFunction(V), TestFunction(V)
    w, q = TrialFunction(W), TestFunction(W)
    a = [
        [
            form(inner(grad(u), grad(v)) * dx + inner(u, v) * dx, dtype=dtype),
            form(a01, dtype=dtype),
        ],
        [
            None if a10 is None else form(a10, dtype=dtype),
            form(inner(grad(w), grad(q)) * dx + inner(w, q) * dx, dtype=dtype),
        ],
    ]
    L = [form(inner(f[0], v) * dx, dtype=dtype), form(inner(f[1], q) * dx, dtype=dtype)]

    mesh = V.mesh
    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bndry_facets = exterior_facet_indices(mesh.topology)
    bcs = []
    for space, ex in zip((V, W), exact, strict=True):
        g = Function(space, dtype=dtype)
        g.interpolate(Expression(ex, space.element.interpolation_points, dtype=dtype))
        bcs.append(dirichletbc(g, locate_dofs_topological(space, facetdim, bndry_facets)))

    A = [[None if aij is None else assemble_matrix(aij, bcs=bcs) for aij in row] for row in a]
    for row in A:
        for Aij in row:
            if Aij is not None:
                Aij.scatter_reverse()
    b = []
    for i, Li in enumerate(L):
        bi = assemble_vector(Li)
        apply_lifting(bi.array, a[i], bcs=[[bcs[0]], [bcs[1]]])
        bi.scatter_reverse(InsertMode.add)
        bcs[i].set(bi.array)
        b.append(bi)

    solver = superlu_dist_solver(superlu_dist_matrix(A))
    uh, wh = Function(V, dtype=dtype), Function(W, dtype=dtype)
    assert solver.solve(b, [uh.x, wh.x]) == 0
    uh.x.scatter_forward()
    wh.x.scatter_forward()

    eps = np.sqrt(np.finfo(dtype).eps)
    for fh, ex in zip((uh, wh), exact, strict=True):
        M = form(inner(ex - fh, ex - fh) * dx, dtype=dtype)
        error = mesh.comm.allreduce(assemble_scalar(M), op=MPI.SUM)
        assert np.isclose(error, 0.0, atol=eps)


_block_dtypes = [
    np.float32,
    np.float64,
    pytest.param(np.complex128, marks=pytest.mark.xfail_win32_complex),
]


@pytest.mark.parametrize("coupled", [True, False], ids=["coupled", "triangular"])
@pytest.mark.parametrize("dtype", _block_dtypes)
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_solver_block_matrix_scalar(dtype, coupled) -> None:
    """Two scalar fields: all blocks have block size 1."""
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 5, dtype=dtype().real.dtype)
    V = functionspace(mesh, ("Lagrange", 2))
    W = functionspace(mesh, ("Lagrange", 2))
    u, v = TrialFunction(V), TestFunction(V)
    w, q = TrialFunction(W), TestFunction(W)

    x = SpatialCoordinate(mesh)
    u_ex = 1 + x[0] ** 2 + 2 * x[1] ** 2
    w_ex = x[0] * x[1] + x[0]
    f0 = -div(grad(u_ex)) + u_ex - w_ex
    f1 = -div(grad(w_ex)) + w_ex - (u_ex if coupled else 0)
    a10 = -inner(u, q) * dx if coupled else None
    _solve_block_system(-inner(w, v) * dx, a10, (f0, f1), (u_ex, w_ex), dtype)


@pytest.mark.parametrize("coupled", [True, False], ids=["coupled", "triangular"])
@pytest.mark.parametrize("dtype", _block_dtypes)
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_solver_block_matrix_vector(dtype, coupled) -> None:
    """A vector and a scalar field: block sizes differ between blocks."""
    mesh = create_unit_square(MPI.COMM_WORLD, 6, 5, dtype=dtype().real.dtype)
    V = functionspace(mesh, ("Lagrange", 2, (2,)))
    W = functionspace(mesh, ("Lagrange", 2))
    u, v = TrialFunction(V), TestFunction(V)
    w, q = TrialFunction(W), TestFunction(W)

    x = SpatialCoordinate(mesh)
    u_ex = as_vector((1 + x[0] ** 2, 2 * x[1] ** 2 - x[0]))
    w_ex = x[0] * x[1] + x[0]
    f0 = -div(grad(u_ex)) + u_ex + grad(w_ex)
    f1 = -div(grad(w_ex)) + w_ex - (div(u_ex) if coupled else 0)
    a10 = -inner(div(u), q) * dx if coupled else None
    _solve_block_system(-inner(w, div(v)) * dx, a10, (f0, f1), (u_ex, w_ex), dtype)


@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_block_matrix_invalid() -> None:
    """Block matrices whose blocks do not form a block layout raise."""
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = functionspace(mesh, ("Lagrange", 1))
    W = functionspace(mesh, ("Lagrange", 2))
    A_V = assemble_matrix(form(inner(TrialFunction(V), TestFunction(V)) * dx))
    A_W = assemble_matrix(form(inner(TrialFunction(W), TestFunction(W)) * dx))

    for A in (
        [[A_V], [A_V, A_V]],  # not rectangular
        [[A_V, None], [None, None]],  # block row and column without a block
        [[A_V, A_W]],  # blocks of a block row with different rows
        [[A_V], [A_W]],  # blocks of a block column with different columns
        [[None]],  # no blocks
    ):
        with pytest.raises(ValueError):
            superlu_dist_matrix(A)

    # Vectors whose owned size differs from the rows of the matrix
    solver = superlu_dist_solver(superlu_dist_matrix([[A_V, None], [None, A_W]]))
    b_V = vector(V.dofmap.index_map, dtype=A_V.data.dtype)
    x_V = vector(V.dofmap.index_map, dtype=A_V.data.dtype)
    with pytest.raises(ValueError):
        solver.solve(b_V, x_V)
    b_W = vector(W.dofmap.index_map, dtype=A_V.data.dtype)
    with pytest.raises(ValueError):
        solver.solve([b_V, b_W], [x_V, x_V])

    # Blocks of different dtypes, checked before their layout. The other
    # dtype differs from the default scalar type of the build.
    other = np.float64 if A_V.data.dtype == np.float32 else np.float32
    mesh_other = create_unit_square(MPI.COMM_WORLD, 4, 4, dtype=other)
    V_other = functionspace(mesh_other, ("Lagrange", 1))
    a_other = form(inner(TrialFunction(V_other), TestFunction(V_other)) * dx, dtype=other)
    with pytest.raises(TypeError):
        superlu_dist_matrix([[A_V, None], [None, assemble_matrix(a_other)]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
@pytest.mark.skipif(MPI.COMM_WORLD.size > 1, reason="Hand-built single-rank matrix")
def test_superlu_solver_asymmetric_blocks(dtype) -> None:
    """Hand-built MatrixCSR with bs[0] = 2 and bs[1] = 3 and final size 6 x 6."""
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    bs0, bs1 = 2, 3
    n_row_blocks, n_col_blocks = 3, 2

    im_row = index_map(MPI.COMM_WORLD, n_row_blocks)
    im_col = index_map(MPI.COMM_WORLD, n_col_blocks)
    sp = dolfinx.la.sparsity_pattern(MPI.COMM_WORLD, [im_row, im_col], [bs0, bs1])
    for i in range(n_row_blocks):
        for j in range(n_col_blocks):
            sp.insert(i, j)
    sp.finalize()

    A = matrix_csr(sp, dtype=dtype)
    assert A.block_size == [bs0, bs1]

    rng = np.random.default_rng(0)
    A_dense = (np.eye(6) * 10.0 + rng.standard_normal((6, 6))).astype(dtype)

    for i in range(n_row_blocks):
        for j in range(n_col_blocks):
            block_idx = i * n_col_blocks + j
            for i0 in range(bs0):
                for i1 in range(bs1):
                    A.data[block_idx * bs0 * bs1 + i0 * bs1 + i1] = A_dense[
                        i * bs0 + i0, j * bs1 + i1
                    ]

    b_np = rng.standard_normal(6).astype(dtype)
    x_expected = np.linalg.solve(A_dense, b_np)

    b = vector(im_row, bs=bs0, dtype=dtype)
    b.array[:] = b_np
    u = vector(im_col, bs=bs1, dtype=dtype)

    A_superlu = superlu_dist_matrix(A)
    solver = superlu_dist_solver(A_superlu)
    error_code = solver.solve(b, u)
    assert error_code == 0
    u.scatter_forward()

    assert np.allclose(u.array, x_expected)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"RowPerm": "NOROWPERM"},
        {"ColPerm": "MMD_AT_PLUS_A"},
    ],
)
@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_solver_default_ordering(options) -> None:
    """Poisson problem on a mesh large enough to need a nontrivial ordering.

    Ranks computing different column orderings deadlock in parallel, which
    small meshes do not expose.
    """
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    mesh = create_unit_square(MPI.COMM_WORLD, 32, 32)
    V = functionspace(mesh, ("Lagrange", 2))
    u, v = TrialFunction(V), TestFunction(V)
    x = SpatialCoordinate(mesh)
    u_ex = 1 + x[0] ** 2 + 2 * x[1] ** 2

    a = form(inner(grad(u), grad(v)) * dx)
    L = form(inner(-div(grad(u_ex)), v) * dx)

    u_bc = Function(V)
    u_bc.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
    facetdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
    bdofs = locate_dofs_topological(V, facetdim, exterior_facet_indices(mesh.topology))
    bc = dirichletbc(u_bc, bdofs)

    b = assemble_vector(L)
    apply_lifting(b.array, [a], bcs=[[bc]])
    b.scatter_reverse(InsertMode.add)
    bc.set(b.array)
    A = assemble_matrix(a, bcs=[bc])
    A.scatter_reverse()

    solver = superlu_dist_solver(superlu_dist_matrix(A))
    for name, value in options.items():
        solver.set_option(name, value)
    uh = Function(V)
    assert solver.solve(b, uh.x) == 0
    uh.x.scatter_forward()

    error = mesh.comm.allreduce(assemble_scalar(form((u_ex - uh) ** 2 * dx)), op=MPI.SUM)
    assert np.isclose(error, 0.0, atol=np.sqrt(np.finfo(np.float64).eps))


@pytest.mark.skipif(not dolfinx.has_superlu_dist, reason="No SuperLU_DIST")
def test_superlu_solver_parsymbfact_requires_parmetis() -> None:
    """ParSymbFact without PARMETIS ordering is rejected before factorisation."""
    from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver

    mesh = create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = TrialFunction(V), TestFunction(V)
    A = assemble_matrix(form(inner(u, v) * dx))
    A.scatter_reverse()
    solver = superlu_dist_solver(superlu_dist_matrix(A))
    solver.set_option("ColPerm", "MMD_AT_PLUS_A")
    solver.set_option("ParSymbFact", "YES")
    b = assemble_vector(form(inner(1.0, v) * dx))
    with pytest.raises(ValueError):
        solver.solve(b, Function(V).x)
