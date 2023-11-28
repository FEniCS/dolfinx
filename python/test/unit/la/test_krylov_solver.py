# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the KrylovSolver interface"""

from contextlib import ExitStack

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import pytest

import ufl
from dolfinx import la
from dolfinx.fem import Function, dirichletbc, form, functionspace, locate_dofs_topological
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from ufl import Identity, TestFunction, TrialFunction, dot, dx, grad, inner, sym, tr


def test_krylov_solver_lu():
    mesh = create_unit_square(MPI.COMM_WORLD, 12, 12)
    V = functionspace(mesh, ("Lagrange", 1))
    u, v = TrialFunction(V), TestFunction(V)

    a = form(inner(u, v) * dx)
    L = form(inner(1.0, v) * dx)
    A = assemble_matrix(a)
    A.assemble()
    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    norm = 13.0

    solver = PETSc.KSP().create(mesh.comm)
    solver.setOptionsPrefix("test_lu_")
    opts = PETSc.Options("test_lu_")
    opts["ksp_type"] = "preonly"
    opts["pc_type"] = "lu"
    solver.setFromOptions()
    x = A.createVecRight()
    solver.setOperators(A)
    solver.solve(b, x)

    # *Tight* tolerance for LU solves
    assert x.norm(PETSc.NormType.N2) == pytest.approx(norm, rel=1.0e7, abs=1.0e-7)

    solver.destroy()
    A.destroy()
    b.destroy()
    x.destroy()


@pytest.mark.skip
def test_krylov_samg_solver_elasticity():
    "Test PETScKrylovSolver with smoothed aggregation AMG"

    def build_nullspace(V, x):
        """Function to build null space for 2D elasticity"""

        # Create list of vectors for null space
        ns = [x.copy() for i in range(3)]

        with ExitStack() as stack:
            vec_local = [stack.enter_context(x.localForm()) for x in ns]
            basis = [np.asarray(x) for x in vec_local]

            # Build null space basis
            dofs = [V.sub(i).dofmap.list.array_r for i in range(2)]
            for i in range(2):
                basis[i][dofs[i]] = 1.0
            x = V.tabulate_dof_coordinates()
            basis[2][dofs[0]] = -x[dofs[0], 1]
            basis[2][dofs[1]] = x[dofs[1], 0]

        la.orthonormalize(ns)
        return ns

    def amg_solve(N, method):
        # Elasticity parameters
        E = 1.0e9
        nu = 0.3
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Stress computation
        def sigma(v):
            return 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(grad(v))) * Identity(2)

        # Define problem
        mesh = create_unit_square(MPI.COMM_WORLD, N, N)
        gdim = mesh.geometry.dim
        V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
        u = TrialFunction(V)
        v = TestFunction(V)

        facetdim = mesh.topology.dim - 1
        bndry_facets = locate_entities_boundary(mesh, facetdim, lambda x: np.full(x.shape[1], True))
        bdofs = locate_dofs_topological(V.sub(0), V, facetdim, bndry_facets)
        bc = dirichletbc(PETSc.ScalarType(0), bdofs, V.sub(0))

        # Forms
        a, L = inner(sigma(u), grad(v)) * dx, dot(ufl.as_vector((1.0, 1.0)), v) * dx

        # Assemble linear algebra objects
        A = assemble_matrix(a, [bc])
        A.assemble()
        b = assemble_vector(L)
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        # Create solution function
        u = Function(V)

        # Create near null space basis and orthonormalize
        null_space = build_nullspace(V, u.vector)

        # Attached near-null space to matrix
        A.set_near_nullspace(null_space)

        # Test that basis is orthonormal
        assert null_space.is_orthonormal()

        # Create PETSC smoothed aggregation AMG preconditioner, and
        # create CG solver
        solver = PETSc.KSP().create(mesh.comm)
        solver.setType("cg")

        # Set matrix operator
        solver.setOperators(A)

        # Compute solution and return number of iterations
        return solver.solve(b, u.vector)

    # Set some multigrid smoother parameters
    opts = PETSc.Options()
    opts["mg_levels_ksp_type"] = "chebyshev"
    opts["mg_levels_pc_type"] = "jacobi"

    # Improve estimate of eigenvalues for Chebyshev smoothing
    opts["mg_levels_esteig_ksp_type"] = "cg"
    opts["mg_levels_ksp_chebyshev_esteig_steps"] = 50

    # Build list of smoothed aggregation preconditioners
    methods = ["petsc_amg"]
    # if "ml_amg" in PETScPreconditioner.preconditioners():
    #    methods.append("ml_amg")

    # Test iteration count with increasing mesh size for each
    # preconditioner
    for method in methods:
        for N in [8, 16, 32, 64]:
            print(f"Testing method '{method}' with {N} x {N} mesh")
            niter = amg_solve(N, method)
            assert niter < 18
