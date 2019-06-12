# -*- coding: utf-8 -*-
# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for the KrylovSolver interface"""

from contextlib import ExitStack

import numpy as np
import pytest

import ufl
from dolfin import (MPI, DirichletBC, Function, FunctionSpace, TestFunction,
                    TrialFunction, UnitSquareMesh, VectorFunctionSpace)
from dolfin.fem import apply_lifting, assemble_matrix, assemble_vector, set_bc
from dolfin.la import PETScKrylovSolver, PETScOptions, VectorSpaceBasis
from petsc4py import PETSc
from ufl import Identity, dot, dx, grad, inner, sym, tr


def test_krylov_solver_lu():

    mesh = UnitSquareMesh(MPI.comm_world, 12, 12)
    V = FunctionSpace(mesh, ("Lagrange", 1))
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(u, v) * dx
    L = inner(1.0, v) * dx
    A = assemble_matrix(a)
    A.assemble()
    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    norm = 13.0

    solver = PETScKrylovSolver(mesh.mpi_comm())
    solver.set_options_prefix("test_lu_")
    PETScOptions.set("test_lu_ksp_type", "preonly")
    PETScOptions.set("test_lu_pc_type", "lu")
    solver.set_from_options()
    x = A.createVecRight()
    solver.set_operator(A)
    solver.solve(x, b)

    # *Tight* tolerance for LU solves
    assert round(x.norm(PETSc.NormType.N2) - norm, 12) == 0


@pytest.mark.skip
def test_krylov_samg_solver_elasticity():
    "Test PETScKrylovSolver with smoothed aggregation AMG"

    def build_nullspace(V, x):
        """Function to build null space for 2D elasticity"""

        # Create list of vectors for null space
        nullspace_basis = [x.copy() for i in range(3)]

        with ExitStack() as stack:
            vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
            basis = [np.asarray(x) for x in vec_local]

            # Build translational null space basis
            V.sub(0).dofmap().set(basis[0], 1.0)
            V.sub(1).dofmap().set(basis[1], 1.0)

            # Build rotational null space basis
            V.sub(0).set_x(basis[2], -1.0, 1)
            V.sub(1).set_x(basis[2], 1.0, 0)

        null_space = VectorSpaceBasis(nullspace_basis)
        null_space.orthonormalize()
        return null_space

    def amg_solve(N, method):
        # Elasticity parameters
        E = 1.0e9
        nu = 0.3
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Stress computation
        def sigma(v):
            return 2.0 * mu * sym(grad(v)) + lmbda * tr(sym(
                grad(v))) * Identity(2)

        # Define problem
        mesh = UnitSquareMesh(MPI.comm_world, N, N)
        V = VectorFunctionSpace(mesh, 'Lagrange', 1)
        bc0 = Function(V)
        with bc0.vector().localForm() as bc_local:
            bc_local.set(0.0)
        bc = DirichletBC(V.sub(0), bc0,
                         lambda x, on_boundary: on_boundary)
        u = TrialFunction(V)
        v = TestFunction(V)

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
        null_space = build_nullspace(V, u.vector())

        # Attached near-null space to matrix
        A.set_near_nullspace(null_space)

        # Test that basis is orthonormal
        assert null_space.is_orthonormal()

        # Create PETSC smoothed aggregation AMG preconditioner, and
        # create CG solver
        solver = PETScKrylovSolver("cg", method)

        # Set matrix operator
        solver.set_operator(A)

        # Compute solution and return number of iterations
        return solver.solve(u.vector(), b)

    # Set some multigrid smoother parameters
    PETScOptions.set("mg_levels_ksp_type", "chebyshev")
    PETScOptions.set("mg_levels_pc_type", "jacobi")

    # Improve estimate of eigenvalues for Chebyshev smoothing
    PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
    PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

    # Build list of smoothed aggregation preconditioners
    methods = ["petsc_amg"]
    # if "ml_amg" in PETScPreconditioner.preconditioners():
    #    methods.append("ml_amg")

    # Test iteration count with increasing mesh size for each
    # preconditioner
    for method in methods:
        for N in [8, 16, 32, 64]:
            print("Testing method '{}' with {} x {} mesh".format(method, N, N))
            niter = amg_solve(N, method)
            assert niter < 18
