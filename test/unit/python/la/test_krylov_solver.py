#!/usr/bin/env py.test

"""Unit tests for the KrylovSolver interface"""

# Copyright (C) 2014 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
import pytest
from dolfin import *
from dolfin_utils.test import skip_if_not_PETSc, skip_in_parallel, pushpop_parameters


@skip_if_not_PETSc
def test_krylov_samg_solver_elasticity(pushpop_parameters):
    "Test PETScKrylovSolver with smoothed aggregation AMG"

    # Set backend
    parameters["linear_algebra_backend"] = "PETSc"

    def build_nullspace(V, x):
        """Function to build null space for 2D elasticity"""

        # Create list of vectors for null space
        nullspace_basis = [x.copy() for i in range(3)]

        # Build translational null space basis
        V.sub(0).dofmap().set(nullspace_basis[0], 1.0)
        V.sub(1).dofmap().set(nullspace_basis[1], 1.0)

        # Build rotational null space basis
        V.sub(0).set_x(nullspace_basis[2], -1.0, 1)
        V.sub(1).set_x(nullspace_basis[2], 1.0, 0)

        for x in nullspace_basis:
            x.apply("insert")

        null_space = VectorSpaceBasis(nullspace_basis)
        null_space.orthonormalize()
        return null_space

    def amg_solve(N, method):
        # Elasticity parameters
        E = 1.0e9
        nu = 0.3
        mu = E/(2.0*(1.0 + nu))
        lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

        # Stress computation
        def sigma(v):
            return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(2)

        # Define problem
        mesh = UnitSquareMesh(N, N)
        V = VectorFunctionSpace(mesh, 'Lagrange', 1)
        bc = DirichletBC(V, Constant((0.0, 0.0)),
                         lambda x, on_boundary: on_boundary)
        u = TrialFunction(V)
        v = TestFunction(V)

        # Forms
        a, L = inner(sigma(u), grad(v))*dx, dot(Constant((1.0, 1.0)), v)*dx

        # Assemble linear algebra objects
        A, b = assemble_system(a, L, bc)

        # Create solution function
        u = Function(V)

        # Create near null space basis and orthonormalize
        null_space = build_nullspace(V, u.vector())

        # Attached near-null space to matrix
        as_backend_type(A).set_near_nullspace(null_space)

        # Test that basis is orthonormal
        assert null_space.is_orthonormal()

        # Create PETSC smoothed aggregation AMG preconditioner, and
        # create CG solver
        pc = PETScPreconditioner(method)
        solver = PETScKrylovSolver("cg", pc)

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


@skip_if_not_PETSc
def test_krylov_reuse_pc():
    "Test preconditioner re-use with PETScKrylovSolver"

    # Define problem
    mesh = UnitSquareMesh(8, 8)
    V = FunctionSpace(mesh, 'Lagrange', 1)
    bc = DirichletBC(V, Constant(0.0), lambda x, on_boundary: on_boundary)
    u = TrialFunction(V)
    v = TestFunction(V)

    # Forms
    a, L = inner(grad(u), grad(v))*dx, dot(Constant(1.0), v)*dx

    A, P = PETScMatrix(), PETScMatrix()
    b = PETScVector()

    # Assemble linear algebra objects
    assemble(a, tensor=A)
    assemble(a, tensor=P)
    assemble(L, tensor=b)

    # Apply boundary conditions
    bc.apply(A)
    bc.apply(P)
    bc.apply(b)

    # Create Krysolv solver and set operators
    solver = PETScKrylovSolver("gmres", "bjacobi")
    solver.set_operators(A, P)

    # Solve
    x = PETScVector()
    num_iter_ref = solver.solve(x, b)

    # Change preconditioner matrix (bad matrix) and solve (PC will be
    # updated)
    a_p = u*v*dx
    assemble(a_p, tensor=P)
    bc.apply(P)
    x = PETScVector()
    num_iter_mod = solver.solve(x, b)
    assert num_iter_mod > num_iter_ref

    # Change preconditioner matrix (good matrix) and solve (PC will be
    # updated)
    a_p = a
    assemble(a_p, tensor=P)
    bc.apply(P)
    x = PETScVector()
    num_iter = solver.solve(x, b)
    assert num_iter == num_iter_ref

    # Change preconditioner matrix (bad matrix) and solve (PC will not
    # be updated)
    solver.set_reuse_preconditioner(True)
    a_p = u*v*dx
    assemble(a_p, tensor=P)
    bc.apply(P)
    x = PETScVector()
    num_iter = solver.solve(x, b)
    assert num_iter == num_iter_ref

    # Update preconditioner (bad PC, will increase iteration count)
    solver.set_reuse_preconditioner(False)
    x = PETScVector()
    num_iter = solver.solve(x, b)
    assert num_iter == num_iter_mod


def test_krylov_tpetra():
    if not has_linear_algebra_backend("Tpetra"):
        return

    mesh = UnitCubeMesh(10, 10, 10)
    Q = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(Q)
    u = TrialFunction(Q)
    a = dot(grad(u), grad(v))*dx
    L = v*dx

    def bound(x):
        return x[0] == 0

    bc = DirichletBC(Q, Constant(0.0), bound)

    A = TpetraMatrix()
    b = TpetraVector()
    assemble(a, A)
    assemble(L, b)
    bc.apply(A)
    bc.apply(b)

    mp = MueluPreconditioner()
    mlp = mp.parameters['muelu']
    mlp['verbosity'] = 'none'
    mlp.add("max_levels", 10)
    mlp.add("coarse:_max_size", 10)
    mlp.add("coarse:_type", "KLU2")
    mlp.add("multigrid_algorithm", "sa")
    mlp.add("aggregation:_type", "uncoupled")
    mlp.add("aggregation:_min_agg_size", 3)
    mlp.add("aggregation:_max_agg_size", 7)

    pre_paramList = Parameters("smoother:_pre_params")
    pre_paramList.add("relaxation:_type", "Symmetric Gauss-Seidel")
    pre_paramList.add("relaxation:_sweeps", 1)
    pre_paramList.add("relaxation:_damping_factor", 0.6)
    mlp.add("smoother:_pre_type", "RELAXATION")
    mlp.add(pre_paramList)

    post_paramList = Parameters("smoother:_post_params")
    post_paramList.add("relaxation:_type", "Symmetric Gauss-Seidel")
    post_paramList.add("relaxation:_sweeps", 1)
    post_paramList.add("relaxation:_damping_factor", 0.9)
    mlp.add("smoother:_post_type", "RELAXATION")
    mlp.add(post_paramList)

    solver = BelosKrylovSolver("cg", mp)
    solver.parameters['relative_tolerance'] = 1e-8
    solver.parameters['monitor_convergence'] = False
    solver.parameters['belos'].add("Maximum_Iterations", 150)

    solver.set_operator(A)

    u = TpetraVector()
    n_iter = solver.solve(u, b)

    # Number of iterations should be around 15
    assert n_iter < 50
