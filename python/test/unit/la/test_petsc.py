"""Unit tests for parts of the PETSc interface not tested via the
GenericFoo interface

"""

# Copyright (C) 2015-2017 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import (PETScVector, PETScMatrix, PETScLUSolver,
                    PETScKrylovSolver, UnitSquareMesh, TrialFunction,
                    TestFunction, MPI,
                    FunctionSpace, assemble, Constant, dx, parameters)
from dolfin_utils.test import (skip_if_not_PETSc,
                               skip_if_not_petsc4py,
                               pushpop_parameters)


@skip_if_not_PETSc
def test_vector():
    "Test PETScVector interface"

    prefix = "my_vector_"
    x = PETScVector(MPI.comm_world)
    x.set_options_prefix(prefix)

    assert x.get_options_prefix() == prefix
    x.init(300)
    assert x.get_options_prefix() == prefix


@skip_if_not_PETSc
def test_krylov_solver_norm_type():
    """Check setting of norm type used in testing for convergence by
    PETScKrylovSolver

    """

    norm_type = (PETScKrylovSolver.norm_type.default_norm,
                 PETScKrylovSolver.norm_type.natural,
                 PETScKrylovSolver.norm_type.preconditioned,
                 PETScKrylovSolver.norm_type.none,
                 PETScKrylovSolver.norm_type.unpreconditioned)

    for norm in norm_type:
        # Solve a system of equations
        mesh = UnitSquareMesh(4, 4)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u, v = TrialFunction(V), TestFunction(V)
        a = u*v*dx
        L = Constant(1.0)*v*dx
        A, b = assemble(a), assemble(L)

        solver = PETScKrylovSolver("cg")
        solver.parameters["maximum_iterations"] = 2
        solver.parameters["error_on_nonconvergence"] = False
        solver.set_norm_type(norm)
        solver.set_operator(A)
        solver.solve(b.copy(), b)
        solver.get_norm_type()

        if norm is not PETScKrylovSolver.norm_type.default_norm:
            assert solver.get_norm_type() == norm


@skip_if_not_PETSc
def test_krylov_solver_options_prefix(pushpop_parameters):
    "Test set/get PETScKrylov solver prefix option"

    # Set backend
    parameters["linear_algebra_backend"] = "PETSc"

    # Prefix
    prefix = "test_foo_"

    # Create solver and set prefix
    solver = PETScKrylovSolver()
    solver.set_options_prefix(prefix)

    # Check prefix (pre solve)
    assert solver.get_options_prefix() == prefix

    # Solve a system of equations
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    a, L = u*v*dx, Constant(1.0)*v*dx
    A, b = assemble(a), assemble(L)
    solver.set_operator(A)
    solver.solve(b.copy(), b)

    # Check prefix (post solve)
    assert solver.get_options_prefix() == prefix


@skip_if_not_PETSc
def test_options_prefix(pushpop_parameters):
    "Test set/get prefix option for PETSc objects"

    def run_test(A, init_function):
        # Prefix
        prefix = "test_foo_"

        # Set prefix
        A.set_options_prefix(prefix)

        # Get prefix (should be empty since vector has been initialised)
        # assert not A.get_options_prefix()

        # Initialise vector
        init_function(A)

        # Check prefix
        assert A.get_options_prefix() == prefix

        # Try changing prefix post-intialisation (should throw error)
        # with pytest.raises(RuntimeError):
        #     A.set_options_prefix("test")

    # Test vector
    def init_vector(x):
        x.init(100)
    x = PETScVector(MPI.comm_world)
    run_test(x, init_vector)

    # Test matrix
    def init_matrix(A):
        mesh = UnitSquareMesh(12, 12)
        V = FunctionSpace(mesh, "Lagrange", 1)
        u, v = TrialFunction(V), TestFunction(V)
        assemble(u*v*dx, tensor=A)
    A = PETScMatrix()
    run_test(A, init_matrix)

    # FIXME: Need to straighten out the calls to PETSc
    # FooSetFromOptions calls in the solver wrapers

    # Test solvers
    # def init_solver(A, solver):
    #     A = PETScMatrix()
    #     init_matrix(A)
    #     solver.set_operator(A)

    # Test Krylov solver
    # solver = PETScKrylovSolver()
    # run_test(solver, init_solver)

    # Test KY solver
    # solver = PETScLUSolver()
    # run_test(solver, init_solver)


@skip_if_not_petsc4py
def test_lu_cholesky():
    """Test that PETScLUSolver selects LU or Cholesky solver based on
    symmetry of matrix operator.

    """

    from petsc4py import PETSc

    mesh = UnitSquareMesh(MPI.comm_world, 12, 12)
    V = FunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    A = PETScMatrix(mesh.mpi_comm())
    assemble(Constant(1.0)*u*v*dx, tensor=A)

    # Check that solver type is LU
    solver = PETScLUSolver(mesh.mpi_comm(), A, "petsc")
    pc_type = solver.ksp().getPC().getType()
    assert pc_type == "lu"

    # Set symmetry flag
    A.mat().setOption(PETSc.Mat.Option.SYMMETRIC, True)

    # Check symmetry flags
    symm = A.mat().isSymmetricKnown()
    assert symm[0] == True
    assert symm[1] == True

    # Check that solver type is Cholesky since matrix has now been
    # marked as symmetric
    solver = PETScLUSolver(mesh.mpi_comm(), A, "petsc")
    pc_type = solver.ksp().getPC().getType()
    assert pc_type == "cholesky"

    # Re-assemble, which resets symmetry flag
    assemble(Constant(1.0)*u*v*dx, tensor=A)
    solver = PETScLUSolver(mesh.mpi_comm(), A, "petsc")
    pc_type = solver.ksp().getPC().getType()
    assert pc_type == "lu"
