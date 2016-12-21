"""Unit tests for geometric multigrid via PETSc"""

# Copyright (C) 2016 Patrick E. Farrell and Garth N. Wells
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


from dolfin import *
import pytest
from dolfin_utils.test import skip_if_not_PETSc, skip_if_not_petsc4py, pushpop_parameters

@skip_if_not_petsc4py
def test_mg_solver_laplace(pushpop_parameters):

    parameters["linear_algebra_backend"] = "PETSc"

    # Create meshes and function spaces
    meshes = [UnitSquareMesh(N, N) for N in [16, 32, 64]]
    V = [FunctionSpace(mesh, "Lagrange", 1) for mesh in meshes]

    # Create variational problem on fine grid
    u, v = TrialFunction(V[-1]), TestFunction(V[-1])
    a = dot(grad(u), grad(v))*dx
    L = v*dx
    bc = DirichletBC(V[-1], Constant(0.0), "on_boundary")
    A, b = assemble_system(a, L, bc)

    # Create collection of PETSc DM objects
    dm_collection = PETScDMCollection(V)

    # Create PETSc Krylov solver and set operator
    solver = PETScKrylovSolver()
    solver.set_operator(A)

    # Set PETSc solver type
    PETScOptions.set("ksp_type", "richardson")
    PETScOptions.set("pc_type", "mg")

    # Set PETSc MG type and levels
    PETScOptions.set("pc_mg_levels", len(V))
    PETScOptions.set("pc_mg_galerkin")

    # Set smoother
    PETScOptions.set("mg_levels_ksp_type", "chebyshev")
    PETScOptions.set("mg_levels_pc_type", "jacobi")

    # Set tolerance and monitor residual
    PETScOptions.set("ksp_monitor_true_residual")
    PETScOptions.set("ksp_atol", 1.0e-12)
    PETScOptions.set("ksp_rtol", 1.0e-12)
    solver.set_from_options()

    # Get fine grid DM and ttach fine grid DM to solver
    solver.set_dm(dm_collection.get_dm(-1))
    solver.set_dm_active(False)

    # Solve
    x = PETScVector()
    solver.solve(x, b)

    # Check multigrid solution against LU solver solution
    solver = LUSolver(A)
    x_lu = Vector()
    solver.solve(x_lu, b)
    assert round((x - x_lu).norm("l2"), 10) == 0


@skip_if_not_petsc4py
def xtest_mg_solver_stokes(pushpop_parameters):

    parameters["linear_algebra_backend"] = "PETSc"

    mesh0 = UnitCubeMesh(2, 2, 2)
    mesh1 = UnitCubeMesh(4, 4, 4)
    mesh2 = UnitCubeMesh(8, 8, 8)

    Ve = VectorElement("CG", mesh0.ufl_cell(), 2)
    Qe = FiniteElement("CG", mesh0.ufl_cell(), 1)
    Ze = MixedElement([Ve, Qe])

    Z0 = FunctionSpace(mesh0, Ze)
    Z1 = FunctionSpace(mesh1, Ze)
    Z2 = FunctionSpace(mesh2, Ze)
    W  = Z2

    # Boundaries
    def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
    def left(x, on_boundary): return x[0] < DOLFIN_EPS
    def top_bottom(x, on_boundary):
        return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

    # No-slip boundary condition for velocity
    noslip = Constant((0.0, 0.0, 0.0))
    bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

    # Inflow boundary condition for velocity
    inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"), degree=2)
    bc1 = DirichletBC(W.sub(0), inflow, right)

    # Collect boundary conditions
    bcs = [bc0, bc1]

    # Define variational problem
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)
    f = Constant((0.0, 0.0, 0.0))
    a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx
    L = inner(f, v)*dx

    # Form for use in constructing preconditioner matrix
    b = inner(grad(u), grad(v))*dx + p*q*dx

    # Assemble system
    A, bb = assemble_system(a, L, bcs)

    # Assemble preconditioner system
    P, btmp = assemble_system(b, L, bcs)

    spaces = [Z0, Z1, Z2]
    dm_collection = PETScDMCollection(spaces)

    solver = PETScKrylovSolver()
    solver.set_operators(A, P)

    PETScOptions.set("ksp_type", "gcr")
    PETScOptions.set("pc_type", "mg")
    PETScOptions.set("pc_mg_levels", 3)
    PETScOptions.set("pc_mg_galerkin")
    PETScOptions.set("ksp_monitor_true_residual")

    PETScOptions.set("ksp_atol", 1.0e-10)
    PETScOptions.set("ksp_rtol", 1.0e-10)
    solver.set_from_options()

    from petsc4py import PETSc

    ksp = solver.ksp()

    ksp.setDM(dm_collection.dm())
    ksp.setDMActive(False)

    x = PETScVector()
    solver.solve(x, bb)

    # Check multigrid solution against LU solver
    solver = LUSolver(A)
    x_lu = Vector()
    solver.solve(x_lu, bb)
    assert round((x - x_lu).norm("l2"), 10) == 0
