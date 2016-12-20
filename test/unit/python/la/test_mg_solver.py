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
from dolfin_utils.test import skip_if_not_PETSc, pushpop_parameters

@skip_if_not_PETSc
def test_mg_solver(pushpop_parameters):

    parameters["linear_algebra_backend"] = "PETSc"

    mesh0 = UnitSquareMesh(8, 8)
    mesh1 = UnitSquareMesh(16, 16)
    mesh2 = UnitSquareMesh(32, 132)

    V0 = FunctionSpace(mesh0, "Lagrange", 1)
    V1 = FunctionSpace(mesh1, "Lagrange", 1)
    V2 = FunctionSpace(mesh2, "Lagrange", 1)

    u, v = TrialFunction(V2), TestFunction(V2)
    A = assemble(Constant(1.0)*u*v*dx)
    b = assemble(Constant(1.0)*v*dx)

    norm = 13.0


    spaces = [V0, V1, V2]
    dm_collection = PETScDMCollection(spaces)

    solver = PETScKrylovSolver()
    solver.set_operator(A)

    PETScOptions.set("ksp_type", "richardson")
    PETScOptions.set("pc_type", "mg")
    PETScOptions.set("pc_mg_levels", 3)

    PETScOptions.set("pc_mg_galerkin")
    PETScOptions.set("ksp_monitor_true_residual")

    PETScOptions.set("ksp_atol", 1.0e-10)
    PETScOptions.set("ksp_rtol", 1.0e-10)
    solver.set_from_options()

    #KSPSetDMActive(ksp, PETSC_FALSE);
    #KSPSetDM(ksp, dm);

    from petsc4py import PETSc

    #ksp = PETSc.KSP(solver.ksp())
    ksp = solver.ksp()

    ksp.setDM(dm_collection.fine())
    ksp.setDMActive(False)

    x = PETScVector()
    solver.solve(x, b)
    #assert round(x.norm("l2") - norm, 10) == 0

    #solver = LUSolver(A)
    #x = Vector()
    #solver.solve(x, b)
    #assert round(x.norm("l2") - norm, 10) == 0

    #solver = LUSolver()
    #x = Vector()
    #solver.set_operator(A)
    #solver.solve(x, b)
    #assert round(x.norm("l2") - norm, 10) == 0
