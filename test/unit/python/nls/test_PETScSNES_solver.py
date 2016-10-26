#!/usr/bin/env py.test

"""Unit test for the SNES nonlinear solver"""

# Copyright (C) 2012 Patrick E. Farrell
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
#
# First added:  2012-10-17
# Last changed: 2016-10-26

from __future__ import print_function

"""Solve the Yamabe PDE which arises in the differential geometry of
general relativity. http://arxiv.org/abs/1107.0360.

The Yamabe equation is highly nonlinear and supports many
solutions. However, only one of these is of physical relevance -- the
positive solution.

This unit test demonstrates the capability of the SNES solver to
accept bounds on the resulting solution. The plain Newton method
converges to an unphysical negative solution, while the SNES solution
with {sign: nonnegative} converges to the physical positive solution.

An alternative interface to SNESVI allows the user to set explicitly
more complex bounds as GenericVectors or Function.
"""

from dolfin import *
import pytest
import os

from dolfin_utils.test import *

parameter_degree = set_parameters_fixture("form_compiler.quadrature_degree", \
                                          [5])
parameter_backend = set_parameters_fixture("linear_algebra_backend", ["PETSc"])

@fixture
def mesh(datadir):
    return Mesh(os.path.join(datadir, "doughnut.xml"))

@fixture
def V(mesh):
    return FunctionSpace(mesh, "CG", 1)

@fixture
def bcs(V):
    return [DirichletBC(V, 1.0, "on_boundary")]

@fixture
def u(V):
    u = Function(V)
    u.interpolate(Constant(-1000.0))
    return u

@fixture
def v(V):
    return TestFunction(V)

@fixture
def F(u, v, mesh):
    x = SpatialCoordinate(mesh)
    r = sqrt(x[0]**2 + x[1]**2)
    rho = 1.0/r**3

    return (8*inner(grad(u), grad(v))*dx + rho * inner(u**5, v)*dx \
            + (-1.0/8.0)*inner(u, v)*dx)

@fixture
def J(V, u, F):
    du = TrialFunction(V)
    return derivative(F, u, du)

@fixture
def lb(V):
    return interpolate(Constant(0.), V)

@fixture
def ub(V):
    return interpolate(Constant(100.), V)

@fixture
def newton_solver_parameters():
    return{"nonlinear_solver": "newton",
            "newton_solver": {"linear_solver": "lu",
            "maximum_iterations": 100,
            "report": False}}

@fixture
def snes_solver_parameters_sign():
    return {"nonlinear_solver": "snes",
            "snes_solver": {"linear_solver": "lu",
            "maximum_iterations": 100,
            "sign": "nonnegative",
            "report": True}}

@fixture
def snes_solver_parameters_bounds():
    return {"nonlinear_solver": "snes",
            "snes_solver": {"linear_solver": "lu",
            "maximum_iterations": 100,
            "sign": "default",
            "report": True}}

@skip_if_not_PETSc
def test_snes_solver(F, bcs, u, snes_solver_parameters_sign, parameter_degree,\
                     parameter_backend):
    u.interpolate(Constant(-1000.0))
    solve(F == 0, u, bcs, solver_parameters=snes_solver_parameters_sign)
    assert u.vector().min() >= 0


@skip_if_not_PETSc
def test_newton_solver(F, u, bcs, newton_solver_parameters, parameter_degree,\
                       parameter_backend):
    u.interpolate(Constant(-1000.0))
    solve(F == 0, u, bcs, solver_parameters=newton_solver_parameters)
    assert u.vector().min() < 0


@skip_if_not_PETSc
def test_snes_solver_bound_functions(F, u, bcs, J, \
                                     snes_solver_parameters_bounds,
                                     lb, ub, parameter_degree, \
                                     parameter_backend):
    u.interpolate(Constant(-1000.0))
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    problem.set_bounds(lb, ub)

    solver  = NonlinearVariationalSolver(problem)
    solver.parameters.update(snes_solver_parameters_bounds)
    solver.solve()
    assert u.vector().min() >= 0


@skip_if_not_PETSc
def test_snes_solver_bound_vectors(F, u, bcs, J, snes_solver_parameters_bounds,
                                   lb, ub, parameter_degree, \
                                   parameter_backend):
    u.interpolate(Constant(-1000.0))
    problem = NonlinearVariationalProblem(F, u, bcs, J)
    problem.set_bounds(lb, ub)

    solver  = NonlinearVariationalSolver(problem)
    solver.parameters.update(snes_solver_parameters_bounds)
    solver.solve()
    assert u.vector().min() >= 0
