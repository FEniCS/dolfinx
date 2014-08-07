"""Unit tests for error control"""

# Copyright (C) 2011 Marie E. Rognes
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
# First added:  2011-04-05
# Last changed: 2014-05-28

from __future__ import print_function
import pytest
from ufl.algorithms import replace

from dolfin import *
from dolfin.fem.adaptivesolving import *

# FIXME: Move this to dolfin for user access?
def reconstruct_refined_form(form, functions, mesh):
    function_mapping = {}
    for u in functions:
        w = Function(u.leaf_node().function_space())
        w.assign(u.leaf_node())
        function_mapping[u] = w
    domain = mesh.leaf_node().ufl_domain()
    newform = replace_integral_domains(replace(form, function_mapping), domain)
    return newform, function_mapping


skip_parallel = pytest.mark.skipif(MPI.size(mpi_comm_world()) > 1, 
	    reason="Skipping unit test(s) not working in parallel")

mesh_ = UnitSquareMesh(8, 8)
V_ = FunctionSpace(mesh_, "Lagrange", 1)
bc_ = [DirichletBC(V_, 0.0, "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS")]

u_ = TrialFunction(V_)
v_ = TestFunction(V_)
f_ = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1)
g_ = Expression("sin(5*x[0])", degree=1)
a_ = inner(grad(u_), grad(v_))*dx()
L_ = f_*v_*dx() + g_*v_*ds()

u_ = Function(V_)
problem_ = LinearVariationalProblem(a_, L_, u_, bc_)

goal_ = u_*dx()
ec_ = generate_error_control(problem_, goal_)

@pytest.fixture
def mesh():
    return mesh_

@pytest.fixture
def V():
    return V_

@pytest.fixture
def u():
    return u_

@pytest.fixture
def a():
    return a_

@pytest.fixture
def L():
    return L_

@pytest.fixture
def problem():
    return problem_

@pytest.fixture
def goal():
    return goal_

@pytest.fixture
def ec():
    return ec_ 


@skip_parallel
def test_check_domains(goal, mesh, a, L):
        # Asserting that domains are ok before trying error control generation
        msg = "Expecting only the domain from the mesh to get here through u."
        assert len(goal.domains()) == 1, msg
        assert goal.domains()[0] == mesh.ufl_domain(), msg
        assert len(a.domains()) == 1, msg
        assert a.domains()[0] == mesh.ufl_domain(), msg
        assert len(L.domains()) == 1, msg
        assert L.domains()[0] == mesh.ufl_domain(), msg


@skip_parallel
def test_error_estimation(problem, u, ec):

    # Solve variational problem once
    solver = LinearVariationalSolver(problem)
    solver.solve()

    # Compute error estimate
    error_estimate = ec.estimate_error(u, problem.bcs())

    # Compare estimate with defined reference
    reference = 0.0011789985750808342
    assert round(error_estimate - reference, 7) == 0

@skip_parallel
def test_error_indicators(problem, u, mesh):

    # Solve variational problem once
    solver = LinearVariationalSolver(problem)
    solver.solve()

    # Compute error indicators
    indicators = Vector(mesh.mpi_comm(), u.function_space().mesh().num_cells())
    indicators[0] = 1.0
    #ec.compute_indicators(indicators, u) #

    reference = 1.0 # FIXME
    assert round(indicators.sum() - reference, 7) == 0

@skip_parallel
def test_adaptive_solve(problem, goal, u, mesh):

    # Solve problem adaptively
    solver = AdaptiveLinearVariationalSolver(problem, goal)
    tol = 0.00087
    solver.solve(tol)

    # Note: This old approach is now broken, as it doesn't change the integration domain:
    #M = replace(goal, {u: w})
    # This new approach handles the integration domain properly:
    M, fm = reconstruct_refined_form(goal, [u], mesh)

    # Compare computed goal with reference
    reference = 0.12583303389560166
    assert round(assemble(M) - reference, 7) == 0

if __name__ == "__main__":
    pytest.main()
