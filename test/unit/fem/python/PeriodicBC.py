"""Unit tests for Dirichlet boundary conditions"""

# Copyright (C) 2012 Garth N. Wells
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
# First added:  2012-08-18
# Last changed:

import unittest
import numpy
from dolfin import *

class PeriodicBoundary2(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]

class PeriodicBoundary3(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < DOLFIN_EPS and x[0] > -DOLFIN_EPS and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]
        y[2] = x[2]

class PeriodicBCTest(unittest.TestCase):

    def test_instantiation(self):
        """ A rudimentary test for instantiation"""
        mesh = UnitCube(8, 8, 8)
        V = FunctionSpace(mesh, "CG", 1)

        sub_domain = PeriodicBoundary3()
        bc0 = PeriodicBC(V, sub_domain)

        # Uncomment if/when PeriodicBC gets a copy constructor
        #bc1 = PeriodicBC(bc0)
        #self.assertTrue(bc0.function_space() == bc1.function_space())

    def test_instantiation_mixed_element(self):
        """ A rudimentary test for instantiation with mixed elements"""

        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        VV = V*V

        pbc = PeriodicBoundary2()
        bc  = PeriodicBC(VV, pbc)

    def test_director_lifetime(self):
        """Test for problems with objects with directors going out
        of scope"""

        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        bc = PeriodicBC(V, PeriodicBoundary2())

        # FIXME: need to wrap output from below function in Python
        #bc.compute_dof_pairs();

    def test_solution(self):
        """Test application Periodic boundary conditions by checking
        solution to a PDE."""

        # FIXME: This hack should be removed once periodic boundary
        # FIXME: conditions have been implemented properly
        if parameters["linear_algebra_backend"] == "Epetra":
            return

        # Create mesh and finite element
        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)

        class DirichletBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) and on_boundary)

        # Dirichlet boundary condition
        dirichlet_boundary = DirichletBoundary()
        bc0 = DirichletBC(V, 0.0, dirichlet_boundary)
        #periodic_boundary = PeriodicBoundary2()
        #bc1 = PeriodicBC(V, periodic_boundary)
        bc1 = PeriodicBC(V, PeriodicBoundary2())
        bcs = [bc0, bc1]

        # Define variational problem, linear formulation
        u, v = TrialFunction(V), TestFunction(V)
        f = Expression("sin(x[0])", degree=2)
        a = dot(grad(u), grad(v))*dx
        L = f*v*dx

        # Compute solution
        u = Function(V)
        solve(a == L, u, bcs)

        self.assertAlmostEqual(u.vector().norm("l2"), 0.3567245204026249, 10)

        # Define variational problem, nonlinear formulation
        u, v = Function(V), TestFunction(V)
        f = Expression("sin(x[0])", degree=2)
        a = dot(grad(u), grad(v))*dx
        L = f*v*dx
        F = a - L

        # Compute solution
        solve(F == 0, u, bcs)

        self.assertAlmostEqual(u.vector().norm("l2"), 0.3567245204026249, 10)

if __name__ == "__main__":
    print ""
    print "Testing Dirichlet boundary conditions"
    print "------------------------------------------------"
    unittest.main()
