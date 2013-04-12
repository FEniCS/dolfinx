"""Unit tests for Periodoc conditions"""

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
# Last changed: 2013-03-08

import unittest
import numpy
from dolfin import *

class PeriodicBoundary2(SubDomain):
    def __init__(self, tolerance=DOLFIN_EPS):
        SubDomain.__init__(self, tolerance)
        self.tol = tolerance
    def inside(self, x, on_boundary):
        return bool(x[0] < self.tol and x[0] > -self.tol and on_boundary)
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

        sub_domain = PeriodicBoundary3()
        mesh = UnitCubeMesh(8, 8, 8)
        V = FunctionSpace(mesh, "CG", 1, constrained_domain=sub_domain)

    def test_instantiation_mixed_element(self):
        """ A rudimentary test for instantiation with mixed elements"""

        pbc = PeriodicBoundary2()
        mesh = UnitSquareMesh(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)
        VV = V*V

    def test_instantiation_mixed_element_real(self):
        """ A rudimentary test for instantiation with mixed elements that include a real space"""

        pbc = PeriodicBoundary2()
        mesh = UnitSquareMesh(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)
        R = FunctionSpace(mesh, "Real", 0, constrained_domain=pbc)
        VV = V*R
        VV = R*V

    def test_instantiation_no_vertex_element_2D(self):
        """ A rudimentary test for instantiation for element that does
        not require number of vertices (2D)"""

        pbc = PeriodicBoundary2()
        mesh = UnitSquareMesh(8, 8)
        V = FunctionSpace(mesh, "BDM", 1, constrained_domain=pbc)

    def test_instantiation_no_vertex_element_3D(self):
        """ A rudimentary test for instantiation for element that does
        not require number of vertices (3D)"""

        pbc = PeriodicBoundary3()
        mesh = UnitCubeMesh(8, 8, 9)
        V = FunctionSpace(mesh, "BDM", 1, constrained_domain=pbc)

    def test_director_lifetime(self):
        """Test for problems with objects with directors going out
        of scope"""

        mesh = UnitSquareMesh(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=PeriodicBoundary2())


    def test_tolerance(self):
        """Test tolerance for matching periodic mesh entities"""
        shift = 0.0001
        mesh = UnitSquareMesh(8, 8)

        # Randomly perturb mesh vertex coordinates
        mesh_perturb = Mesh(mesh)
        import random
        for x in mesh_perturb.coordinates():
            x[0] += random.uniform(-shift, shift)
            x[1] += random.uniform(-shift, shift)

        pbc = PeriodicBoundary2()
        pbc_tol = PeriodicBoundary2(2*shift)

        for dim in range(mesh.geometry().dim()):
            periodic_pairs = PeriodicBoundaryComputation.compute_periodic_pairs(mesh, pbc, dim)
            num_periodic_pairs0 =  len(periodic_pairs)

            periodic_pairs = PeriodicBoundaryComputation.compute_periodic_pairs(mesh_perturb, pbc_tol, dim)
            num_periodic_pairs1 = len(periodic_pairs)
            self.assertEqual(num_periodic_pairs0, num_periodic_pairs1)


    def test_solution(self):
        """Test application Periodic boundary conditions by checking
        solution to a PDE."""

        # Create mesh and finite element
        mesh = UnitSquareMesh(8, 8)
        pbc = PeriodicBoundary2()
        V = FunctionSpace(mesh, "Lagrange", 1, constrained_domain=pbc)

        class DirichletBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return bool((x[1] < DOLFIN_EPS or x[1] > (1.0 - DOLFIN_EPS)) and on_boundary)

        # Dirichlet boundary condition
        dirichlet_boundary = DirichletBoundary()
        bc0 = DirichletBC(V, 0.0, dirichlet_boundary)
        bcs = [bc0]

        # Define variational problem, linear formulation
        u, v = TrialFunction(V), TestFunction(V)
        f = Expression("sin(x[0])", degree=2)
        a = dot(grad(u), grad(v))*dx
        L = f*v*dx

        # Compute solution
        u = Function(V)
        solve(a == L, u, bcs)

        self.assertAlmostEqual(u.vector().norm("l2"), 0.3368694028630991, 10)

        # Define variational problem, nonlinear formulation
        u, v = Function(V), TestFunction(V)
        f = Expression("sin(x[0])", degree=2)
        a = dot(grad(u), grad(v))*dx
        L = f*v*dx
        F = a - L

        # Compute solution
        solve(F == 0, u, bcs)

        self.assertAlmostEqual(u.vector().norm("l2"), 0.3368694028630991, 10)

if __name__ == "__main__":
    print ""
    print "Testing Dirichlet boundary conditions"
    print "------------------------------------------------"
    unittest.main()
