"""Unit tests for class SystemAssembler"""

# Copyright (C) 2011 Garth N. Wells
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
# Modified by Marie E. Rognes 2011
# Modified by Anders Logg 2011
#
# First added:  2011-10-04
# Last changed: 2011-10-04

import unittest
import numpy
from dolfin import *

class TestSystemAssembler(unittest.TestCase):

    def test_cell_assembly(self):

        mesh = UnitCubeMesh(4, 4, 4)
        V = VectorFunctionSpace(mesh, "DG", 1)

        v = TestFunction(V)
        u = TrialFunction(V)
        f = Constant((10, 20, 30))

        def epsilon(v):
            return 0.5*(grad(v) + grad(v).T)

        a = inner(epsilon(v), epsilon(u))*dx
        L = inner(v, f)*dx

        A_frobenius_norm =  4.3969686527582512
        b_l2_norm = 0.95470326978246278

        # Assemble system
        A, b = assemble_system(a, L)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

    def test_facet_assembly(self):

        if MPI.num_processes() > 1:
            print "FIXME: This unit test does not work in parallel, skipping"
            return

        mesh = UnitSquareMesh(24, 24)
        V = FunctionSpace(mesh, "DG", 1)

        # Define test and trial functions
        v = TestFunction(V)
        u = TrialFunction(V)

        # Define normal component, mesh size and right-hand side
        n = V.cell().n
        h = CellSize(mesh)
        h_avg = (h('+') + h('-'))/2
        f = Expression("500.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=1)

        # Define bilinear form
        a = dot(grad(v), grad(u))*dx \
            - dot(avg(grad(v)), jump(u, n))*dS \
            - dot(jump(v, n), avg(grad(u)))*dS \
            + 4.0/h_avg*dot(jump(v, n), jump(u, n))*dS \
            - dot(grad(v), u*n)*ds \
            - dot(v*n, grad(u))*ds \
            + 8.0/h*v*u*ds

        # Define linear form
        L = v*f*dx

        # Reference values
        A_frobenius_norm = 157.867392938645
        b_l2_norm = 1.48087142738768

        # Assemble system
        A, b = assemble_system(a, L)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

    def test_subdomain_assembly_meshdomains(self):
        "Test assembly over subdomains with markers stored as part of mesh"

        # Create a mesh of the unit cube
        mesh = UnitCubeMesh(4, 4, 4)

        # Define subdomains for 3 faces of the unit cube
        class F0(SubDomain):
            def inside(self, x, inside):
                return near(x[0], 0.0)
        class F1(SubDomain):
            def inside(self, x, inside):
                return near(x[1], 0.0)
        class F2(SubDomain):
            def inside(self, x, inside):
                return near(x[2], 0.0)

        # Define subdomain for left of x = 0.5
        class S0(SubDomain):
            def inside(self, x, inside):
                return x[0] < 0.5 + DOLFIN_EPS

        # Mark mesh
        f0 = F0()
        f1 = F1()
        f2 = F2()
        s0 = S0()
        f0.mark_facets(mesh, 0)
        f1.mark_facets(mesh, 1)
        f2.mark_facets(mesh, 2)
        s0.mark_cells(mesh, 0)

        # Define test and trial functions
        V = FunctionSpace(mesh, "Lagrange", 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        # Define forms on marked subdomains
        a0 = 1*u*v*dx(0) + 2*u*v*ds(0) + 3*u*v*ds(1) + 4*u*v*ds(2)
        L0 = 1*v*dx(0) + 2*v*ds(0) + 3*v*ds(1) + 4*v*ds(2)

        # Defined forms on unmarked subdomains (should be zero)
        a1 = 1*u*v*dx(1) + 2*u*v*ds(3)
        L1 = 1*v*dx(1) + 2*v*ds(3)

        # Used for computing reference values
        #A0 = assemble(a0)
        #b0 = assemble(L0)
        #A1 = assemble(a1)
        #b1 = assemble(L1)

        # Assemble system
        A0, b0 = assemble_system(a0, L0)
        A1, b1 = assemble_system(a1, L1)

        # Assemble and check values
        self.assertAlmostEqual(A0.norm("frobenius"), 0.693043954566, 10)
        self.assertAlmostEqual(b0.norm("l2"),        1.28061997552,  10)
        self.assertAlmostEqual(A1.norm("frobenius"), 0.0,  10)
        self.assertAlmostEqual(b1.norm("l2"),        0.0,  10)

if __name__ == "__main__":
    print ""
    print "Testing class SystemAssembler"
    print "-----------------------------"
    unittest.main()
