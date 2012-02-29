"""Unit tests for class SymmetricAssembler"""

# Copyright (C) 2011 Garth N. Wells
# Copyright (C) 2012 Joachim B. Haga
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
# First added:  2012-01-01 (modified from SystemAssembler.py by jobh@simula.no)

import unittest
import numpy
from dolfin import *

class TestSymmetricAssembler(unittest.TestCase):

    def _check_against_reference(self, a, L, bc):

        # Assemble LHS using symmetric assembler
        A, A_n = symmetric_assemble(a, bcs=bc)

        # Assemble LHS using regular assembler
        A_ref = assemble(a, bcs=bc)

        # Check that the symmetric assemble matches the reference
        N = A + A_n - A_ref
        self.assertAlmostEqual(N.norm("frobenius"), 0.0, 10)

        # Check that A is symmetric
        X = assemble(L) # just to get the size
        X.set_local(numpy.random.random(X.local_size()))
        AT_X = Vector()
        A.transpmult(X, AT_X)
        N = A*X - AT_X
        self.assertAlmostEqual(N.norm("l2"), 0.0, 10)

    def test_cell_assembly(self):

        mesh = UnitCube(4, 4, 4)
        V = VectorFunctionSpace(mesh, "CG", 1)

        v = TestFunction(V)
        u = TrialFunction(V)
        f = Constant((10, 20, 30))

        def epsilon(v):
            return 0.5*(grad(v) + grad(v).T)

        a = inner(epsilon(v), epsilon(u))*dx
        L = inner(v, f)*dx

        # Define boundary condition
        def boundary(x):
            return near(x[0], 0.0) or near(x[0], 1.0)
        u0 = Constant((1.0, 2.0, 3.0))
        bc = DirichletBC(V, u0, boundary)

        self._check_against_reference(a, L, bc)

    def test_facet_assembly(self):

        if MPI.num_processes() > 1:
            print "FIXME: This unit test does not work in parallel, skipping"
            return

        mesh = UnitSquare(24, 24)
        V = FunctionSpace(mesh, "CG", 1)

        # Define test and trial functions
        v = TestFunction(V)
        u = TrialFunction(V)

        # Define normal component, mesh size and right-hand side
        n = V.cell().n
        h = CellSize(mesh)
        h_avg = (h('+')+h('-'))/2
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

        # Define boundary condition
        def boundary(x):
            return near(x[0], 0.0) or near(x[0], 1.0)
        u0 = Constant(1.0)
        bc = DirichletBC(V, u0, boundary, method="pointwise")

        self._check_against_reference(a, L, bc)

    def test_subdomain_assembly_meshdomains(self):
        "Test assembly over subdomains with markers stored as part of mesh"

        # Create a mesh of the unit cube
        mesh = UnitCube(4, 4, 4)

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

        # Define subdomain for right of x = 0.5
        class S1(SubDomain):
            def inside(self, x, inside):
                return x[0] >= 0.5 + DOLFIN_EPS

        # Mark mesh
        f0 = F0()
        f1 = F1()
        f2 = F2()
        s0 = S0()
        s1 = S1()
        f0.mark_facets(mesh, 0)
        f1.mark_facets(mesh, 1)
        f2.mark_facets(mesh, 2)
        s0.mark_cells(mesh, 0)
        s1.mark_cells(mesh, 1)

        # Define test and trial functions
        V = FunctionSpace(mesh, "CG", 1)
        u = TrialFunction(V)
        v = TestFunction(V)

        # FIXME: If the Z terms are not present, PETSc will claim:
        #    Object is in wrong state!
        #    Matrix is missing diagonal entry in row 124!
        Z = Constant(0.0)

        # Define forms on marked subdomains
        a0 = 1*u*v*dx(0) + 2*u*v*ds(0) + 3*u*v*ds(1) + 4*u*v*ds(2) + Z*u*v*dx(1)
        L0 = 1*v*dx(0) + 2*v*ds(0) + 3*v*ds(1) + 4*v*ds(2)

        # Defined forms on unmarked subdomains (should be zero)
        a1 = 1*u*v*dx(2) + 2*u*v*ds(3) + Z*u*v*dx(0) + Z*u*v*dx(1)
        L1 = 1*v*dx(2) + 2*v*ds(3)

        # Define boundary condition
        def boundary(x):
            return near(x[0], 0.0) or near(x[0], 1.0)
        u0 = Constant(1.0)
        bc = DirichletBC(V, u0, boundary, method="pointwise")

        self._check_against_reference(a0, L0, bc)
        self._check_against_reference(a1, L1, bc)

if __name__ == "__main__":
    print ""
    print "Testing class SymmetricAssembler"
    print "-----------------------------"
    unittest.main()
