"""Unit tests for class SystemAssembler"""

# Copyright (C) 2011-2013 Garth N. Wells, 2013 Jan Blechta
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
# Last changed: 2013-06-02

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

        A_frobenius_norm = 4.3969686527582512
        b_l2_norm = 0.95470326978246278

        # Assemble system
        A, b = assemble_system(a, L)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        # SystemAssembler construction
        assembler = SystemAssembler(a, L)

        # Test SystemAssembler for LHS and RHS
        A = Matrix()
        b = Vector()
        assembler.assemble(A, b)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        A = Matrix()
        b = Vector()

        assembler.assemble(A)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)

        assembler.assemble(b)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

    def test_cell_assembly_bc(self):

        mesh = UnitCubeMesh(4, 4, 4)
        V = FunctionSpace(mesh, "Lagrange", 1)
        bc = DirichletBC(V, 1.0, "on_boundary")

        u, v = TrialFunction(V), TestFunction(V)
        f = Constant(10)

        a = inner(grad(u), grad(v))*dx
        L = inner(f, v)*dx

        A_frobenius_norm = 96.847818767384
        b_l2_norm =  96.564760289080

        # Assemble system
        A, b = assemble_system(a, L, bc)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        # Create assembler
        assembler = SystemAssembler(a, L, bc)

        # Test for assembling A and b via assembler object
        A, b = Matrix(), Vector()
        assembler.assemble(A, b)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        # Assemble LHS only (first time)
        A = Matrix()
        assembler.assemble(A)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)

        # Assemble LHS only (second time)
        assembler.assemble(A)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)

        # Assemble RHS only (first time)
        b = Vector()
        assembler.assemble(b)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        # Assemble RHS only (second time time)
        assembler.assemble(b)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        # Do not reset sparsity
        assemble.reset_sparsity = False
        assembler.assemble(A)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        assembler.assemble(b)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)


    def test_facet_assembly(self):

        mesh = UnitSquareMesh(24, 24)

        if MPI.num_processes(mesh.mpi_comm()) > 1:
            print "FIXME: This unit test does not work in parallel, skipping"
            return

        V = FunctionSpace(mesh, "DG", 1)

        # Define test and trial functions
        v = TestFunction(V)
        u = TrialFunction(V)

        # Define normal component, mesh size and right-hand side
        n = FacetNormal(mesh)
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

        # Test SystemAssembler
        assembler = SystemAssembler(a, L)
        A = Matrix()
        b = Vector()

        assembler.assemble(A, b)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)

        A = Matrix()
        b = Vector()

        assembler.assemble(A)
        self.assertAlmostEqual(A.norm("frobenius"), A_frobenius_norm, 10)

        assembler.assemble(b)
        self.assertAlmostEqual(b.norm("l2"), b_l2_norm, 10)


    def test_incremental_assembly(self):

        for f in [Constant(0.0), Constant(1e4)]:

            # Laplace/Poisson problem
            mesh = UnitSquareMesh(20, 20)
            V = FunctionSpace(mesh, 'CG', 1)
            u, v = TrialFunction(V), TestFunction(V)
            a, L = inner(grad(u), grad(v))*dx, f*v*dx
            uD = Expression("42.0*(2.0*x[0]-1.0)")
            bc = DirichletBC(V, uD, "on_boundary")

            # Initialize initial guess by some number
            u = Function(V)
            x = u.vector()
            x[:] = 30.0
            u.update()

            # Assemble incremental system
            assembler = SystemAssembler(a, -L, bc)
            A, b = Matrix(), Vector()
            assembler.assemble(A, b, x)

            # Solve for (negative) increment
            Dx = Vector(x)
            Dx.zero()
            solve(A, Dx, b)

            # Update solution
            x[:] -= Dx[:]
            u.update()

            # Check solution
            u_true = Function(V)
            solve(a == L, u_true, bc)
            u.vector()[:] -= u_true.vector()[:]
            u.update()
            error = norm(u.vector(), 'linf')
            self.assertAlmostEqual(error, 0.0)


if __name__ == "__main__":
    print ""
    print "Testing class SystemAssembler"
    print "-----------------------------"
    unittest.main()
