"""Unit tests for Dirichlet boundary conditions"""

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
# Modified by Kent-Andre Mardal 2011
# Modified by Anders Logg 2011
#
# First added:  2011-09-19
# Last changed: 2011-09-19

import unittest
import numpy
from dolfin import *

class DirichletBCTest(unittest.TestCase):

    def test_director_lifetime(self):
        """Test for any problems with objects with directors going out
        of scope"""

        class Boundary(SubDomain):
            def inside(self, x, on_boundary): return on_boundary

        class BoundaryFunction(Expression):
            def eval(self, values, x): values[0] = 1.0

        mesh = UnitSquare(8, 8)
        V = FunctionSpace(mesh, "Lagrange", 1)
        v, u = TestFunction(V), TrialFunction(V)
        A = assemble(v*u*dx)
        bc = DirichletBC(V, BoundaryFunction(), Boundary())

        bc.apply(A)

    def test_get_values(self):
        mesh = UnitSquare(8, 8)
        dofs = numpy.zeros(3, dtype="I")

        def upper(x, on_boundary):
            return x[1] > 0.5 + DOLFIN_EPS

        V = FunctionSpace(mesh, "CG", 1)
        bc = DirichletBC(V, 0.0, upper)
        bc_values = bc.get_boundary_values()

        for cell in cells(mesh):
            V.dofmap().tabulate_dofs(dofs, cell)
            coords = V.dofmap().tabulate_coordinates(cell)
            for i, dof in enumerate(dofs):
                if upper(coords[i, :], None):
                    self.assertTrue(dofs[i] in bc_values)
                    self.assertAlmostEqual(bc_values[dofs[i]], 0.0)
                else:
                    self.assertTrue(dofs[i] not in bc_values)

    def test_meshdomain_bcs(self):
        """Test application of Dirichlet boundary conditions stored as
        part of the mesh. This test is also a compatibility test for
        VMTK."""

        mesh = Mesh("../../../../data/meshes/aneurysm.xml.gz")
        V = FunctionSpace(mesh, "CG", 1)

        u = TrialFunction(V)
        v = TestFunction(V)

        f = Constant(0)
        u1 = Constant(1)
        u2 = Constant(2)
        u3 = Constant(3)

        bc1 = DirichletBC(V, u1, 1)
        bc2 = DirichletBC(V, u2, 2)
        bc3 = DirichletBC(V, u3, 3)

        bcs = [bc1, bc2, bc3]

        a = inner(grad(u), grad(v))*dx
        L = f*v*dx

        u = Function(V)
        solve(a == L, u, bcs)

        self.assertAlmostEqual(u.vector().norm("l2"), 98.9500304934, 10)

if __name__ == "__main__":
    print ""
    print "Testing Dirichlet boundary conditions"
    print "------------------------------------------------"
    unittest.main()
