"""Unit tests for Dirichlet boundary conditions"""

# Copyright (C) 2011 Anders Logg and Kent-Andre Mardal
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
# First added:  2011-09-19
# Last changed: 2011-09-19

import unittest
from dolfin import *

class DirichletBCTest(unittest.TestCase):

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
