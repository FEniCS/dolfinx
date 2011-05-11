"Demonstrating the assembly on a  restricted FunctionSpace."

# Copyright (C) 2008 Kent-Andre Mardal
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2009-02-09
# Last changed: 2009-02-09

from dolfin import *
import unittest


class LeftSide(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < 0.5 + DOLFIN_EPS)

class RestrictionTest(unittest.TestCase): 
    def test(self):

        mesh = UnitSquare(2,2)
        mesh_function = MeshFunction("bool", mesh, mesh.topology().dim())
        subdomain = LeftSide()
        for cell in cells(mesh): 
            p = cell.midpoint()
            mesh_function.array()[cell.index()] = int(subdomain.inside(p, False))

        V = FunctionSpace(mesh, "Lagrange", 1)
        fv = Function(V, cppexpr='1.0')
        vv = TestFunction(V)
        uv = TrialFunction(V)

        m = uv*vv*dx
        L0 = fv*vv*dx 
        M = assemble(m)
        b = assemble(L0)

        self.assertEqual(M.size(0), 9) 
        self.assertEqual(M.size(1), 9) 

        W = V.restriction(mesh_function) 
        fw = Function(W, cppexpr='1.0')
        vw = TestFunction(W)
        uw = TrialFunction(W)

        m = uw*vw*dx
        L1 = fw*vw*dx 
        M = assemble(m)
        b = assemble(L1)

        self.assertEqual(M.size(0), 6) 
        self.assertEqual(M.size(1), 6) 


        m = uw*vv*dx
        M = assemble(m)
        self.assertEqual(M.size(0), 9) 
        self.assertEqual(M.size(1), 6) 

        m = uv*vw*dx
        M = assemble(m)
        self.assertEqual(M.size(0), 6) 
        self.assertEqual(M.size(1), 9) 


if __name__ == "__main__":
    print ""
    print "Testing restricted FunctionSpace."
    print "------------------------------------------------"
    unittest.main()



