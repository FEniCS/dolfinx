"""Unit tests for the FunctionSpace class"""

# Copyright (C) 2011 Johan Hake
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
# First added:  2011-09-21
# Last changed: 2011-09-21

import unittest
from dolfin import *

mesh = UnitCube(8, 8, 8)
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)
f = Function(V)
V2 = f.function_space()
g = Function(V)
W2 = g.function_space()

class Interface(unittest.TestCase):

    def test_equality(self):
        self.assertEqual(V, V)
        self.assertEqual(V, V2)
        self.assertEqual(W, W)
        self.assertEqual(W, W2)

    def test_not_equal(self):
        self.assertNotEqual(W, V)
        self.assertNotEqual(W2, V2)

    def test_sub_equality(self):
        self.assertEqual(W.sub(0), W.sub(0))
        self.assertNotEqual(W.sub(0), W.sub(1))

    def test_in_operator(self):
        self.assertTrue(f in V)
        self.assertTrue(f in V2)
        self.assertTrue(g in W)
        self.assertTrue(g in W2)

    def test_collapse(self):
        Vs = W.sub(2)
        self.assertRaises(RuntimeError, Function, Vs)
        self.assertNotEqual(Vs.dofmap().cell_dofs(0)[0], \
                            V.dofmap().cell_dofs(0)[0],)

        # Collapse the space it should now be the same as V
        Vc, dofmap_new_old = Vs.collapse(True)
        self.assertEqual(Vc.dofmap().cell_dofs(0)[0], \
                         V.dofmap().cell_dofs(0)[0],)
        f0 = Function(V)
        f1 = Function(Vc)
        self.assertEqual(len(f0.vector()), len(f1.vector()))
        
if __name__ == "__main__":
    unittest.main()
