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

mesh = UnitCubeMesh(8, 8, 8)
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)
Q = W*V
f = Function(V)
V2 = f.function_space()
g = Function(W)
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
        self.assertEqual(W.sub(0), W.extract_sub_space([0]))
        self.assertEqual(W.sub(1), W.extract_sub_space([1]))
        self.assertEqual(Q.sub(0), Q.extract_sub_space([0]))

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

    def test_argument_equality(self):
        mesh2 = UnitCubeMesh(1, 1, 1)
        V3 = FunctionSpace(mesh2, 'CG', 1)
        W3 = VectorFunctionSpace(mesh2, 'CG', 1)

        for TF in (TestFunction, TrialFunction):
            v = TF(V)
            v2 = TF(V2)
            v3 = TF(V3)
            self.assertEqual(v, v2)
            self.assertEqual(v2, v)
            self.assertNotEqual(V, V3)
            self.assertNotEqual(V2, V3)
            self.assertTrue(not(v == v3))
            self.assertTrue(not(v2 == v3))
            self.assertTrue(v != v3)
            self.assertTrue(v2 != v3)
            self.assertNotEqual(v, v3)
            self.assertNotEqual(v2, v3)

            w = TF(W)
            w2 = TF(W2)
            w3 = TF(W3)
            self.assertEqual(w, w2)
            self.assertEqual(w2, w)
            self.assertNotEqual(w, w3)
            self.assertNotEqual(w2, w3)

            self.assertNotEqual(v, w)
            self.assertNotEqual(w, v)

            s1 = set((v, w))
            s2 = set((v2, w2))
            s3 = set((v, v2, w, w2))
            self.assertEqual(len(s1), 2)
            self.assertEqual(len(s2), 2)
            self.assertEqual(len(s3), 2)
            self.assertEqual(s1, s2)
            self.assertEqual(s1, s3)
            self.assertEqual(s2, s3)

if __name__ == "__main__":
    unittest.main()
