"""Unit tests for the FunctionAssigner class"""

# Copyright (C) 2013 Johan Hake
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
# First added:  2013-11-07
# Last changed: 2013-11-07

import unittest
from dolfin import *
import numpy as np

mesh = UnitCubeMesh(8, 8, 8)
V = FunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 2)
W = VectorFunctionSpace(mesh, "CG", 1)
QQ = VectorFunctionSpace(mesh, "CG", 2)
R = FunctionSpace(mesh, "R", 0)
RR = VectorFunctionSpace(mesh, "R", 0)
QQV = QQ*V
WW = W*W
WR = W*R
WRR = W*RR

class Creation(unittest.TestCase):
    def setUp(self):

        self.u0 = Function(V)
        self.u0.vector()[:] = 1.
        self.u1 = Function(V)
        self.u1.vector()[:] = 2.
        self.u2 = Function(V)
        self.u2.vector()[:] = 3.

        self.r = Function(R)
        self.r.vector()[:] = 3.

        self.rr = Function(RR)
        self.rr.vector()[:] = 2.

        self.w = Function(W)
        self.w.vector()[:] = 4.

        self.ww = Function(WW)
        self.ww.vector()[:] = 5.

        self.wr = Function(WR)
        self.wr.vector()[:] = 6.

        self.wrr = Function(WRR)
        self.wrr.vector()[:] = 7.

        self.q = Function(Q)
        self.q.vector()[:] = 1.

        self.qq = Function(QQ)
        self.qq.vector()[:] = 2.

        self.qqv = Function(QQV)
        self.qqv.vector()[:] = 3.

    def test_1_1_assigner(self):

        assigner = FunctionAssigner(W.sub(0), V)
        assigner.assign(self.w.sub(0), self.u0)

        self.assertTrue(np.all(self.w.sub(0, deepcopy=True).vector().array() == \
                               self.u0.vector().array()))

        assign(self.w.sub(2), self.u2)
        self.assertTrue(np.all(self.w.sub(2, deepcopy=True).vector().array() == \
                               self.u2.vector().array()))

        assigner = FunctionAssigner(V, W.sub(2))
        assigner.assign(self.u0, self.w.sub(2))

        self.assertTrue(np.all(self.u0.vector().array() == \
                               self.w.sub(2, deepcopy=True).vector().array()))

        assign(self.u1, self.w.sub(1))
        self.assertTrue(np.all(self.u1.vector().array() == \
                               self.w.sub(1, deepcopy=True).vector().array()))

        assigner = FunctionAssigner(WW.sub(0), W)
        assigner.assign(self.ww.sub(0), self.w)

        assign(self.wr.sub(0), self.w)
        self.assertTrue(np.all(self.wr.sub(0, deepcopy=True).vector().array() == \
                               self.w.vector().array()))

        assign(self.wr.sub(1), self.r)
        self.assertTrue(np.all(self.wr.sub(1, deepcopy=True).vector().array() == \
                               self.r.vector().array()))

        assign(self.qqv.sub(0).sub(0), self.q)
        self.assertTrue(np.all(self.qqv.sub(0).sub(0, deepcopy=True).vector().array() == \
                               self.q.vector().array()))

        self.assertRaises(RuntimeError, lambda : assign(self.qqv.sub(0), self.q))
        self.assertRaises(RuntimeError, lambda : assign(self.qqv.sub(1), self.q))
        self.assertRaises(RuntimeError, lambda : assign(self.wrr.sub(1), self.w))

    def test_N_1_assigner(self):

        vv = Function(W)
        assigner = FunctionAssigner(W, [V,V,V])
        assigner.assign(vv, [self.u0, self.u1, self.u2])

        self.assertTrue(np.all(vv.sub(0, deepcopy=True).vector().array() == \
                               self.u0.vector().array()))
        self.assertTrue(np.all(vv.sub(1, deepcopy=True).vector().array() == \
                               self.u1.vector().array()))
        self.assertTrue(np.all(vv.sub(2, deepcopy=True).vector().array() == \
                               self.u2.vector().array()))

        assign(self.qqv, [self.qq, self.u1])
        self.assertTrue(np.all(self.qqv.sub(0, deepcopy=True).vector().array() == \
                              self.qq.vector().array()))
        self.assertTrue(np.all(self.qqv.sub(1, deepcopy=True).vector().array() == \
                               self.u1.vector().array()))

        assign(self.wrr, [self.w, self.rr])
        self.assertTrue(np.all(self.wrr.sub(0, deepcopy=True).vector().array() ==
                               self.w.vector().array()))
        self.assertTrue(np.all(self.wrr.sub(1, deepcopy=True).vector().array() ==
                               self.rr.vector().array()))

        self.assertRaises(RuntimeError, lambda : assign(self.qqv, \
                                                        [self.qq, self.u1, self.u1]))

        self.assertRaises(RuntimeError, lambda : assign(self.wrr, \
                                                        [self.w, self.r, self.r]))

    def test_1_N_assigner(self):

        assigner = FunctionAssigner([V,V,V], W)
        assigner.assign([self.u0, self.u1, self.u2], self.w)

        self.assertTrue(np.all(self.w.sub(0, deepcopy=True).vector().array() == \
                               self.u0.vector().array()))
        self.assertTrue(np.all(self.w.sub(1, deepcopy=True).vector().array() == \
                               self.u1.vector().array()))
        self.assertTrue(np.all(self.w.sub(2, deepcopy=True).vector().array() == \
                               self.u2.vector().array()))

        assign([self.qq, self.u1], self.qqv)

        self.assertTrue(np.all(self.qqv.sub(0, deepcopy=True).vector().array() == \
                              self.qq.vector().array()))
        self.assertTrue(np.all(self.qqv.sub(1, deepcopy=True).vector().array() == \
                               self.u1.vector().array()))

if __name__ == "__main__":
    unittest.main()
