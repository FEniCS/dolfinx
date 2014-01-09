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

        self.assertEqual(self.w.vector().sum(), 2*4*V.dim()+1*V.dim())

        assigner = FunctionAssigner(V, W.sub(2))
        assigner.assign(self.u0, self.w.sub(2))

        self.assertEqual(self.u0.vector().sum(), 4*V.dim())

        assign(self.w.sub(2), self.u2)
        self.assertEqual(self.w.vector().sum(), (4+3+1)*V.dim())

        assign(self.u1, self.w.sub(1))
        self.assertEqual(self.u1.vector().sum(), 4.0*V.dim())

        # FIXME: With better block detection it should be OK to run the
        # FIXME: rest of the tests in parallel too
        if MPI.num_processes()>1:
            return

        assigner = FunctionAssigner(WW.sub(0), W)
        assigner.assign(self.ww.sub(0), self.w)

        self.assertEqual(self.ww.vector().sum(), 5*W.dim()+(4+3+1)*V.dim())

        assign(self.wr.sub(0), self.w)
        self.assertEqual(self.wr.vector().sum(), (4+3+1)*V.dim() + 6)

        assign(self.wr.sub(1), self.r)
        self.assertEqual(self.wr.vector().sum(), (4+3+1)*V.dim() + 3)

        assign(self.qqv.sub(0).sub(0), self.q)
        self.assertEqual(self.qqv.vector().sum(), (2*3*Q.dim()+1*Q.dim()+3*V.dim()))

        self.assertRaises(RuntimeError, lambda : assign(self.qqv.sub(0), self.q))
        self.assertRaises(RuntimeError, lambda : assign(self.qqv.sub(1), self.q))
        self.assertRaises(RuntimeError, lambda : assign(self.wrr.sub(1), self.w))
        self.assertRaises(RuntimeError, lambda : assign(self.wrr.sub(1), self.r))

    def test_N_1_assigner(self):

        vv = Function(W)
        assigner = FunctionAssigner(W, [V,V,V])
        assigner.assign(vv, [self.u0, self.u1, self.u2])

        self.assertEqual(vv.vector().sum(), (1+2+3)*V.dim())

        # FIXME: With better block detection it should be OK to run the
        # FIXME: rest of the tests in parallel too
        if MPI.num_processes()>1:
            return

        assign(self.qqv, [self.qq, self.u1])
        self.assertEqual(self.qqv.vector().sum(), 2*QQV.dim())

        assign(self.wrr, [self.w, self.rr])
        self.assertEqual(self.wrr.vector().sum(), self.w.vector().sum() + \
                         self.rr.vector().sum())


        self.assertRaises(RuntimeError, lambda : assign(self.qqv, \
                                                        [self.qq, self.u1, self.u1]))

        self.assertRaises(RuntimeError, lambda : assign(self.wrr, \
                                                        [self.w, self.r, self.r]))

    def test_1_N_assigner(self):

        assigner = FunctionAssigner([V,V,V], W)
        assigner.assign([self.u0, self.u1, self.u2], self.w)

        self.assertEqual(self.u0.vector().sum() + self.u1.vector().sum() + \
                         self.u2.vector().sum(), self.w.vector().sum())

        # FIXME: With better block detection it should be OK to run the
        # FIXME: rest of the tests in parallel too
        if MPI.num_processes()>1:
            return

        assign([self.qq, self.u1], self.qqv)
        self.assertEqual(self.qqv.vector().sum(), self.qq.vector().sum() + \
                         self.u1.vector().sum())

if __name__ == "__main__":
    unittest.main()
