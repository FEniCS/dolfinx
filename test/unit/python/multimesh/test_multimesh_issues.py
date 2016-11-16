#!/usr/bin/env py.test

"""Unit tests for multimesh volume computation"""

# Copyright (C) 2016 Anders Logg
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
# Modified by August Johansson 2016
#
# First added:  2016-05-03
# Last changed: 2016-11-16


from __future__ import print_function
import pytest

from dolfin import *
from dolfin_utils.test import skip_in_parallel

def test_issue_754():
    N = 3
    meshes = [UnitSquareMesh(2*N, 2*N),
              RectangleMesh(Point(0.1, 0.15), Point(0.6, 0.65), N, N),
              RectangleMesh(Point(0.4, 0.35), Point(0.9, 0.85), N, N)]

    multimesh = MultiMesh()
    for mesh in meshes:
        multimesh.add(mesh)
    multimesh.build()

    V = MultiMeshFunctionSpace(multimesh, "CG", 1)
    u, v = TrialFunction(V), TestFunction(V)

    a = inner(u,v) * dX
    b = a + inner(jump(u),jump(v)) * dI
    A = assemble_multimesh(a)
    B = assemble_multimesh(b)

    # create multimesh function w such that
    #     w(x) =  1  if x is in the last mesh
    #          =  0  otherwise

    w = MultiMeshFunction(V); x = w.vector()
    dofs = V.part(V.num_parts()-1).dofmap().dofs() \
           + sum(V.part(i).dim() for i in range(V.num_parts()-1))
    w.vector()[dofs] = 1.

    # Compute the area and perimeter of the last mesh
    a = w.vector().inner(A * w.vector())
    p = w.vector().inner(B * w.vector()) - a

    # print("Computed area (a) and perimeter (p) of last mesh:")
    # print("  a = {0:1.4f} (exact value is .25)".format(a))
    # print("  p = {0:1.4f} (exact value is 2.0)".format(p))
    assert(abs(a - 0.25) < DOLFIN_EPS_LARGE)
    assert(abs(p - 2.) < DOLFIN_EPS_LARGE)
