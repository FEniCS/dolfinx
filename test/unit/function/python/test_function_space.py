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
# Modified by Oeyvind Evju 2013
#
# First added:  2011-09-21
# Last changed: 2013-10-11

import pytest
from dolfin import *

mesh = UnitCubeMesh(8, 8, 8)
V = FunctionSpace(mesh, 'CG', 1)
W = VectorFunctionSpace(mesh, 'CG', 1)
Q = W*V
f = Function(V)
V2 = f.function_space()
g = Function(W)
W2 = g.function_space()


def test_python_interface():
    # Test Python interface of cpp generated FunctionSpace
    assert isinstance(V, FunctionSpaceBase)
    assert isinstance(W, FunctionSpaceBase)
    assert isinstance(V2, FunctionSpaceBase)
    assert isinstance(W2, FunctionSpaceBase)

    assert V.cell() == V2.cell()
    assert W.cell() == W2.cell()
    assert V.dolfin_element().signature() == V2.dolfin_element().signature()
    assert W.dolfin_element().signature() == W2.dolfin_element().signature()
    assert V.ufl_element() == V2.ufl_element()
    assert W.ufl_element() == W2.ufl_element()
    assert W.id() == W2.id()
    assert V.id() == V2.id()

    Q2 = W2*V2
    assert Q2.dim() == Q.dim()

def test_component():
    assert not W.component()
    assert not V.component()
    assert W.sub(0).component()[0] == 0
    assert W.sub(1).component()[0] == 1
    assert Q.sub(0).component()[0] == 0
    assert Q.sub(1).component()[0] == 1

def test_equality():
    assert V == V
    assert V == V2
    assert W == W
    assert W == W2

def test_boundary():
    bmesh = BoundaryMesh(mesh, "exterior")
    Vb = FunctionSpace(bmesh, "DG", 0)
    Wb = VectorFunctionSpace(bmesh, "CG", 1)
    assert Vb.dim() == 768
    assert Wb.dim() == 1158

def test_not_equal():
    assert W != V
    assert W2 != V2

def test_sub_equality():
    assert W.sub(0) == W.sub(0)
    assert W.sub(0) != W.sub(1)
    assert W.sub(0) == W.extract_sub_space([0])
    assert W.sub(1) == W.extract_sub_space([1])
    assert Q.sub(0) == Q.extract_sub_space([0])

def test_in_operator():
    assert f in V
    assert f in V2
    assert g in W
    assert g in W2

def test_collapse():
    Vs = W.sub(2)
    with pytest.raises(RuntimeError):
        Function(Vs)
    assert Vs.dofmap().cell_dofs(0)[0] !=  V.dofmap().cell_dofs(0)[0]

    # Collapse the space it should now be the same as V
    Vc, dofmap_new_old = Vs.collapse(True)
    assert Vc.dofmap().cell_dofs(0)[0] == V.dofmap().cell_dofs(0)[0]
    f0 = Function(V)
    f1 = Function(Vc)
    assert len(f0.vector()) == len(f1.vector())

def test_argument_equality():
    "Placed this test here because it's mainly about detecting differing function spaces."
    mesh2 = UnitCubeMesh(8, 8, 8)
    V3 = FunctionSpace(mesh2, 'CG', 1)
    W3 = VectorFunctionSpace(mesh2, 'CG', 1)

    for TF in (TestFunction, TrialFunction):
        v = TF(V)
        v2 = TF(V2)
        v3 = TF(V3)
        assert v == v2
        assert v2 == v
        assert V != V3
        assert V2 != V3
        assert not(v == v3)
        assert not(v2 == v3)
        assert v != v3
        assert v2 != v3
        assert v != v3
        assert v2 != v3

        w = TF(W)
        w2 = TF(W2)
        w3 = TF(W3)
        assert w == w2
        assert w2 == w
        assert w != w3
        assert w2 != w3

        assert v != w
        assert w != v

        s1 = set((v, w))
        s2 = set((v2, w2))
        s3 = set((v, v2, w, w2))
        assert len(s1) == 2
        assert len(s2) == 2
        assert len(s3) == 2
        assert s1 == s2
        assert s1 == s3
        assert s2 == s3

        # Test that the dolfin implementation of Argument.__eq__
        # is triggered when comparing ufl expressions
        assert grad(v) == grad(v2)
        assert grad(v) != grad(v3)

if __name__ == "__main__":
    pytest.main()
