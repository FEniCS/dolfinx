"""Tests for DOLFIN integration of various form operations"""

# Copyright (C) 2011 Marie E. Rognes
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later


from dolfin import *

def test_lhs_rhs_simple():
    """Test taking lhs/rhs of DOLFIN specific forms (constants
    without cell). """

    mesh = RectangleMesh(Point(0, 0), Point(2, 1), 3, 5)
    V = FunctionSpace(mesh, "CG", 1)
    f = Constant(2.0)
    g = Constant(3.0)
    v = TestFunction(V)
    u = TrialFunction(V)

    F = inner(g*grad(f*v), grad(u))*dx + f*v*dx
    a, L = system(F)

    Fl = lhs(F)
    Fr = rhs(F)

    a0 = inner(grad(v), grad(u))*dx

    n = assemble(a).norm("frobenius")
    nl = assemble(Fl).norm("frobenius")
    n0 = 6.0*assemble(a0).norm("frobenius")

    assert round(n - n0, 7) == 0
    assert round(n - nl, 7) == 0

