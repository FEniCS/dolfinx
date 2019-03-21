"""Tests for DOLFIN integration of various form operations"""

# Copyright (C) 2011 Marie E. Rognes
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pytest
from dolfin import (RectangleMesh, MPI, Point, CellType, FunctionSpace,
                    TestFunction, TrialFunction, inner, grad, dx, system, lhs, rhs)


@pytest.mark.skip
def test_lhs_rhs_simple():
    """Test taking lhs/rhs of DOLFIN specific forms (constants
    without cell). """

    mesh = RectangleMesh(MPI.comm_world, [Point(0, 0), Point(2, 1)], [3, 5], CellType.Type.triangle)
    V = FunctionSpace(mesh, "CG", 1)
    f = 2.0
    g = 3.0
    v = TestFunction(V)
    u = TrialFunction(V)

    F = inner(g * grad(f * v), grad(u)) * dx + f * v * dx
    a, L = system(F)

    Fl = lhs(F)
    Fr = rhs(F)
    assert(Fr)

    a0 = inner(grad(v), grad(u)) * dx

    n = assemble(a).norm("frobenius")  # noqa
    nl = assemble(Fl).norm("frobenius")  # noqa
    n0 = 6.0 * assemble(a0).norm("frobenius")  # noqa

    assert round(n - n0, 7) == 0
    assert round(n - nl, 7) == 0
