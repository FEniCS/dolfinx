"""Tests for DOLFIN integration of various form operations"""

# Copyright (C) 2011 Marie E. Rognes
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numpy
import pytest

from dolfinx import MPI, FunctionSpace, RectangleMesh
from dolfinx.cpp.mesh import CellType
from ufl import TestFunction, TrialFunction, dx, grad, inner, lhs, rhs, system


@pytest.mark.skip
def test_lhs_rhs_simple():
    """Test taking lhs/rhs of DOLFIN specific forms (constants
    without cell). """

    mesh = RectangleMesh(MPI.comm_world, [numpy.array([0.0, 0.0, 0.0]),
                                          numpy.array([2.0, 1.0, 0.0])],
                         [3, 5], CellType.triangle)
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
