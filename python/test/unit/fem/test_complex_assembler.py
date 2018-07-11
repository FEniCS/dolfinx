# Copyright (C) 2018 Igor A. Baratta
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly in complex mode"""

import ufl
import dolfin
from ufl import dx, inner
import numpy as np
import pytest


@pytest.mark.skipif(not dolfin.has_petsc_complex(),
                    reason="This test only works in complex mode.")
def test_complex_vector_assembly():
    """Test assembly of complex scalars"""

    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 32, 31)
    P2 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), 2)
    V = dolfin.function.functionspace.FunctionSpace(mesh, P2)

    u = dolfin.function.argument.TrialFunction(V)
    v = dolfin.function.argument.TestFunction(V)

    g = dolfin.function.constant.Constant(-2 + 3.0j)
    zero = dolfin.function.constant.Constant(0.0j)

    a = zero * inner(u, v) * dx
    L1 = inner(g, v) * dx
    assembler = dolfin.fem.assembling.Assembler(a, L1)
    A0, b0 = assembler.assemble()
    bnorm = b0.norm(dolfin.cpp.la.Norm.l1)
    b_norm_ref = abs(-2 + 3.0j)
    assert (np.isclose(bnorm, b_norm_ref))

    f = dolfin.Expression("j*sin(2*pi*x[0])", degree=2)
    L0 = inner(f, v) * dx
    assembler = dolfin.fem.assembling.Assembler(a, L0)
    A1, b1 = assembler.assemble()
    b1_norm = b1.norm(dolfin.cpp.la.Norm.l2)

    f = dolfin.Expression("sin(2*pi*x[0])", degree=2)
    L2 = inner(f, v) * dx
    assembler = dolfin.fem.assembling.Assembler(a, L2)
    A2, b2 = assembler.assemble()
    b2_norm = b1.norm(dolfin.cpp.la.Norm.l2)

    b1_inf = b1.norm(dolfin.cpp.la.Norm.linf)
    b2_inf = b2.norm(dolfin.cpp.la.Norm.linf)

    assert (np.isclose(b2_norm, b1_norm))
    assert (np.isclose(b1_inf, b2_inf))
