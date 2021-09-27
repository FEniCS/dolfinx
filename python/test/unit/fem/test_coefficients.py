# Copyright (C) 2021 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for packing coefficients"""


from mpi4py import MPI
import dolfinx
import ufl


def test_no_coeffs():
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx

    cpp_form = dolfinx.Form(a)._cpp_object
    form_coeffs = dolfinx.cpp.fem.pack_coefficients(cpp_form)
    assert form_coeffs.shape == (0, 0)
