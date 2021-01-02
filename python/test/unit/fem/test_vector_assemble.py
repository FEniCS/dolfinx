# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINX (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly on vector spaces"""


import dolfinx
import ufl
from mpi4py import MPI


def test_vector_assemble_matrix_exterior():
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.ds
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()


def test_vector_assemble_matrix_interior():
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 3, 3)
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.jump(u), ufl.jump(v)) * ufl.dS
    A = dolfinx.fem.assemble_matrix(a)
    A.assemble()
