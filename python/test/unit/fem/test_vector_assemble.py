# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly on vector spaces"""


import ufl
from dolfinx.fem import VectorFunctionSpace, form
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import create_unit_square

from mpi4py import MPI


def test_vector_assemble_matrix_exterior():
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 3)
    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(ufl.inner(u, v) * ufl.ds)
    A = assemble_matrix(a)
    A.assemble()


def test_vector_assemble_matrix_interior():
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 3)
    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(ufl.inner(ufl.jump(u), ufl.jump(v)) * ufl.dS)
    A = assemble_matrix(a)
    A.assemble()
