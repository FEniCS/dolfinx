# Copyright (C) 2020 Jørgen S. Dokken
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly on vector spaces"""

from mpi4py import MPI

import ufl
from dolfinx.fem import assemble_matrix, form, functionspace
from dolfinx.mesh import create_unit_square


def test_vector_assemble_matrix_exterior():
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 3)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(ufl.inner(u, v) * ufl.ds)
    A = assemble_matrix(a)
    A.scatter_reverse()


def test_vector_assemble_matrix_interior():
    mesh = create_unit_square(MPI.COMM_WORLD, 3, 3)
    gdim = mesh.geometry.dim
    V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = form(ufl.inner(ufl.jump(u), ufl.jump(v)) * ufl.dS)
    A = assemble_matrix(a)
    A.scatter_reverse()
