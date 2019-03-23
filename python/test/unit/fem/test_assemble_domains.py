# Copyright (C) 2019 Chris Richardson
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for assembly over domains"""

import math

import numpy
import pytest
from petsc4py import PETSc

import dolfin
import ufl
from ufl import ds, dx, inner


def test_basic_assembly_domains():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 1, 1)
    V = dolfin.FunctionSpace(mesh, ("DG", 0))
    u, v = dolfin.TrialFunction(V), dolfin.TestFunction(V)

    marker = dolfin.MeshFunction("size_t", mesh, mesh.topology.dim, 1)
    dx = dolfin.Measure('dx', subdomain_data=marker)

    a = inner(u, v) * dx + inner(u, v) * dx(1)
    print(a.subdomain_data())

    # Initial assembly
    A = dolfin.fem.assemble_matrix(a)
    A.assemble()
    norm1 = A.norm()

    # Set markers to 2 (should stop assembly of dx(1))
    ma = marker.array()
    ma += 1

    A2 = dolfin.fem.assemble_matrix(a)
    A2.assemble()
    norm2 = A2.norm()

    print(A[:,:])
    print(A2[:, :])
    print(norm1, norm2)

    assert numpy.isclose(norm1, norm2 * 2)
