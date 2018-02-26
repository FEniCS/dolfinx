"""Unit tests for the Scalar interface"""

# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *


def test_scalar_parallel_sum():
    a = Scalar()
    b = 1.0
    a.add_local_value(b)
    a.apply("add")
    assert round(a.get_scalar_value() - b*MPI.size(a.mpi_comm()), 7) == 0


def test_scalar_assembly():
    mesh = UnitSquareMesh(3, 3)

    S = Scalar()
    assemble(Constant(1.0)*dx(mesh), tensor=S)
    assemble(Constant(1.0)*dx(mesh), tensor=S)
    assert round(S.get_scalar_value() - 1.0, 7) == 0

    S = Scalar()
    assemble(Constant(1.0)*dx(mesh), tensor=S)
    assemble(Constant(1.0)*dx(mesh), tensor=S, add_values=True)
    assert round(S.get_scalar_value() - 2.0, 7) == 0

    S = Scalar()
    assemble(Constant(1.0)*dx(mesh), tensor=S, add_values=True)
    assemble(Constant(1.0)*dx(mesh), tensor=S, add_values=True)
    assert round(S.get_scalar_value() - 2.0, 7) == 0
