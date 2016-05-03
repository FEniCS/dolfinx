#!/usr/bin/env py.test

"""Unit tests for the Scalar interface"""

# Copyright (C) 2011 Garth N. Wells
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

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
