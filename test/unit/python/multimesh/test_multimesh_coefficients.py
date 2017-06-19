#!/usr/bin/env py.test

"Unit tests for multimesh coefficients"

# Copyright (C) 2006 Magne Nordaas
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
#
# Modified by August Johansson 2017
#
# First added:  2017-04-02
# Last changed: 2017-06-20

from __future__ import print_function
import pytest
from dolfin import *
from dolfin_utils.test import fixture, skip_in_parallel

@fixture
def mesh0():
    return UnitSquareMesh(2,2)

@fixture
def mesh1():
    return RectangleMesh(Point(0.25, 0.25), Point(0.75, 0.75), 2, 2)

@fixture
def multimesh(mesh0, mesh1):
    multimesh = MultiMesh()
    multimesh.add(mesh0)
    multimesh.add(mesh1)
    multimesh.build()
    return multimesh

@fixture
def V(multimesh):
    return MultiMeshFunctionSpace(multimesh, "P", 1)

@fixture
def V0(mesh0):
    return FunctionSpace(mesh0, "P", 1)

@fixture
def V1(mesh1):
    return FunctionSpace(mesh1, "P", 1)

@fixture
def f(V, V0, V1):
    f = MultiMeshFunction(V)
    f.assign_part(0, interpolate(Constant(1.0), V0))
    f.assign_part(1, interpolate(Constant(2.0), V1))
    return f

@fixture
def g():
    return Constant(0.5)

@fixture
def h(V, V0, V1):
    h = MultiMeshFunction(V)
    h.assign_part(0, interpolate(Constant(1.0), V0))
    h.assign_part(1, interpolate(Constant(1.5), V1))
    return h

@skip_in_parallel
def test_dX_integral(f, g, h):
    f_dX = assemble_multimesh(f * dX)
    assert abs(f_dX - 1.25) < DOLFIN_EPS_LARGE

    fgh_dX = assemble_multimesh(f*g*h * dX)
    assert abs(fgh_dX - 0.75) < DOLFIN_EPS_LARGE

@skip_in_parallel
def test_dI_integral(f, g, h):
    f_dI0 = assemble_multimesh(f("-") * dI)
    assert abs(f_dI0 - 4.0) < DOLFIN_EPS_LARGE

    f_dI1 = assemble_multimesh(f("+") * dI)
    assert abs(f_dI1 - 2.0) < DOLFIN_EPS_LARGE

    fgh_dI0 = assemble_multimesh(f("-")*g("-")*h("-") * dI)
    assert abs(fgh_dI0 - 3.0) < DOLFIN_EPS_LARGE

    fgh_dI1 = assemble_multimesh(f("+")*g("+")*h("+") * dI)
    assert abs(fgh_dI1 - 1.0) < DOLFIN_EPS_LARGE

@skip_in_parallel
def test_dO_integral(f, g, h):
    f_dO0 = assemble_multimesh(f("-") * dO)
    assert abs(f_dO0 - 0.50) < DOLFIN_EPS_LARGE

    f_dO1 = assemble_multimesh(f("+") * dO)
    assert abs(f_dO1 - 0.25) < DOLFIN_EPS_LARGE

    fgh_dO0 = assemble_multimesh(f("-")*g("-")*h("-") * dO)
    assert abs(fgh_dO0 - 0.375) < DOLFIN_EPS_LARGE

    fgh_dO1 = assemble_multimesh(f("+")*g("+")*h("+") * dO)
    assert abs(fgh_dO1 - 0.125) < DOLFIN_EPS_LARGE
