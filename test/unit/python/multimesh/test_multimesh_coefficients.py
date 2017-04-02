from __future__ import print_function
import pytest
from dolfin import *
from dolfin_utils.test import skip_in_parallel

mesh0 = UnitSquareMesh(2,2)
mesh1 = RectangleMesh(Point(0.25, 0.25), Point(0.75, 0.75), 2, 2)

multimesh = MultiMesh()
multimesh.add(mesh0)
multimesh.add(mesh1)

multimesh.build()

V = MultiMeshFunctionSpace(multimesh, "P", 1)
V0 = FunctionSpace(mesh0, "P", 1)
V1 = FunctionSpace(mesh1, "P", 1)

f = MultiMeshFunction(V)
f.assign_part(0, interpolate(Constant(1.0), V0))
f.assign_part(1, interpolate(Constant(2.0), V1))

@skip_in_parallel
def test_dX_integral():
    f_dX = assemble_multimesh(f * dX)
    assert abs(f_dX - 1.25) < DOLFIN_EPS_LARGE

@skip_in_parallel
def test_dI_integral():
    f_dI0 = assemble_multimesh(f("-") * dI)
    assert abs(f_dI0 - 2.0) < DOLFIN_EPS_LARGE

    f_dI1 = assemble_multimesh(f("+") * dI)
    assert abs(f_dI1 - 4.0) < DOLFIN_EPS_LARGE

@skip_in_parallel
def test_dO_integral():
    f_dO0 = assemble_multimesh(f("-") * dO)
    assert abs(f_dO0 - 0.25) < DOLFIN_EPS_LARGE

    f_dO1 = assemble_multimesh(f("+") * dO)
    assert abs(f_dO1 - 0.50) < DOLFIN_EPS_LARGE


