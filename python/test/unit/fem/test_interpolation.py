import pytest

import ufl
import dolfin


def test_interpolation():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 5, 5)
    vP1 = dolfin.VectorFunctionSpace(mesh, ("P", 1))

    u1 = dolfin.Function(vP1)
    u1.vector.set(1.0)

    u2 = dolfin.Function(vP1)

    k1 = dolfin.Constant(mesh, [[1.0, 2.0], [3.0, 4.0]])
    x = ufl.SpatialCoordinate(mesh)
    expr = u2 * ufl.zero(()) + ufl.inner(x, u1) * ufl.dot(k1, u1)

    f = dolfin.Function(vP1)
    dolfin.fem.interpolation.compiled_interpolation(expr, vP1, f)


def test_interpolation_laplace():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 5, 5)

    P2 = dolfin.FunctionSpace(mesh, ("P", 2))
    dP0 = dolfin.FunctionSpace(mesh, ("DP", 0))

    u = dolfin.Function(P2)
    f = dolfin.Function(dP0)

    def evaluate(values, x):
        values[:, 0] = x[:, 0] ** 2

    u.interpolate(evaluate)
    ufl_expr = ufl.div(ufl.grad(u))
    dolfin.fem.interpolation.compiled_interpolation(ufl_expr, dP0, f)

    assert pytest.approx((f.vector - 2.0).norm(), 0.0)
