import cffi
import numpy as np

import ufl
import dolfin


def test_interpolation_grad():
    """Interpolates grad(x^2 + y^2) into vector dP1 space. Checks
    the result is [2 * x, 2 * y].
    """
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 5, 5)

    P2 = dolfin.FunctionSpace(mesh, ("P", 2))
    dP1 = dolfin.VectorFunctionSpace(mesh, ("DP", 1))

    u = dolfin.Function(P2)
    f = dolfin.Function(dP1)

    def expr1(x):
        return x[:, 0] ** 2 + x[:, 1] ** 2

    u.interpolate(expr1)
    ufl_expr = ufl.grad(u)

    # Attach the expression with P1 points of degrees of freedom
    expr = (ufl_expr, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) )

    module = dolfin.jit.ffc_jit(expr)
    kernel = module.tabulate_expression

    ffi = cffi.FFI()

    L = dolfin.cpp.fem.Form([dP1._cpp_object])
    L.set_tabulate_cell(-1, ffi.cast("uintptr_t", kernel))
    L.set_coefficient(0, u._cpp_object)

    dolfin.fem.assemble_vector(f.vector, L, mode=dolfin.cpp.fem.InsertMode.set)

    def expr2(x):
        return np.array([2.0 * x[:, 0], 2.0 * x[:, 1]]).transpose()

    f2 = dolfin.Function(dP1)
    f2.interpolate(expr2)

    assert np.isclose((f.vector - f2.vector).norm(), 0.0)
