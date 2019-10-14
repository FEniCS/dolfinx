import cffi
from petsc4py import PETSc
import numpy as np

import ufl
import dolfin


def test_interpolation_laplace():
    mesh = dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 5, 5)

    P2 = dolfin.FunctionSpace(mesh, ("P", 2))
    dP0 = dolfin.FunctionSpace(mesh, ("DP", 0))

    u = dolfin.Function(P2)
    f = dolfin.Function(dP0)

    def evaluate(x):
        return x[:, 0] ** 2

    u.interpolate(evaluate)
    ufl_expr = ufl.div(ufl.grad(u))

    # Attach the expression with cell midpoint
    expr = (ufl_expr, np.array([[0.25, 0.25]]))

    module = dolfin.jit.ffc_jit(expr)
    kernel = module.tabulate_expression

    ffi = cffi.FFI()

    L = dolfin.cpp.fem.Form([dP0._cpp_object])
    L.set_tabulate_cell(-1, ffi.cast("uintptr_t", kernel))
    L.set_coefficient(0, u._cpp_object)

    dolfin.fem.assemble_vector(f.vector, L, mode=dolfin.cpp.fem.InsertMode.set)

    # Update ghosts with the same mode as assembly
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    assert np.isclose((f.vector - 2.0).norm(), 0.0)
