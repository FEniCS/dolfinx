import dolfinx
import dolfinx.io
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


def expr(x):
    return 1j * x[0] + 1 * x[1]


def exp2(x):
    return 2 * np.ones(x.shape[1])


def vec_expr(x):
    vals = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    vals[0] = x[0] + 1j * x[1] * x[1]
    vals[1] = x[1]
    return vals


def vec_expr2(x):
    vals = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
    vals[0] = x[0]
    vals[1] = x[1] - 1
    return vals


def ten_expr(x):
    vals = np.zeros((4, x.shape[1]), dtype=PETSc.ScalarType)
    vals[0] = x[0]
    vals[1] = x[1] - 1
    vals[2] = 5
    vals[3] = 4
    return vals


mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 2, 2)
V = dolfinx.FunctionSpace(mesh, ("DG", 1))
# V2 = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
u = dolfinx.Function(V)
u.interpolate(expr)
u.name = "u1"
u.vector.assemble()
print(u.vector.array)
# u2 = dolfinx.Function(V2)
# u2.name = "u_vec"
# u2.interpolate(vec_expr)
# u3 = dolfinx.Function(V)
# u3.interpolate(exp2)
# u3.name = "Constant"
# u4 = dolfinx.Function(V2)
# u4.interpolate(vec_expr2)
# u4.name = "Vec2"

# V3 = dolfinx.TensorFunctionSpace(mesh, ("CG", 1))
# u5 = dolfinx.Function(V3)
# u5.name = "Tensor"
# u5.interpolate(ten_expr)

# V0 = dolfinx.FunctionSpace(mesh, ("DG", 0))
# v0 = dolfinx.Function(V0)
# v0.vector.setValueLocal(0, MPI.COMM_WORLD.rank + 1 + 2j)
# print(MPI.COMM_WORLD.rank, v0.vector.array)
# v0.name = "DG"
with dolfinx.cpp.io.VTKFileNew(MPI.COMM_WORLD, "results/u.pvd", "w") as vtk:
    vtk.write([u._cpp_object], 0.)

    # vtk.write([u._cpp_object, u2._cpp_object, u4._cpp_object, u3._cpp_object, u5._cpp_object, v0._cpp_object], 0.)
    #vtk.write([u._cpp_object, v0._cpp_object], 1.)
