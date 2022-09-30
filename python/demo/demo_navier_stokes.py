from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from ufl import TrialFunction, TestFunction, CellDiameter, FacetNormal


n = 16
num_time_steps = 25
t_end = 10
R_e = 25
k = 1


def u_e_expr(x):
    return np.vstack((1 - np.exp(
        (R_e / 2 - np.sqrt(R_e**2 / 4 + 4 * np.pi**2)) * x[0])
        * np.cos(2 * np.pi * x[1]),
        (R_e / 2 - np.sqrt(R_e**2 / 4 + 4 * np.pi**2))
        / (2 * np.pi) * np.exp(
            (R_e / 2 - np.sqrt(R_e**2 / 4 + 4 * np.pi**2)) * x[0])
        * np.sin(2 * np.pi * x[1])))


def p_e_expr(x):
    return (1 / 2) * (1 - np.exp(
        2 * (R_e / 2 - np.sqrt(R_e**2 / 4 + 4 * np.pi**2)) * x[0]))


def f_expr(x):
    return np.vstack((np.zeros_like(x[0]),
                      np.zeros_like(x[0])))


def boundary_marker(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | \
        np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)


msh = mesh.create_unit_square(MPI.COMM_WORLD, n, n)

V = fem.FunctionSpace(msh, ("Raviart-Thomas", k + 1))
Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
W = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k + 1))

u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)

delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
alpha = fem.Constant(msh, PETSc.ScalarType(6.0 * k**2))
R_e = fem.Constant(msh, PETSc.ScalarType(R_e))

h = CellDiameter(msh)
n = FacetNormal(msh)
