from dolfinx import mesh, fem
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from ufl import (TrialFunction, TestFunction, CellDiameter, FacetNormal,
                 inner, grad, dx, dS, ds, avg, outer, div)


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


def jump(phi, n):
    return outer(phi("+"), n("+")) + outer(phi("-"), n("-"))


a_00 = 1 / R_e * (inner(grad(u), grad(v)) * dx -
                  inner(avg(grad(u)), jump(v, n)) * dS
                  - inner(jump(u, n), avg(grad(v))) * dS
                  + alpha / avg(h) * inner(jump(u, n), jump(v, n)) * dS
                  - inner(grad(u), outer(v, n)) * ds
                  - inner(outer(u, n), grad(v)) * ds
                  + alpha / h * inner(outer(u, n), outer(v, n)) * ds)
a_01 = - inner(p, div(v)) * dx
a_10 = - inner(div(u), q) * dx
a_11 = fem.Constant(msh, PETSc.ScalarType(0.0)) * inner(p, q) * dx

a = fem.form([[a_00, a_01],
              [a_10, a_11]])

f = fem.Function(W)
u_bc = fem.Function(V)
L_0 = inner(f, v) * dx + \
    1 / R_e * (- inner(outer(u_bc, n), grad(v)) * ds
               + alpha / h * inner(outer(u_bc, n), outer(v, n)) * ds)
L_1 = inner(fem.Constant(msh, PETSc.ScalarType(0.0)), q) * dx

L = fem.form([L_0,
              L_1])

boundary_facets = mesh.locate_entities_boundary(
    msh, msh.topology.dim - 1, boundary_marker)
boundary_vel_dofs = fem.locate_dofs_topological(
    V, msh.topology.dim - 1, boundary_facets)
bc_u = fem.dirichletbc(u_bc, boundary_vel_dofs)

# TODO TIDY
pressure_dofs = fem.locate_dofs_geometrical(
    Q, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                np.isclose(x[1], 0.0)))
if len(pressure_dofs) > 0:
    pressure_dof = [pressure_dofs[0]]
else:
    pressure_dof = []
bc_p = fem.dirichletbc(PETSc.ScalarType(0.0),
                       np.array(pressure_dof, dtype=np.int32),
                       Q)

bcs = [bc_u, bc_p]
