"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
As d2_p2D.py, but LinearVariationalProblem and LinearVariationalSolver
are used instead of the solve(a == L, u, bc) call in d2_p2D.py.

-Laplace(u) = f on the unit square.
u = u0 on the boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""

from __future__ import print_function
import os
from dolfin import *

# Create mesh and define function space
#mesh = UnitSquareMesh(600, 400)
mesh = UnitSquareMesh(60, 40)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def u0_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u0, u0_boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
problem = LinearVariationalProblem(a, L, u, bc)
solver = LinearVariationalSolver(problem)
solver.parameters['linear_solver'] = 'gmres'
solver.parameters['preconditioner'] = 'ilu'
info(solver.parameters, True)

print(parameters['linear_algebra_backend'])
cg_prm = solver.parameters['krylov_solver'] # short form
cg_prm['absolute_tolerance'] = 1E-7
cg_prm['relative_tolerance'] = 1E-4
cg_prm['maximum_iterations'] = 10000
#cg_prm['preconditioner']['ilu']['fill_level'] = 0

set_log_level(DEBUG)
solver.solve()


# Plot solution and mesh
#plot(u)
#plot(mesh)

# Dump solution to file in VTK format
file = File('poisson.pvd')
file << u

# Hold plot
interactive()
