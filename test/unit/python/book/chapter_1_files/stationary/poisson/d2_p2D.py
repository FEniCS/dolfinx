"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
As d1_p2D.py, but chosing linear solver and preconditioner is demonstrated.

-Laplace(u) = f on the unit square.
u = u0 on the boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""

from __future__ import print_function
import os
from dolfin import *

# Create mesh and define function space
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
info(parameters, True)

prm = parameters['krylov_solver'] # short form
prm['absolute_tolerance'] = 1E-5
prm['relative_tolerance'] = 1E-3
prm['maximum_iterations'] = 1000
#prm['preconditioner']['ilu']['fill_level'] = 0
print(parameters['linear_algebra_backend'])
#set_log_level(PROGRESS)
set_log_level(DEBUG)

#solve(a == L, u, bc,
#      solver_parameters={'linear_solver': 'cg',
#                         'preconditioner': 'ilu'})
solve(a == L, u, bc,
      solver_parameters={'linear_solver': 'gmres',
                         'preconditioner': 'ilu'})

# Alternative syntax
solve(a == L, u, bc,
      solver_parameters=dict(linear_solver='gmres',
                             preconditioner='ilu'))
#solve(a == L, u, bc,
#      solver_parameters=dict(linear_solver='cg',
#                             preconditioner='ilu'))

# Plot solution and mesh
#plot(u)
#plot(mesh)

# Dump solution to file in VTK format
file = File('poisson.pvd')
file << u

# Hold plot
interactive()
