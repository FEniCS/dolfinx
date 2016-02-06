"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
As d4_p2D.py, but with computations and visualizations of grad(u).

-Laplace(u) = f on the unit square.
u = u0 on the boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""

from __future__ import print_function
import os
from dolfin import *
import numpy

# Create mesh and define function space
#mesh = UnitSquareMesh(8, 8)
mesh = UnitSquareMesh(2, 2)
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
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)
u.rename('u', 'solution field')

# Compute gradient
V_g = VectorFunctionSpace(mesh, 'Lagrange', 1)
v = TestFunction(V_g)
w = TrialFunction(V_g)

a = inner(w, v)*dx
L = inner(grad(u), v)*dx
grad_u = Function(V_g)
solve(a == L, grad_u)
grad_u.rename('grad(u)', 'continuous gradient field')

#plot(u, title=u.name())
#plot(grad_u, title=grad_u.name())

grad_u_x, grad_u_y = grad_u.split(deepcopy=True)  # extract components
grad_u_x.rename('grad(u)_x', 'x-component of grad(u)')
grad_u_y.rename('grad(u)_y', 'y-component of grad(u)')
#plot(grad_u_x, title=grad_u_x.label())
#plot(grad_u_y, title=grad_u_y.label())

# Quick summary print of key variables
print(mesh)
print(u)
print(grad_u)
print(grad_u_x)
print(grad_u_y)

# Alternative computation of grad(u)
grad_u2 = project(grad(u), VectorFunctionSpace(mesh, 'Lagrange', 1))

# Dump solution and grad(u) to the screen with errors
# (in case of linear Lagrange elements only)
u_array = u.vector().array()
if mesh.num_cells() < 160 and u_array.size == mesh.num_vertices():
    grad_u_x_array = grad_u_x.vector().array()
    grad_u_y_array = grad_u_y.vector().array()
    coor = mesh.coordinates()
    for i in range(len(u_array)):
        x, y = coor[i]
        print('Node (%.3f,%.3f): u = %.4f (%9.2e), '\
              'grad(u)_x = %.4f  (%9.2e), grad(u)_y = %.4f  (%9.2e)' % \
              (x, y, u_array[i], 1 + x**2 + 2*y**2 - u_array[i],
               grad_u_x_array[i], 2*x - grad_u_x_array[i],
               grad_u_y_array[i], 4*y - grad_u_y_array[i]))

grad_u_array = grad_u.vector().array()
print('grad_u array:', grad_u_array, len(grad_u_array), grad_u_array.shape)

# Verification
u_e = interpolate(u0, V)
u_e_array = u_e.vector().array()
print('Max error:', numpy.abs(u_e_array - u_array).max())

# Compare numerical and exact solution
center = (0.5, 0.5)
print('numerical u at the center point:', u(center))
print('exact     u at the center point:', u0(center))

# Normalize solution such that max(u) = 1:
max_u = u_array.max()
u_array /= max_u
#u.vector().set_local(u_array)
u.vector()[:] = u_array
#u.vector().set_local(u_array)  # safer for parallel computing
print('\nNormalized solution:\n', u.vector().array())

file = File('poisson.pvd')
file << u
file << grad_u
file << grad_u_x
file << grad_u_y

# Should be at the end
interactive()
