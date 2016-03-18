"""
FEniCS tutorial demo program:
Poisson equation with Dirichlet and Neumann conditions.
As dn1_p2D.py, but the Dirichlet boundary is split into two
parts (separate objects) with different Expression objects
to set the conditions.

-Laplace(u) = f on the unit square.
u = 1 + 2y^2 on x=0.
u = 2 + 2y^2 on x=1.
-du/dn = g on y=0 and y=1.
u = 1 + x^2 + 2y^2, f = -6, g = -4y.
"""

from __future__ import print_function
from dolfin import *
import numpy

# Create mesh and define function space
mesh = UnitSquareMesh(3, 2)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define Dirichlet conditions for x=0 boundary

u_L = Expression('1 + 2*x[1]*x[1]', degree=2)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0]) < tol

Gamma_0 = DirichletBC(V, u_L, LeftBoundary())

# Define Dirichlet conditions for x=1 boundary

u_R = Expression('2 + 2*x[1]*x[1]', degree=2)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0] - 1) < tol

Gamma_1 = DirichletBC(V, u_R, RightBoundary())

bcs = [Gamma_0, Gamma_1]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
g = Expression('-4*x[1]', degree=1)
a = inner(grad(u), grad(v))*dx
L = f*v*dx - g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

#plot(u)

print("""
Solution of the Poisson problem -Laplace(u) = f,
with u = u0 on x=0,1 and -du/dn = g at y=0,1.
%s
""" % mesh)

# Dump solution to the screen
u_nodal_values = u.vector()
u_array = u_nodal_values.array()
coor = mesh.coordinates()
for i in range(len(u_array)):
    print('u(%8g,%8g) = %g' % (coor[i][0], coor[i][1], u_array[i]))


# Exact solution:
u_exact = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

# Verification
u_e = interpolate(u_exact, V)
u_e_array = u_e.vector().array()
print('Max error:', numpy.abs(u_e_array - u_array).max())

# Compare numerical and exact solution
center = (0.5, 0.5)
print('numerical u at the center point:', u(center))
print('exact     u at the center point:', u_exact(center))

#interactive()
