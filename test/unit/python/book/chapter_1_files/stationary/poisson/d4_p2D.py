"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
As d1_p2D.py, but the solution is checked to coincide with the exact solution
at all nodes.

-Laplace(u) = f on the unit square.
u = u0 on the boundary.
u0 = u = 1 + x^2 + 2y^2, f = -6.
"""

from __future__ import print_function
from dolfin import *
import numpy

# Create mesh and define function space
mesh = UnitSquareMesh(3, 3)
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
solve(a == L, u, bc)

#plot(u)

print("""
Solution of the Poisson problem -Laplace(u) = f,
with u = u0 on the boundary and a
%s
""" % str(mesh))

# Dump solution to the screen
u_nodal_values = u.vector()
u_array = u_nodal_values.array()
coor = mesh.coordinates()
if coor.shape[0] == u_array.shape[0]:  # degree 1 element
    for i in range(len(u_array)):
        print('u(%8g,%8g) = %g' % (coor[i][0], coor[i][1], u_array[i]))
else:
    print("""\
Cannot print vertex coordinates and corresponding values
because the element is not a first-order Lagrange element.
""")

# Note: u_nodal_values.array() returns a copy
print(id(u_nodal_values.array()), id(u_array))

# Verification
u_e = interpolate(u0, V)
u_e_array = u_e.vector().array()
print('Max error:', numpy.abs(u_e_array - u_array).max())

# Compare numerical and exact solution
center = (0.5, 0.5)
u_value = u(center)
u0_value = u0(center)
print('numerical u at the center point:', u_value)
print('exact     u at the center point:', u0_value)

# Normalize solution such that max(u) = 1:
max_u = u_array.max()
u_array /= max_u
#u.vector().set_local(u_array)
u.vector()[:] = u_array
#u.vector().set_local(u_array)  # safer for parallel computing
print('\nNormalized solution:\n', u.vector().array())

# interactive() always be at the end
#interactive()
