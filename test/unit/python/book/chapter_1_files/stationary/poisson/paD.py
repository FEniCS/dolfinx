"""
FEniCS tutorial demo program: Poisson equation in anyD (1D, 2D, 3D)
with Dirichlet and Neumann conditions.

-Laplace(u) = 2 on the unit hypercube.
u = 0 at x=0,
u = 1 at x=1,
du/dn = 0 at all other boundaries.
Exact solution: u(x, ...) = x^2
"""

from __future__ import print_function
from dolfin import *
import numpy, sys

# Create mesh and define function space
degree = int(sys.argv[1])
divisions = [int(arg) for arg in sys.argv[2:]]
d = len(divisions)
domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
mesh = domain_type[d-1](*divisions)
V = FunctionSpace(mesh, 'Lagrange', degree)

# Define Dirichlet conditions at two sides

tol = 1E-14   # tolerance for coordinate comparisons
def Dirichlet_boundary0(x, on_boundary):
    return on_boundary and abs(x[0]) < tol

def Dirichlet_boundary1(x, on_boundary):
    return on_boundary and abs(x[0] - 1) < tol

bc0 = DirichletBC(V, Constant(0), Dirichlet_boundary0)
bc1 = DirichletBC(V, Constant(1), Dirichlet_boundary1)
bcs = [bc0, bc1]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

print(mesh)

# Verification
u_exact = Expression('x[0]*x[0]', degree=2)
u_e = interpolate(u_exact, V)
print('Max error:', \
      numpy.abs(u_e.vector().array() - u.vector().array()).max())
