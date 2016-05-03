"""
FEniCS tutorial demo program: Poisson equation with Dirichlet,
Neumann and Robin conditions.
The solution is checked to coincide with the exact solution at all nodes.

The file is a modification of dn2_p2D.py. Note that the boundary is now also
split into two distinct parts (separate objects and integrations)
and we have a Robin condition instead of a Neumann condition at y=0.
"""

from __future__ import print_function
from dolfin import *
import numpy

#-------------- Preprocessing step -----------------

# Create mesh and define function space
mesh = UnitSquareMesh(3, 2)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary segments for Neumann, Robin and Dirichlet conditions

# Create mesh function over cell facets
boundary_parts = MeshFunction("size_t", mesh, mesh.topology().dim()-1)

# Mark lower boundary facets as subdomain 0
class LowerRobinBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[1]) < tol

Gamma_R = LowerRobinBoundary()
Gamma_R.mark(boundary_parts, 0)
q = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
p = Constant(100)  # arbitrary function can go here

# Mark upper boundary facets as subdomain 1
class UpperNeumannBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[1] - 1) < tol

Gamma_N = UpperNeumannBoundary()
Gamma_N.mark(boundary_parts, 1)
g = Expression('-4*x[1]', degree=1)

# Mark left boundary as subdomain 2
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0]) < tol

Gamma_0 = LeftBoundary()
Gamma_0.mark(boundary_parts, 2)

# Mark right boundary as subdomain 3
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0] - 1) < tol

Gamma_1 = RightBoundary()
Gamma_1.mark(boundary_parts, 3)

#-------------- Solution and problem definition step -----------------
# given mesh and boundary_parts

u_L = Expression('1 + 2*x[1]*x[1]', degree=2)
u_R = Expression('2 + 2*x[1]*x[1]', degree=2)
bcs = [DirichletBC(V, u_L, boundary_parts, 2),
       DirichletBC(V, u_R, boundary_parts, 3)]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
ds = ds(0, subdomain_data=boundary_parts)
a = inner(nabla_grad(u), nabla_grad(v))*dx \
    + p*u*v*ds(0)
L = f*v*dx - g*v*ds(1) \
    + p*q*v*ds(0)

# Compute solution
A = assemble(a)
b = assemble(L)
for condition in bcs: condition.apply(A, b)

# Alternative is not yet supported
#A, b = assemble_system(a, L, bc)

u = Function(V)
solve(A, u.vector(), b, 'lu')

print(mesh)

# Verification
u_exact = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
u_e = interpolate(u_exact, V)
print('Max error:', abs(u_e.vector().array() - u.vector().array()).max())

#interactive()
