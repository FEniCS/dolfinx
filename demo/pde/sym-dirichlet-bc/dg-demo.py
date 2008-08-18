# This demo program solves Poisson's using 
# using a discontinuous Galerkin formulation 
# and enforcing boundary conditions in a symmetric way.
# The demo builds on the standard dg demo 
# demo/pde/dg/poisson/python/demo.py

__author__    = "Kent-Andre Mardal"
__date__      = "2008"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"
__license__   = "GNU LGPL Version 2.1"


from dolfin import *

dolfin_set("linear algebra backend", "uBLAS")


# Create mesh and finite element
mesh = UnitSquare(16, 16)

# Source term
class Source(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary) 



# Define variational problem
element = FiniteElement("Discontinuous Lagrange", "triangle", 1)
v = TestFunction(element)
u = TrialFunction(element)
f = Source(element, mesh)

# Normal component, mesh size and right-hand side
n = FacetNormal("triangle", mesh)
h = AvgMeshSize("triangle", mesh)

# Parameters
alpha = 4.0
gamma = 8.0

# Define boundary condition
u0 = Function(mesh, 0.0)
boundary = DirichletBoundary()
bc = DirichletBC(u0, mesh, boundary)



# Bilinear form
a = dot(grad(v), grad(u))*dx \
   - dot(avg(grad(v)), jump(u, n))*dS \
   - dot(jump(v, n), avg(grad(u)))*dS \
   + alpha/h('+')*dot(jump(v, n), jump(u, n))*dS \
   - dot(grad(v), mult(u, n))*ds \
   - dot(mult(v, n), grad(u))*ds \
   + gamma/h*v*u*ds

# Linear form
L = v*f*dx

# Standard way of computing A and b 
A = assemble(a, mesh)
b = assemble(L, mesh)
x = b.copy()
x.zero()
solve(A, x, b)

# Project u
u = Function(element, mesh, x)
P1 = FiniteElement("Lagrange", "triangle", 1)
u_proj = project(u, P1)

# Save solution to file
file = File("poisson.pvd")
file << u_proj

# Plot solution
plot(u_proj, interactive=True)


# Symmetric way of computing A and b 
A, b = assemble_system(a, L, bc, mesh)
x = b.copy()
x.zero()
solve(A, x, b)


# Project u
u = Function(element, mesh, x)
P1 = FiniteElement("Lagrange", "triangle", 1)
u_proj = project(u, P1)

# Save solution to file
file = File("poisson.pvd")
file << u_proj

# Plot solution
plot(u_proj, interactive=True)



