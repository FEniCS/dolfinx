# Efficiency test  

__author__    = "Kent-Andre Mardal"
__date__      = "2008"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"
__license__   = "GNU LGPL Version 2.1"



from dolfin import *

dolfin_set("linear algebra backend", "uBLAS")


# Create mesh and finite element
mesh = UnitSquare(300,300)

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
#f = Source(element, mesh)
f = Function(element, mesh, 1.0)
#g = Function(element, mesh, 1.0)



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

# Solve PDE
#pde = LinearPDE(a, L, mesh, bc)
#u = pde.solve()

import time

t0 = time.time()
A = assemble(a, mesh)
b = assemble(L, mesh)
bc.apply(A, b, a)
t1 = time.time()
print "time for standard assembly ", t1-t0
 

t0 = time.time()
A, b = assemble_system(a, L, bc, mesh)
t1 = time.time()
print "time for new assembly      ", t1-t0


#x = b.copy()
#x.zero()
#solve(A, x, b)


#u = Function(element, mesh, x)

# Project u
#P1 = FiniteElement("Lagrange", "triangle", 1)
#u_proj = project(u, P1)

# Save solution to file
#file = File("poisson.pvd")
#file << u_proj

# Plot solution
#plot(u_proj, interactive=True)

