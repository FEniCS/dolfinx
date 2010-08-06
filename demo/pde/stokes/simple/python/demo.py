"""This demo solves the Stokes equations, using quadratic elements for
the velocity and first degree elements for the pressure (Taylor-Hood
elements). The sub domains for the different boundary conditions
used in this simulation are computed by the demo program in
src/demo/mesh/subdomains."""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-16 -- 2009-11-26"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008-2009.

from dolfin import *

# Load mesh and subdomains
mesh = UnitSquare(32, 32)

# Define function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# Boundaries
def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS) and on_boundary
def left(x, on_boundary): return x[0] < DOLFIN_EPS and on_boundary
def top_bottom(x, on_boundary):
    return (x[1] > (1.0 - DOLFIN_EPS) and on_boundary) or (x[1] < DOLFIN_EPS and on_boundary)

# No-slip boundary condition for velocity
noslip = Constant((0, 0))
bc0 = DirichletBC(W.sub(0), noslip, top_bottom)

# Inflow boundary condition for velocity
inflow = Expression(("-sin(x[1]*pi)", "0.0"))
bc1 = DirichletBC(W.sub(0), inflow, right)

# Boundary condition for pressure at outflow
zero = Constant(0)
bc2 = DirichletBC(W.sub(1), zero, left)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(v, q) = TestFunctions(W)
(u, p) = TrialFunctions(W)
f = Constant((0, 0))
a = (inner(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
L = inner(v, f)*dx

# Compute solution
problem = VariationalProblem(a, L, bcs)
U = problem.solve()

# Split the mixed solution using deepcopy
# (needed for further computation on coefficient vector)
(u, p) = U.split(True)

print "Norm of velocity coefficient vector: %.15g" % u.vector().norm("l2")
print "Norm of pressure coefficient vector: %.15g" % p.vector().norm("l2")

# Split the mixed solution using a shallow copy
(u, p) = U.split()

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << u
pfile_pvd = File("pressure.pvd")
pfile_pvd << p

# Plot solution
plot(u)
plot(p)
interactive()
