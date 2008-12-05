"""This demo solves the Stokes equations, using quadratic elements for
the velocity and first degree elements for the pressure (Taylor-Hood
elements). The sub domains for the different boundary conditions
used in this simulation are computed by the demo program in
src/demo/mesh/subdomains."""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-16 -- 2008-12-05"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

# Load mesh
mesh = Mesh("../../../../../data/meshes/dolfin-2.xml.gz")

# Define function spaces
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)
W = V + Q

# FIXME: Does not seem optimal, first we put them together, than extract subspaces
# Define subspaces
Vu = SubSpace(V, 0)
Vp = SubSpace(V, 1)

# Load subdomains
sub_domains = MeshFunction("uint", mesh, "../subdomains.xml.gz")

# FIXME: Replace by simple Constant
# Function for no-slip boundary condition for velocity
class Noslip(Function):

    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def rank(self):
        return 1

    def dim(self, i):
        return 2

# FIXME: Replace by simple cppexpr
# Function for inflow boundary condition for velocity
class Inflow(Function):

    def eval(self, values, x):
        values[0] = -sin(x[1]*DOLFIN_PI)
        values[1] = 0.0

    def rank(self):
        return 1

    def dim(self, i):
        return 2

# Create functions for boundary conditions
noslip = Noslip(V)
inflow = Inflow(V)
zero = Constant("triangle", 0.0)

# No-slip boundary condition for velocity
bc0 = DirichletBC(noslip, V, sub_domains, 0)

# Inflow boundary condition for velocity
bc1 = DirichletBC(inflow, V, sub_domains, 1)

# Boundary condition for pressure at outflow
bc2 = DirichletBC(zero, Q, sub_domains, 2)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# FIXME: Handle constants in forms

# Define variational problem
(v, q) = TestFunctions(W)
(u, p) = TrialFunctions(W)
f = Function(V, cppexpr=("0", "0", "0"))
a = (dot(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
L = dot(v, f)*dx

# Solve PDE
pde = LinearPDE(a, L, bcs)

w = pde.solve()
#(u, p) = pde.solve().split()

# Save solution in DOLFIN XML format
#ufile = File("velocity.xml")
#ufile << u
#pfile = File("pressure.xml")
#pfile << p

# Save solution in VTK format
#ufile_pvd = File("velocity.pvd")
#ufile_pvd << u
#pfile_pvd = File("pressure.pvd")
#pfile_pvd << p

# Plot solution
#plot(u)
#plot(p)
#interactive()
