# This demo solves the Stokes equations, using quadratic
# elements for the velocity and first degree elements for
# the pressure (Taylor-Hood elements). The sub domains
# for the different boundary conditions used in this
# simulation are computed by the demo program in
# src/demo/mesh/subdomains.
#
# Original implementation: ../cpp/main.cpp by Anders Logg
#

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-16 -- 2007-12-03"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Load mesh and create finite elements
mesh = Mesh("../../../../../data/meshes/dolfin-2.xml.gz")
scalar = FiniteElement("Lagrange", "triangle", 1)
vector = VectorElement("Lagrange", "triangle", 2)
system = vector + scalar

# Load subdomains
sub_domains = MeshFunction("uint", mesh, "../subdomains.xml.gz")

# Function for no-slip boundary condition for velocity
class Noslip(Function):
    def __init__(self, mesh):
        Function.__init__(self, mesh)

    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0

    def rank(self):
        return 1

    def dim(self, i):
        return 2

# Function for inflow boundary condition for velocity
class Inflow(Function):
    def __init__(self, mesh):
        Function.__init__(self, mesh)

    def eval(self, values, x):
        values[0] = -sin(x[1]*DOLFIN_PI)
        values[1] = 0.0

    def rank(self):
        return 1

    def dim(self, i):
        return 2

# Create functions for boundary conditions
noslip = Noslip(mesh)
inflow = Inflow(mesh)
zero = Function(mesh, 0.0)

# Define sub systems for boundary conditions
velocity = SubSystem(0)
pressure = SubSystem(1)

# No-slip boundary condition for velocity
bc0 = DirichletBC(noslip, sub_domains, 0, velocity)

# Inflow boundary condition for velocity
bc1 = DirichletBC(inflow, sub_domains, 1, velocity)

# Boundary condition for pressure at outflow
bc2 = DirichletBC(zero, sub_domains, 2, pressure)

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

# Define variational problem
(v, q) = TestFunctions(system)
(u, p) = TrialFunctions(system)
f = Function(vector, mesh, (0.0, 0.0))

a = (dot(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
L = dot(v, f)*dx

# Set up PDE
pde = LinearPDE(a, L, mesh, bcs)

# Solve PDE
(U, P) = pde.solve().split()

# Save solution
ufile = File("velocity.xml")
ufile << U
pfile = File("pressure.xml")
pfile << P

# Save solution in VTK format
ufile_pvd = File("velocity.pvd")
ufile_pvd << U
pfile_pvd = File("pressure.pvd")
pfile_pvd << P

# Plot solution
plot(U)
plot(P)
interactive()
