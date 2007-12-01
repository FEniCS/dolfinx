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
__date__ = "2007-11-16 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

#
# THIS DEMO IS CURRENTLY NOT WORKING, SEE NOTE IN CODE.
#

# Load mesh and create finite elements
mesh = Mesh("../../../../../../data/meshes/dolfin-2.xml.gz")
scalar = FiniteElement("Lagrange", "triangle", 1)
vector = VectorElement("Lagrange", "triangle", 2)
system = vector + scalar

# Load subdomains
sub_domains = MeshFunction("uint", mesh, "../subdomains.xml.gz");

# Function for no-slip boundary condition for velocity
class Noslip(Function):
    def __init__(self, mesh):
        Function.__init__(self, mesh)

    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
# ERROR:
# terminate called after throwing an instance of 'Swig::DirectorMethodException'
# Aborted (core dumped)

# It looks like vector valued functions are not supported. If values[1] and values[2]
# are commented out, the error doesn't occur (the results are not correct of course)
# Same error as in the elasticity demo


# Function for inflow boundary condition for velocity
class Inflow(Function):
    def __init__(self, mesh):
        Function.__init__(self, mesh)

    def eval(self, values, x):
        values[0] = -1.0
        values[1] = 0.0
# ERROR:
# terminate called after throwing an instance of 'Swig::DirectorMethodException'
# Aborted (core dumped)

# It looks like vector valued functions are not supported. If values[1] and values[2]
# are commented out, the error doesn't occur (the results are not correct of course)
# Same error as in the elasticity demo


# Create functions for boundary conditions
#noslip = Noslip(vector, mesh)
#inflow = Inflow(vector, mesh)
#zero = Function(scalar, mesh, 0.0)
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


# # Define variational problem
(v, q) = TestFunctions(system)
(u, p) = TrialFunctions(system)
f = Function(vector, mesh, 0.0)

a = (dot(grad(v), grad(u)) - div(v)*p + q*div(u))*dx
L = dot(v, f)*dx

# Set up PDE
pde = LinearPDE(a, L, mesh, bcs)

#   // Solve PDE
#u = Function(mesh)
#p = Function(mesh)
(U, P) = pde.solve().split()

#   // Plot solution
#   plot(u);
#   plot(p);


# # Solve PDE and plot solution
# pde = LinearPDE(a, L, mesh, bcs)
# u = pde.solve()
# plot(u)

# Save solution to file
# file = File("poisson.pvd")
# file << u







