# This demo solves the time-dependent convection-diffusion equation by
# a least-squares stabilized cG(1)cG(1) method. The velocity field used
# in the simulation is the output from the Stokes (Taylor-Hood) demo.
# The sub domains for the different boundary conditions are computed
# by the demo program in src/demo/subdomains.
#
# Modified by Anders Logg, 2008

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2008-01-04"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Load mesh and create finite elements
mesh = Mesh("../../../../data/meshes/dolfin-2.xml.gz")
scalar = FiniteElement("Lagrange", "triangle", 1)
vector = VectorElement("Lagrange", "triangle", 2)

# Load subdomains and velocity
sub_domains = MeshFunction("uint", mesh, "../subdomains.xml.gz");
velocity = Function(vector, "../velocity.xml.gz");

# Initialise source function and previous solution function
f = Function(scalar, mesh, 0.0)
u0 = Function(scalar, mesh, 0.0)

# Parameters
T = 1.0
k = 0.05
t = k
c = 0.005

# Test and trial functions
v = TestFunction(scalar)
u = TrialFunction(scalar)

# Functions
u0 = Function(scalar, mesh, Vector())
u1 = Function(scalar, mesh, Vector())

# Variational problem
a = v*u*dx + 0.5*k*(v*dot(velocity, grad(u))*dx + c*dot(grad(v), grad(u))*dx)
L = v*u0*dx - 0.5*k*(v*dot(velocity, grad(u0))*dx + c*dot(grad(v), grad(u0))*dx) + k*v*f*dx

# Set up boundary condition
g = Function(mesh, 1.0)
bc = DirichletBC(g, sub_domains, 1)

# Assemble matrix
A = assemble(a, mesh)

# Output file
out_file = File("temperature.pvd")

# Time-stepping
while ( t < T ):

    # Assemble vector and apply boundary conditions
    b = assemble(L, mesh)
    bc.apply(A, b, a)
    
    # Solve the linear system
    solve(A, u1.vector(), b)
    
    # Save the solution to file
    out_file << u1

    # Move to next interval
    t += k
    u0.assign(u1)

# Plot solution
plot(u1)
