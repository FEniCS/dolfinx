# This demo solves the time-dependent convection-diffusion equation by
# a least-squares stabilized cG(1)cG(1) method. The velocity field used
# in the simulation is the output from the Stokes (Taylor-Hood) demo.
# The sub domains for the different boundary conditions are computed
# by the demo program in src/demo/subdomains.
#
# Modified by Anders Logg, 2008
# Modified by Johan Hake, 2008
# Modified by Garth N. Wells, 2009

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2009-10-06"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Load mesh and subdomains
mesh = Mesh("../mesh.xml.gz")
sub_domains = MeshFunction("uint", mesh, "../subdomains.xml.gz");

# Create FunctionSpaces
Q = FunctionSpace(mesh, "CG", 1)
V = VectorFunctionSpace(mesh, "CG", 2)

# Create velocity Function
velocity = Function(V, "../velocity.xml.gz");

# Initialise source function and previous solution function
f  = Constant(0.0)
u0 = Function(Q)

# Parameters
T = 5.0
k = 0.1
t = k
c = 0.005

# Test and trial functions
v = TestFunction(Q)
u = TrialFunction(Q)

# Variational problem
a = v*u*dx + 0.5*k*(v*dot(velocity, grad(u))*dx + c*dot(grad(v), grad(u))*dx)
L = v*u0*dx - 0.5*k*(v*dot(velocity, grad(u0))*dx + c*dot(grad(v), grad(u0))*dx) + k*v*f*dx

# Set up boundary condition
g  = Constant(1.0)
bc = DirichletBC(Q, g, sub_domains, 1)

# Assemble matrix
A = assemble(a)

# Output file
out_file = File("temperature.pvd")

# Set intial condition
u = u0

# Time-stepping
while t < T:

    # Assemble vector and apply boundary conditions
    b = assemble(L)
    bc.apply(A, b)

    # Solve the linear system
    solve(A, u.vector(), b)

    # Copy solution from previous interval
    u0.assign(u)

    # Plot solution
    plot(u)

    # Save the solution to file
    out_file << u

    # Move to next interval
    t += k

# Hold plot
interactive()
