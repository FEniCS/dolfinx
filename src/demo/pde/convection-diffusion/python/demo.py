# This demo solves the time-dependent convection-diffusion equation by
# a least-squares stabilized cG(1)cG(1) method. The velocity field used
# in the simulation is the output from the Stokes (Taylor-Hood) demo.
# The sub domains for the different boundary conditions are computed
# by the demo program in src/demo/subdomains.
#
# Original implementation: ../cpp/main.cpp by Anders Logg
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

#
# THIS DEMO WORKS, BUT...
# it is very slow because Instant recompiles the form every timestep
#

# Load mesh and create finite elements
mesh = Mesh("../../../../../data/meshes/dolfin-2.xml.gz")
scalar = FiniteElement("Lagrange", "triangle", 1)
vector = VectorElement("Lagrange", "triangle", 2)

# Load subdomains and velocity
sub_domains = MeshFunction("uint", mesh, "../subdomains.xml.gz");
velocity = Function(vector, "../velocity.xml.gz");

# Initialise source function and previous solution function
f = Function(scalar, mesh, 0.0)
u0 = Function(scalar, mesh, 0.0)

# Parameters for time-stepping
T = 1.0
k = 0.05
t = k

# Define variational problem
# Test and trial functions
v  = TestFunction(scalar)
u1 = TrialFunction(scalar)

# Diffusion parameter
c = 0.005

# Bilinear form
a = v*u1*dx + 0.5*k*(v*dot(velocity, grad(u1))*dx + c*dot(grad(v), grad(u1))*dx)

# Set up boundary condition
g = Function(mesh, 1.0)
bc = DirichletBC(g, sub_domains, 1)

# Linear system
# Assemble matrix
A = assemble(a, mesh)

b = Vector()
x = Vector()

# FIXME: Maybe there is a better solution?
# Compile form, needed to create discrete function
(compiled_form, module, form_data) = jit(a)

# Solution vector
sol = Function(scalar, mesh, x, compiled_form)

# Output file
out_file = File("temperature.pvd")

# Time-stepping (This is super slow because Instant recompiles the form every timestep)
while ( t < T ):

    # Linear form (update RHS)
    L = v*u0*dx - 0.5*k*(v*dot(velocity, grad(u0))*dx + c*dot(grad(v), grad(u0))*dx) + k*v*f*dx

    # Assemble vector and apply boundary conditions
    b = assemble(L, mesh)
    bc.apply(A, b, compiled_form)
    
    # Solve the linear system
    solve(A, x, b)
    
    # Save the solution to file
    out_file << sol

    # Move to next interval
    t += k
    u0 = sol

# Plot solution
plot(sol)





