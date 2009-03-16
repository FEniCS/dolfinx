""" Steady state advection-diffusion equation,
discontinuous formulation using full upwinding.

Implemented in python from cpp demo by Johan Hake.
"""

__author__ = "Johan Hake (hake@simula.no)"
__date__ = "2008-06-19 -- 2008-12-23"
__copyright__ = "Copyright (C) 2008 Johan Hake"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

# Load mesh
mesh = Mesh("../mesh.xml.gz")

# Defining the function spaces
V_dg = FunctionSpace(mesh, "DG", 1)
V_cg = FunctionSpace(mesh, "CG", 1)
V_b  = VectorFunctionSpace(mesh, "CG", 2)

# Create velocity Function 
velocity = Function(V_b, "../velocity.xml.gz");

# Test and trial functions
v = TestFunction(V_dg)
u = TrialFunction(V_dg)

# Diffusivity
kappa = Constant(mesh, 0.0)

# Source term
f = Constant(mesh, 0.0)

# Penalty term
alpha = Constant(mesh, 5.0)

# Mesh-related functions
n = FacetNormal(mesh)
h = AvgMeshSize(mesh)

# IsOutflow facet function
of = IsOutflowFacet(velocity)

def upwind(u, b):
  return [b[i]('+')*(of('+')*u('+') + of('-')*u('-')) for i in range(len(b))]

# Bilinear form
a_int = dot(grad(v), mult(kappa, grad(u)) - mult(velocity, u))*dx

a_fac = kappa('+')*alpha('+')/h('+')*dot(jump(v, n), jump(u, n))*dS \
      - kappa('+')*dot(avg(grad(v)), jump(u, n))*dS \
      - kappa('+')*dot(jump(v, n), avg(grad(u)))*dS

a_vel = dot(jump(v, n), upwind(u, velocity))*dS + dot(mult(v, n), mult(velocity, of*u))*ds

a = a_int + a_fac + a_vel

# Linear form
L = v*f*dx

# Set up boundary condition (apply strong BCs)
g = Function(V_dg,"sin(pi*5.0*x[1])")
bc = DirichletBC(V_dg, g, DirichletBoundary(), geometric)

# Solution function
uh = Function(V_dg)

# Assemble and apply boundary conditions
A = assemble(a)
b = assemble(L)
bc.apply(A, b)

# Solve system
solve(A, uh.vector(), b)

# Project solution to a continuous function space
up = project(uh,V_cg)

file = File("temperature.pvd")
file << up

# Plot solution
plot(up, interactive=True)
