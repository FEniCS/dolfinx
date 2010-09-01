"""This demo program solves the Biharmonic equation,

    nabla^4 u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 4 pi^4 sin(pi*x)*sin(pi*y)

and boundary conditions given by

    u(x, y)         = 0
    nabla^2 u(x, y) = 0

using a discontinuous Galerkin formulation (interior penalty method).
"""

__author__    = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__      = "2009-06-26 -- 2009-06-26"
__copyright__ = "Copyright (C) 2009 Kristian B. Oelgaard"
__license__   = "GNU LGPL Version 2.1"

# Begin demo

from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Create mesh and define function space
mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, "CG", 2)

# Define Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Source(Expression):
    def eval(self, values, x):
        values[0] = 4.0*pi**4*sin(pi*x[0])*sin(pi*x[1])

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Define normal component, mesh size and right-hand side
h = CellSize(mesh)
h_avg = (h('+') + h('-'))/2.0
n = FacetNormal(mesh)
f = Source(V)

# Penalty parameter
alpha = Constant(8.0)

# Define bilinear form
a = inner(div(grad(u)), div(grad(v)))*dx \
  - inner(avg(div(grad(u))), jump(grad(v), n))*dS \
  - inner(jump(grad(u), n), avg(div(grad(v))))*dS \
  + alpha('+')/h_avg*inner(jump(grad(u),n), jump(grad(v),n))*dS

# Define linear form
L = f*v*dx

# Create variational problem and solve
problem = VariationalProblem(a, L, bc)
u = problem.solve()

# Save solution to file
file = File("biharmonic.pvd")
file << u

# Plot solution
plot(u, interactive=True)
