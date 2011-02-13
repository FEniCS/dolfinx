__author__ = "Marie E. Rognes (meg@simula.no) and Anders Logg (logg@simula.no)"
__copyright__ = "Copyright (C) 2011 Marie Rognes and Anders Logg"
__license__  = "GNU LGPL version 3 or any later version"

# Begin demo

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(8, 8)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition
u0 = Function(V)
bc = DirichletBC(V, u0, "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)",
               degree=1)
g = Expression("sin(5*x[0])", degree=1)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds
problem = VariationalProblem(a, L, bc)

# Define goal (quantity of interest)
u = Function(V) # FIXME
M = u*dx

# Compute solution (adaptively) with accuracy to within tol
tol = 1.e-5
problem.solve(u, tol, M)
