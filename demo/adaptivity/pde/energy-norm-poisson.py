"""
This demo illustrates how to use automated energy-norm error control
for the Poisson equation in 2 dimensions.

This is experimental.
"""

__author__ = "Marie E. Rognes (meg@simula.no)"
__copyright__ = "Copyright (C) 2010 Marie E. Rognes"
__license__  = "GNU GPL version 3 or any later version"

from dolfin import *

# Create mesh and define function space
mesh = UnitSquare(4, 4)
V = FunctionSpace(mesh, "CG", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(0.0)
bcs = [DirichletBC(V, u0, DirichletBoundary())]

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
g = Expression("sin(5*x[0])")
a = inner(grad(v), grad(u))*dx
L = v*f*dx + v*g*ds

# Define adaptive variational problem based on energy norm error
# estimator (no goal given)
pde = AdaptiveVariationalProblem(a - L, bcs=bcs)

# Use Bank-Weiser style error estimator (just for fun)
pde.parameters["error_estimation"]["estimator"] = "bank_weiser"

TOL = 0.06
u = pde.solve(TOL)

# Plot solution
plot(u, interactive=True)
