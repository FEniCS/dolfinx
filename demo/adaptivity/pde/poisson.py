"""
This demo illustrates automated goal-oriented error control and
adaptivity for the Poisson equation in three dimensions.
"""

__author__ = "Marie E. Rognes (meg@simula.no)"
__copyright__ = "Copyright (C) 2010 Marie E. Rognes"
__license__  = "GNU GPL version 3 or any later version"

from dolfin import *

# Define Dirichlet boundary (x = 0, 1 or y = 1)
def dirichlet_boundary(x):
    return (x[0] < DOLFIN_EPS or x[1] < DOLFIN_EPS
            or x[2] < DOLFIN_EPS or abs(x[2] - 1.0) < DOLFIN_EPS)

# Create mesh and define function space
mesh = UnitCube(4, 4, 4)
V = FunctionSpace(mesh, "CG", 1)

# Define boundary condition
bcs = [DirichletBC(V, Constant(0.0), dirichlet_boundary)]

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)

f = Expression("pow(pi, 2)*x[0]*x[1]*sin(pi*x[2])", degree=3)
G = Expression(("x[1]*sin(pi*x[2])",
                "x[0]*sin(pi*x[2])",
                "pi*x[0]*x[1]*cos(pi*x[2])"), degree=3)

a = inner(grad(v), grad(u))*dx
n = FacetNormal(mesh)
L = v*f*dx + v*dot(G, n)*ds

# Define goal functional (average value over bulk)
M = u*dx
value = 1.0/(2*pi)

# Initialize adaptive pde
pde = AdaptiveVariationalProblem(a - L, bcs=bcs, goal_functional=M,
                                 reference=value)

# Compute solution given tolerance TOL
TOL = 0.0015
u = pde.solve(TOL)

# Plot solution
plot(u)
interactive()
