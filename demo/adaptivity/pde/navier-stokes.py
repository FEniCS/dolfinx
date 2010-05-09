"""
This demo illustrates

1. Automated goal-oriented error control and adaptivity for the
stationary Navier-Stokes equations with a non-linear goal functional
and a known smooth exact solution.

2. How to use a reference value to examine the reliability of the
error estimation

3. How to use a Lagrange multiplier to uniquely determine the pressure
when no-slip conditions are prescribed for the velocity on the entire
boundary.
"""

__author__ = "Marie E. Rognes (meg@simula.no)"
__copyright__ = "Copyright (C) 2010 Marie Rognes"
__license__  = "GNU GPL version 3 or any later version"

from dolfin import *

# Material parameters
nu = 1.0e-2

# Exact solutions
u = Expression(('-(cos(pi*(x[0]))*sin(pi*(x[1])))',
                ' (cos(pi*(x[1]))*sin(pi*(x[0])))'), degree=4)
p = Expression('-0.25*(cos(2*pi*(x[0])) + cos(2*pi*(x[1])))', degree=4)

# Mesh: [-1, 1] x [-1, 1]
n = 12
mesh = UnitSquare(n, n)
scale = 2*(mesh.coordinates() - 0.5)
mesh.coordinates()[:, :] = scale

# Sources
f = Expression(("-(0.5)*sin(2*pi*x[0])*pi-cos(pi*x[0])*pi*pow(sin(x[1]*pi), 2)*sin(pi*x[0])-cos(pi*x[0])*pow(cos(x[1]*pi), 2)*pi*sin(pi*x[0])-2*cos(pi*x[0])*nu*pow(pi,2)*sin(x[1]*pi)",
                "2*cos(x[1]*pi)*nu*pow(pi, 2)*sin(pi*x[0])-(0.5)*pi*sin(2*x[1]*pi)-cos(x[1]*pi)*pi*sin(x[1]*pi)*pow(sin(pi*x[0]), 2)- pow(cos(pi*x[0]), 2)*cos(x[1]*pi)*pi*sin(x[1]*pi)"), degree=4)
f.nu = nu

# Define function spaces (Taylor-Hood)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
W = MixedFunctionSpace([V, Q, R])

# Define unknown and test function(s)
(v, q, d) = TestFunctions(W)
w_h = Function(W)
(u_h, p_h, c_h) = (as_vector((w_h[0], w_h[1])), w_h[2], w_h[3])

# Define variational forms
a = (nu*inner(grad(v), grad(u_h)) + div(v)*p_h + q*div(u_h) + q*c_h + d*p_h)*dx
a = a + inner(v, grad(u_h)*u_h)*dx
L = inner(v, f)*dx
F = a - L

# Define boundary conditions
bcs = [DirichletBC(W.sub(0), u, lambda x, on_boundary: on_boundary)]

# Define goal functional and reference
M = 0.5*dot(u_h, u_h)*dx
reference = 1.0

# Define adaptive problem
pde = AdaptiveVariationalProblem(F, bcs=bcs, goal_functional=M, u=w_h,
                                 reference=reference)
# Compute solution
TOL = 0.001
(u_h, p_h, c_h) = pde.solve(TOL).split()

mesh = u_h.function_space().mesh()
# Plot solution
plot(u_h, title="Final velocity")
plot(u, mesh=mesh, title="Exact velocity")
plot(p_h, title="Final pressure")
plot(p, mesh=mesh, title="Exact pressure")
interactive()


