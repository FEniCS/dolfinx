"""
This demo illustrates automated goal-oriented error control and
adaptivity for a mixed formulation using H(div) and L^2 conforming
finite element spaces for the Poisson equation in two dimensions.

However, it mainly illustrates how to adjust parameters for the
adaptive variational problem (hence the length). The default
parameters are the recommended parameters, hence this demo is not
particularly efficient.
"""

__author__ = "Marie E. Rognes (meg@simula.no)"
__copyright__ = "Copyright (C) 2010 Marie Rognes"
__license__  = "GNU GPL version 3 or any later version"

from dolfin import *

# Create mesh and define function spaces
n = 6
mesh = UnitSquare(n, n)

RT = FunctionSpace(mesh, "RT", 2)
DG = FunctionSpace(mesh, "DG", 1)
V = RT * DG

# Define source
u = Expression('x[1]*sin(pi*x[0])', degree=3)
f = Expression('x[1]*pow(pi, 2)*sin(pi*x[0])', degree=3)

# Define variational problem
(tau, v) = TestFunctions(V)
(sigma_h, u_h) = TrialFunctions(V)
n = FacetNormal(mesh)
a = (inner(tau, sigma_h) - div(tau)*u_h + v*div(sigma_h))*dx
L = - dot(tau, n)*u*ds + v*f*dx

# Define goal functional
M = f*sigma_h[1]*dx

# Define problem
pde = AdaptiveVariationalProblem(a - L, goal_functional=M)

# Use a higher-order strategy for computing the dual (default is
# "extrapolation")
pde.parameters["error_estimation"]["dual_strategy"] = "higher_order"

# Use a more robust error estimate (default is "error_representation")
pde.parameters["error_estimation"]["estimator"] = "dual_weighted_residual"

# Increase the number of max_iterations (default is 20)
pde.parameters["max_iterations"] = 100

# Do not plot error indicators
pde.parameters["plot_indicators"] = False

# Save error indicators, meshes and adaptive data to "tmp...". Default
# is "indicators...". Do not use same name as script.
pde.parameters["save_indicators"] = "mixed-poisson-data"

# Linear solver for primal and dual can be adjusted with this
# parameter
pde.parameters["linear_solver"] = "direct"

# Use different mesh marking and refinement fraction
pde.parameters["marking"]["strategy"] = "fixed_fraction"
pde.parameters["marking"]["fraction"] = 0.1

# Solve to given TOL
TOL = 1.e-3
(sigma_h, u_h) = pde.solve(TOL).split()

u_L2 = errornorm(u, u_h, "L2")
print "||u - u_h||_0 = ", u_L2

# Plot solution
mesh = u_h.function_space().mesh()
plot(project(u_h, FunctionSpace(mesh, "CG", 1)))
interactive()
