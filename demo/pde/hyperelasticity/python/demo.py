""" This demo program solves a hyperelastic problem. It is implemented
in Python by Johan Hake following the C++ demo by Harish Narayanan"""

__author__ = "Johan Hake (hake@simula.no)"
__date__ = "2009-10-11 -- 2009-10-11"
__copyright__ = "Copyright (C) 2008 Johan Hake"
__license__  = "GNU LGPL Version 2.1"

# Modified by Harish Narayanan, 2009.
# Modified by Garth N. Wells, 2010.

from dolfin import *

# Optimize compilation of the form
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"]     = True

# Create mesh and define function space
mesh = UnitCube(8, 8, 8)
V = VectorFunctionSpace(mesh, "CG", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0", "0.0"))
r = Expression(("0.0",
                "y0 + (x[1] - y0) * cos(theta) - (x[2] - z0) * sin(theta) - x[1]",
                "z0 + (x[1] - y0) * sin(theta) + (x[2] - z0) * cos(theta) - x[2]"),
                defaults = dict(y0 = 0.5, z0 = 0.5, theta = pi / 3))

left, right = compile_subdomains(["(fabs(x[0]) < DOLFIN_EPS) && on_boundary",
                                  "(fabs(x[0] - 1.0) < DOLFIN_EPS) && on_boundary"])

bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, r, right)

# Define variational problem
v  = TestFunction(V)           # Test function
du = TrialFunction(V)          # Incremental displacement
u  = Function(V)               # Displacement from previous iteration
B  = Constant((0.0, 0.0, 0.0))  # Body force per unit mass
T  = Constant((0.0, 0.0, 0.0)) # Traction force on the boundary

# Kinematics
I = Identity(V.cell().d)    # Identity tensor
F = I + grad(u)            # Deformation gradient
E = (F.T*F - I)/2          # Green-Lagrange strain tensor

# Material constants
Em = 10.0
nu = 0.3
mu    = Constant(Em / (2*(1 + nu))) # Lame's constants
lmbda = Constant(Em * nu / ((1 + nu) * (1 - 2 * nu)))

# Strain energy function
psi = (lmbda/2*(tr(E)**2) + mu*tr(E*E))*dx - inner(B, u)*dx - inner(T, u)*ds

# Take directional derivative about u in the direction of v
L = derivative(psi, u, v)

# Compute Jacobian
a = derivative(L, u, du)

# Solve nonlinear variational problem
problem = VariationalProblem(a, L, [bcl, bcr], nonlinear = True)
problem.solve(u)

# Save solution in VTK format
file = File("displacement.pvd");
file << u;

# Plot and hold solution
plot(u, mode = "displacement", interactive = True)
