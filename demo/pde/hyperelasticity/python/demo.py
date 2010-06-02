""" This demo program solves a hyperelastic problem. It is implemented
in Python by Johan Hake following the C++ demo by Harish Narayanan"""

__author__ = "Johan Hake (hake@simula.no)"
__date__ = "2009-10-11 -- 2009-10-11"
__copyright__ = "Copyright (C) 2008 Johan Hake"
__license__  = "GNU LGPL Version 2.1"

# Modified by Harish Narayanan, 2009.

from dolfin import *

# Optimize compilation of the form
parameters.optimize = True

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
v  = TestFunction(V)      # Test function
du = TrialFunction(V)     # Incremental displacement
u  = Function(V)          # Displacement from previous iteration
B  = Expression(("0.0", "0.0", "0.0"))          # Body force per unit mass
T  = Expression(("0.0", "0.0", "0.0"))          # Traction force on the boundary

# Kinematics
I = Identity(v.cell().d)        # Identity tensor
F = I + grad(u)                 # Deformation gradient
C = F.T*F                       # Right Cauchy-Green tensor
E = (C - I)/2                   # Euler-Lagrange strain tensor
E = variable(E)

# Material constants
Em = 10.0
nu = 0.3

mu    = Constant(Em / (2*(1 + nu))) # Lame's constants
lmbda = Constant(Em * nu / ((1 + nu) * (1 - 2 * nu)))

# Strain energy function (material model)
psi = lmbda/2*(tr(E)**2) + mu*tr(E*E)

S = diff(psi, E)                # Second Piola-Kirchhoff stress tensor
P = F*S                         # First Piola-Kirchoff stress tensor

# The variational problem corresponding to hyperelasticity
L = inner(P, grad(v))*dx - inner(B, v)*dx - inner(T, v)*ds
a = derivative(L, u, du)

# Solve nonlinear variational problem
problem = VariationalProblem(a, L, [bcl, bcr], nonlinear = True)
problem.solve(u)

# Save solution in VTK format
file = File("displacement.pvd");
file << u;

# Plot and hold solution
plot(u, mode = "displacement", interactive = True)
