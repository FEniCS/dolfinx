""" This demo program solves a hyperelastic problem. It is implemented
in Python by Johan Hake following the C++ demo by Harish Narayanan"""

__author__ = "Johan Hake (hake@simula.no)"
__date__ = "2009-10-11 -- 2010-08-28"
__copyright__ = "Copyright (C) 2008-2010 Johan Hake and Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

# Modified by Harish Narayanan, 2009.

# Begin demo

from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
mesh = UnitCube(16, 16, 16)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomians
left, right = compile_subdomains(["(std::abs(x[0])       < DOLFIN_EPS) && on_boundary",
                                  "(std::abs(x[0] - 1.0) < DOLFIN_EPS) && on_boundary"])

# Define Dirichlet boundary (x = 0 or x = 1)
c = Expression(("0.0", "0.0", "0.0"))
r = Expression(("scale*0.0",
                "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
                "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
                defaults = dict(scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3))

bcl = DirichletBC(V, c, left)
bcr = DirichletBC(V, r, right)

# Define functions
v  = TestFunction(V)             # Test function
du = TrialFunction(V)            # Incremental displacement
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.5, 0.0))  # Body force per unit mass
T  = Constant((0.1,  0.0, 0.0))  # Traction force on the boundary

# Kinematics
I = Identity(V.cell().d)    # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
L = derivative(Pi, u, v)

# Compute Jacobian of L
a = derivative(L, u, du)

# Create nonlinear variational problem and solve
problem = VariationalProblem(a, L, [bcl, bcr], nonlinear = True, form_compiler_parameters = ffc_options)
problem.solve(u)

# Save solution in VTK format
file = File("displacement.pvd");
file << u;

# Plot and hold solution
plot(u, mode = "displacement", interactive = True)
