"""This demo program uses the interface to SNES solver for variational
 inequalities to solve a contact mechanics problems in FEniCS.  The
 example considers a heavy hyperelastic circle in a box of the same
 size"""

# Copyright (C) 2012 Corrado Maurini
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import *
import matplotlib.pyplot as plt


# This demo requires PETSc
if not has_petsc():
    print("DOLFIN must be compiled with PETSc to run this demo.")
    exit(0)

# Create mesh
mesh = Mesh("../circle_yplane.xml.gz")

V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.05))      # Body force per unit volume

# Kinematics
I = Identity(len(u))  # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx

# Compute first variation of Pi (directional derivative about u in the
# direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Symmetry condition (to block rigid body rotations)
tol = mesh.hmin()
def symmetry_line(x):
    return abs(x[0]) < DOLFIN_EPS
bc = DirichletBC(V.sub(0), 0., symmetry_line, method="pointwise")

# The displacement u must be such that the current configuration x+u
# remains in the box [xmin,xmax] x [umin,ymax]
constraint_u = Expression(("xmax - x[0]","ymax - x[1]"),
                          xmax=1.0+DOLFIN_EPS,  ymax=1.0, degree=1)
constraint_l = Expression(("xmin - x[0]","ymin - x[1]"),
                          xmin=-1.0-DOLFIN_EPS, ymin=-1.0, degree=1)
umin = interpolate(constraint_l, V)
umax = interpolate(constraint_u, V)

# Define the solver parameters
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "lu",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": False}}

# Set up the non-linear problem
problem = NonlinearVariationalProblem(F, u, bc, J=J)
problem.set_bounds(umin, umax)

# Set up the non-linear solver
solver  = NonlinearVariationalSolver(problem)
solver.parameters.update(snes_solver_parameters)
info(solver.parameters, True)

# Solve the problem
(iter, converged) = solver.solve()

# Check for convergence
if not converged:
    warning("This demo is a complex nonlinear problem. Convergence is not guaranteed when modifying some parameters or using PETSC 3.2.")

# Save solution in VTK format
file = File("displacement.pvd")
file << u

# plot the current configuration
plot(u, mode="displacement", wireframe=True, title="Displacement field")
plt.show()
