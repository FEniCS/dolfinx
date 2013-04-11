"""This demo program uses of the interface to SNES solver for variational inequalities 
 to solve a contact mechanics problems in FEnics. 
The example considers a heavy hyperelastic circle in a box of the same size"""
# Copyright (C) 2012 Corrado Maurini
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# Modified by Corrado Maurini 2013
#
# First added:  2012-09-03
# Last changed: 2013-04-11
# 
from dolfin import *
    
# Create mesh (use cgal if available)
if has_cgal():
    circle = Circle (0, 0, 1);
    mesh = Mesh(circle,30)
else:
    mesh = UnitCircleMesh(30)

V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.05))      # Body force per unit volume

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
psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx 

# Compute first variation of Pi (directional derivative about u in the direction of v)
F = derivative(Pi, u, v)

# Compute Jacobian of F
J = derivative(F, u, du)

# Symmetry condition (to avoid rotations)
tol=mesh.hmin()
def symmetry_line(x):
    return abs(x[0]) < DOLFIN_EPS
bc = DirichletBC(V.sub(0), 0., symmetry_line,method="pointwise")

# The displacement u must be such that the current configuration x+u
# does dot escape the xbox [xmin,xmax]x[umin,ymax]
constraint_u = Expression( ("xmax-x[0]","ymax-x[1]"), xmax =  1+DOLFIN_EPS, ymax =  1.)
constraint_l = Expression( ("xmin-x[0]","ymin-x[1]"), xmin = -1-DOLFIN_EPS, ymin = -1.)
u_min = interpolate(constraint_l, V)
u_max = interpolate(constraint_u, V)

# Take the PETScVector of the solution function
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "linear_solver"   : "lu",
                          "snes_solver"     : { "maximum_iterations": 20,
                                                "report": True,
                                                "error_on_nonconvergence": False,
                                               }}

problem = NonlinearVariationalProblem(F, u,bc,J=J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters.update(snes_solver_parameters)
info(solver.parameters,True)
(iter,converged)=solver.solve(u_min,u_max)

# Check for convergence. Convergence is one modifies the loading and the mesh size. 
if not converged: 
   warning("This demo is a complex nonlinear problem. Convergence is not garanteed if you modify some parameters or use PETSC 3.2.")
    
# Save solution in VTK format
file = File("displacement.pvd")
file << u

# plot the current configuration
plot(u, mode = "displacement",wireframe=True, title="Displacement field")
interactive()
