"""This demo program uses of the interface to TAO solver for variational inequalities 
 to solve a contact mechanics problem in FEnics. 
 The example considers a heavy elastic circle in a box of the same size
 """
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
# Last changed: 2013-04-15
# 
from dolfin import *

if not has_tao():
    print "DOLFIN must be compiled with TAO to run this demo."
    exit(0)
     
# Create mesh (use cgal if available)
mesh = UnitCircleMesh(50)

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Create test and trial functions, and source term
u, w = TrialFunction(V), TestFunction(V)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2.0*(1.0 + nu))), Constant(E*nu/((1.0 + nu)*(1.0 -2.0*nu)))
f = Constant((0.,-.1))

# Stress and strains
def eps(u):
    return sym(grad(u))

def sigma(epsilon):
    return  2*mu*epsilon + lmbda*tr(epsilon)*Identity(w.cell().d)

# Weak formulation
F = inner(sigma(eps(u)), eps(w))*dx - dot(f, w)*dx

# Extract bilinear and linear forms from F
a, L = lhs(F), rhs(F)

# Assemble the linear system
A, b = assemble_system(a, L)

# Define the constraints for the displacement 
# The displacement u must be such that the current configuration x+u
# does dot escape the xbox [xmin,xmax]x[umin,ymax]
constraint_u = Expression( ("xmax-x[0]","ymax-x[1]"), xmax =  1., ymax =  1.)
constraint_l = Expression( ("xmin-x[0]","ymin-x[1]"), xmin = -1., ymin = -1.)
u_min = interpolate(constraint_l, V)
u_max = interpolate(constraint_u, V)

# Define the function to store the solution 
usol=Function(V)

# Create the TAOLinearBoundSolver
solver=TAOLinearBoundSolver("tao_tron","tfqmr")

#Set some parameters
solver.parameters["monitor_convergence"]=True
solver.parameters["report"]=True
solver.parameters["krylov_solver"]["absolute_tolerance"] = 1e-7
solver.parameters["krylov_solver"]["relative_tolerance"] = 1e-7
solver.parameters["krylov_solver"]["monitor_convergence"]= False
info(solver.parameters,True) # Uncomment this line to see the available parameters

# Solve the problem
solver.solve(A, usol.vector(), b , u_min.vector(), u_max.vector())

# Save solution in VTK format
file = File("displacement.pvd")
file << usol

# plot the stress
stress=sigma(eps(usol))
plot(tr(stress),title="Trace of the stress tensor",mode = "color")

# plot the current configuration
plot(usol, mode = "displacement",wireframe=True, title="Displacement field")
interactive()
