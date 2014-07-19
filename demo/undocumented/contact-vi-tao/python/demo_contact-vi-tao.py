"""This demo uses PETSc's TAO solver for nonlinear (bound-constrained)
optimisation problems to solve a contact mechanics problem in FEniCS.
We consider here a heavy hyperelastic circle under body force
in a box of the same size."""

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
# Modified by Tianyi Li 2014
#
# First added:  2012-09-03
# Last changed: 2014-07-19

from dolfin import *

if not has_petsc_tao():
    print "DOLFIN must be compiled with TAO to run this demo."
    exit(0)

# Read mesh
mesh = Mesh("../circle_yplane.xml.gz")

# Create function space
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Create solution, trial and test functions
u, du, v = Function(V), TrialFunction(V), TestFunction(V)

# Elasticity parameters
E, nu = 10.0, 0.3
mu = Constant(E/(2.0*(1.0+nu)))
lmbda = Constant(E*nu/((1.0+nu)*(1.0-2.0*nu)))

# Compressible neo-Hookean model
I = Identity(mesh.geometry().dim())
F = I + grad(u)
C = F.T*F
Ic = tr(C)
J  = det(F)
psi = (mu/2)*(Ic-2)-mu*ln(J)+(lmbda/2)*(ln(J))**2

# Body force (you can play with its intensity (try -10.0), TAO is robust enough)
f = Constant((0.0, -1.0))

# Variational formulation
elastic_energy = psi*dx - dot(f, u)*dx
grad_elastic_energy = derivative(elastic_energy, u, v)
H_elastic_energy = derivative(grad_elastic_energy, u, du)

# Define the minimisation problem by using OptimisationProblem class
class ContactProblem(OptimisationProblem):

    def __init__(self):
        OptimisationProblem.__init__(self)
    
    # Objective function
    def f(self, x):
        u.vector()[:] = x
        return assemble(elastic_energy)

    # Gradient of the objective function
    def F(self, b, x):
        u.vector()[:] = x
        assemble(grad_elastic_energy, tensor=b)

    # Hessian of the objective function
    def J(self, A, x):
        u.vector()[:] = x
        assemble(H_elastic_energy, tensor=A)

ContactPb = ContactProblem()

# The displacement u must be such that the current configuration
# doesn't escape the box [xmin, xmax] x [ymin, ymax]
constraint_u = Expression(("xmax-x[0]", "ymax-x[1]"), xmax=1.0, ymax=1.0)
constraint_l = Expression(("xmin-x[0]", "ymin-x[1]"), xmin=-1.0, ymin=-1.0)
u_min = interpolate(constraint_l, V)
u_max = interpolate(constraint_u, V)

# Symmetry condition (to block rigid body rotations)
def symmetry_line(x, on_boundary):
    return near(x[0], 0)
bc = DirichletBC(V.sub(0), Constant(0.0), symmetry_line)
bc.apply(u_min.vector())  # BC will be incorporated into the lower
bc.apply(u_max.vector())  # and upper bounds

# Create the PETScTAOSolver
solver = PETScTAOSolver("tron", "stcg")

# Set some parameters
solver.parameters["monitor_convergence"] = True
solver.parameters["report"] = True

# Uncomment this line to see the available parameters
# info(parameters, True)

# Solve the problem
solver.solve(ContactPb, u.vector(), u_min.vector(), u_max.vector())

# Save solution in XDMF format
out = File("u.xdmf")
out << u

# Plot the current configuration
plot(u, mode="displacement", wireframe=True, title="Displacement field")
interactive()