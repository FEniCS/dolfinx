"""Eddy currents phenomena in low conducting body can be described
using electric vector potential and curl-curl operator:

   \nabla \times \nabla \times T = - \frac{\partial B}{\partial t}

Electric vector potential defined as:

   \nabla \times T = J

Boundary condition:

   J_n = 0,

   T_t=T_w=0, \frac{\partial T_n}{\partial n} = 0

which is naturaly fulfilled for zero Dirichlet BC with Nedelec (edge)
elements.

"""

# Copyright (C) 2009-2015 Bartosz Sawicki and Garth N. Wells
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
# Modified by Anders Logg 2011

from dolfin import *
from petsc4py import *

# Set PETSc as default linear algebra backend
parameters["linear_algebra_backend"] = "PETSc";

# Load sphere mesh
mesh = Mesh("../sphere.xml.gz")
mesh = refine(mesh)

# Define function spaces
P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
PN = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)

# Define test and trial functions
v0 = TestFunction(PN)
u0 = TrialFunction(PN)
v1 = TestFunction(P1)
u1 = TrialFunction(P1)

# Define functions
dbdt = Expression(("0.0", "0.0", "1.0"), degree=1)
zero = Expression(("0.0", "0.0", "0.0"), degree=1)

T = Function(PN)
J = Function(P1)

# Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Boundary condition
bc = DirichletBC(PN, zero, DirichletBoundary())

# Forms for the eddy-current equation
a = inner(curl(v0), curl(u0))*dx
L = -inner(v0, dbdt)*dx

# Assemble system
A, b = assemble_system(a, L, bc)

# Create Krylov solver
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setType("cg")
ksp.setTolerances(rtol=1.0e-8, atol=1.0e-12, divtol=1.0e10, max_it=300)

#def monitor(ksp, its, rnorm):
#   print "Ooosp", its, rnorm

#ksp.setMonitor(monitor)

# Get the preconditioner and set type
pc = ksp.getPC()
pc.setType("hypre")

opts = PETSc.Options()
opts.setValue("-pc_hypre_type", "ams")
pc.setFromOptions()

#opts.setValue("-ksp_view", True)

# Build discrete gradient
P1s = FunctionSpace(mesh, "Lagrange", 1)
G = DiscreteOperators.build_gradient(PN, P1s)

# Attach discrete gradient to preconditioner
G = as_backend_type(G)
pc.setHYPREDiscreteGradient(G.mat())

# Inform preconditioner of constants in the Nedelec space
constants = [Function(PN)]*3;
constants[0].interpolate(Constant((1.0, 0.0, 0.0)));
constants[1].interpolate(Constant((0.0, 1.0, 0.0)));
constants[2].interpolate(Constant((0.0, 0.0, 1.0)));

cvecs = [as_backend_type(constant.vector()).vec() for constant in constants]
pc.setHYPRESetEdgeConstantVectors(cvecs[0], cvecs[1], cvecs[2])

pc.setHYPRESetBetaPoissonMatrix()

# Set operator
ksp.setOperators(as_backend_type(A).mat())

#opts.setValue("-eddy_ksp_monitor_true_residual", True)
#ksp.setOptionsPrefix("eddy_")
#opts.setValue("-ksp_view", True)
#ksp.setFromOptions()

#print(PETSc.Options().getAll())
ksp.setConvergenceHistory()
#ksp.setFromOptions()
ksp.solve(as_backend_type(b).vec(), as_backend_type(T.vector()).vec())

history = ksp.getConvergenceHistory()
print(history)
ksp.view()

#print("Test norm: {}".format(T.vector().norm("l2")))

# Solve eddy currents equation (using potential T)
#solve(inner(curl(v0), curl(u0))*dx == -inner(v0, dbdt)*dx, T, bc)

# Solve density equation
#solve(inner(v1, u1)*dx == dot(v1, curl(T))*dx, J)

# Plot solution
#plot(J)

#file=File("current_density.pvd")
#file << J

# Hold plot
#interactive()
