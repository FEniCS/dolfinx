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
#mesh = UnitCubeMesh(1, 1, 1)
#mesh = refine(mesh)

# Define function spaces
P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
V = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)

# Define test and trial functions
v0 = TestFunction(V)
u0 = TrialFunction(V)
v1 = TestFunction(P1)
u1 = TrialFunction(P1)

# Define functions
dbdt = Expression(("0.0", "0.0", "1.0"), degree=1)
zero = Expression(("0.0", "0.0", "0.0"), degree=1)

T = Function(V)
J = Function(P1)

# Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Boundary condition
bc = DirichletBC(V, zero, DirichletBoundary())

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

# Get the preconditioner and set type
pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

opts = PETSc.Options()
opts.setValue("-ksp_monitor_true_residual", None)

# Build discrete gradient
P1s = FunctionSpace(mesh, "Lagrange", 1)
G = DiscreteOperators.build_gradient(V, P1s)
print "G Norm: ", G.norm("frobenius")

# Attach discrete gradient to preconditioner
G = as_backend_type(G)
pc.setHYPREDiscreteGradient(G.mat())

# Inform preconditioner of constants in the Nedelec space
constants = [Function(V) for i in range(3)]
for i, c in enumerate(constants):
    direction = [1 if i == j else 0 for j in range(3)]
    c.interpolate(Constant(direction))

cvecs = [as_backend_type(constant.vector()).vec() for constant in constants]

pc.setHYPRESetEdgeConstantVectors(cvecs[0], cvecs[1], cvecs[2])

pc.setHYPRESetBetaPoissonMatrix(None)

# Set operator
ksp.setOperators(as_backend_type(A).mat(), as_backend_type(A).mat())

# Solve
ksp.setFromOptions()
ksp.solve(as_backend_type(b).vec(), as_backend_type(T.vector()).vec())

# Show solver details
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
