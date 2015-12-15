"""Eddy currents phenomena in low conducting body can be described
using electric vector potential and curl-curl operator:

   \nabla \times \nabla \times T = - \frac{\partial B}{\partial t}

Electric vector potential defined as:

   \nabla \times T = J

Boundary condition:

   J_n = 0,

   T_t = T_w = 0, \frac{\partial T_n}{\partial n} = 0

which is naturally fulfilled for zero Dirichlet BC with Nedelec (edge)
elements.

This demo uses the auxiliary Maxwell space multigrid preconditioner
from HYPRE (via PETSc). To run this demo, PETSc must be configured
with HYPRE, and petsc4py must be available.

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

# Check that DOLFIN has been configured with PETSc
if not has_petsc():
    print("This demo requires DOLFIN to be configured with PETSc.")
    exit()

# Check that PETSc has been configured with HYPRE
if not "hypre_amg" in PETScPreconditioner.preconditioners():
    print("This demo requires PETSc to be configured with HYPRE.")
    exit()

if not has_petsc4py():
    print("DOLFIN has not been compiled with petsc4py support.")
    exit()

# Import petsc4py and check that HYPRE bindings are available
from petsc4py import *
try:
    getattr(PETSc.PC, 'getHYPREType')
except AttributeError:
    print("This demo requires a recent petsc4py with HYPRE bindings.")
    exit()

# Set PETSc as default linear algebra backend
parameters["linear_algebra_backend"] = "PETSc";

# Load sphere mesh
mesh = Mesh("../sphere.xml.gz")
mesh = refine(mesh)

# Define function spaces
V = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)

# Define test and trial functions
v = TestFunction(V)
u = TrialFunction(V)

# Define functions for boundary condiitons
dbdt = Constant((0.0, 0.0, 1.0))
zero = Constant((0.0, 0.0, 0.0))

# Magnetic field (to be computed)
T = Function(V)

# Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Boundary condition
bc = DirichletBC(V, zero, DirichletBoundary())

# Forms for the eddy-current equation
a = inner(curl(v), curl(u))*dx
L = -inner(v, dbdt)*dx

# Assemble system
A, b = assemble_system(a, L, bc)

# Create PETSc Krylov solver (from petsc4py)
ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)

# Set the Krylov solver type and set tolerances
ksp.setType("cg")
ksp.setTolerances(rtol=1.0e-8, atol=1.0e-12, divtol=1.0e10, max_it=300)

# Get the preconditioner and set type (HYPRE AMS)
pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

# Build discrete gradient
P1 = FunctionSpace(mesh, "Lagrange", 1)
G = DiscreteOperators.build_gradient(V, P1)

# Attach discrete gradient to preconditioner
pc.setHYPREDiscreteGradient(as_backend_type(G).mat())

# Build constants basis for the Nedelec space
constants = [Function(V) for i in range(3)]
for i, c in enumerate(constants):
    direction = [1.0 if i == j else 0.0 for j in range(3)]
    c.interpolate(Constant(direction))

# Inform preconditioner of constants in the Nedelec space
cvecs = [as_backend_type(constant.vector()).vec() for constant in constants]
pc.setHYPRESetEdgeConstantVectors(cvecs[0], cvecs[1], cvecs[2])

# We are dealing with a zero conductivity problem (no mass term), so
# we need to tell the preconditioner
pc.setHYPRESetBetaPoissonMatrix(None)

# Set operator for the linear solver
ksp.setOperators(as_backend_type(A).mat())

# Set options prefix
ksp.setOptionsPrefix("eddy_")

# Turn on monitoring of residual
opts = PETSc.Options()
opts.setValue("-eddy_ksp_monitor_true_residual", None)

# Solve eddy currents equation (using potential T)
ksp.setFromOptions()
ksp.solve(as_backend_type(b).vec(), as_backend_type(T.vector()).vec())

# Show linear solver details
ksp.view()

# Test and trial functions for density equation
W = VectorFunctionSpace(mesh, "Lagrange", 1)
v = TestFunction(W)
u = TrialFunction(W)

# Solve density equation (use conjugate gradient linear solver)
J = Function(W)
solve(inner(v, u)*dx == dot(v, curl(T))*dx, J,
      solver_parameters={"linear_solver": "cg"} )

# Write solution to PVD/ParaView file
file = File("current_density.pvd")
file << J

# Plot solution and hold plot
plot(J)
interactive()
