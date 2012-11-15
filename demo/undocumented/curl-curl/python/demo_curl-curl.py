""" Eddy currents phenomena in low conducting body can be
described using electric vector potential and curl-curl operator:
   \nabla \times \nabla \times T = - \frac{\partial B}{\partial t}
Electric vector potential defined as:
   \nabla \times T = J

Boundary condition:
   J_n = 0,
   T_t=T_w=0, \frac{\partial T_n}{\partial n} = 0
which is naturaly fulfilled for zero Dirichlet BC with Nedelec (edge)
elements.
"""

# Copyright (C) 2009 Bartosz Sawicki
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
#
# First added:  2009-04-02
# Last changed: 2012-11-12

from dolfin import *

# Create mesh
mesh = UnitSphereMesh(8)

# Define function spaces
PN = FunctionSpace(mesh, "Nedelec 1st kind H(curl)", 1)
P1 = VectorFunctionSpace(mesh, "CG", 1)

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

# Solve eddy currents equation (using potential T)
solve(inner(curl(v0), curl(u0))*dx == -inner(v0, dbdt)*dx, T, bc)

# Solve density equation
solve(inner(v1, u1)*dx == dot(v1, curl(T))*dx, J)

# Plot solution
plot(J)

file=File("current_density.pvd")
file << J

# Hold plot
interactive()
