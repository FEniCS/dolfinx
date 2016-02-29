# Copyright (C) 2008 Anders Logg
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
# First added:  2008-10-17
# Last changed: 2012-11-12

from __future__ import print_function
from dolfin import *

# Create mesh
mesh = UnitSquareMesh(16, 16)

# Define function spaces
P1 = FunctionSpace(mesh, "CG", 1)
P0 = FunctionSpace(mesh, "DG", 0)

# Define test and trial functions
v1 = TestFunction(P1)
w1 = TrialFunction(P1)
w0 = TestFunction(P0)

# Define functions
u  = Function(P1)
z  = Function(P1)
f  = Constant(1.0)
p  = Function(P0)
u0 = Expression("x[0]*(1.0 - x[0])*x[1]*(1.0 - x[1])", degree=2)

# Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Boundary condition
bc = DirichletBC(P1, u0, DirichletBoundary())

# Goal functional
J = (u - u0)*(u - u0)*dx(mesh)

# Forward problem
problem = (inner(grad(v1), p*grad(w1))*dx, v1*f*dx)

# Adjoint problem
adjoint = (inner(grad(w1), p*grad(v1))*dx, -2*v1*(u - u0)*dx)

# Update of parameter
gradient = -inner(grad(z), w0*grad(u))*dx
px = p.vector()

# Set initial value for parameter
px[:] = 1.0

# Iterate until convergence
for i in range(100):

    # Solve forward problem
    A = assemble(problem[0])
    b = assemble(problem[1])
    bc.apply(A, b)
    solve(A, u.vector(), b)

    # Solve adjoint problem
    A = assemble(adjoint[0])
    b = assemble(adjoint[1])
    bc.apply(A, b)
    solve(A, z.vector(), b)

    # Update parameter
    dp = assemble(gradient)
    dp *= 100.0
    px -= dp

    # Print value of functional
    jval = assemble(J)
    print("J = ", jval)
    print(u.vector().max())

# Plot solution and parameter
plot(u,  title="Solution",  rescale=True)
plot(z,  title="Adjoint",   rescale=True)
plot(p,  title="Parameter", rescale=True)
plot(u0, mesh=mesh, title="Target", rescale=True)

# Hold plot
interactive()
