"""This demo program solves the mixed formulation of Poisson's
equation:

    sigma + grad(u) = 0    in Omega
         div(sigma) = f    in Omega
              du/dn = g    on Gamma_N
                  u = u_D  on Gamma_D

The corresponding weak (variational problem)

    <sigma, tau> + <grad(u), tau>   = 0
                                              for all tau
                 - <sigma, grad(v)> = <f, v> + <g, v>
                                              for all v

is solved using DRT (Discontinuous Raviart-Thomas) elements
of degree k for (sigma, tau) and CG (Lagrange) elements
of degree k + 1 for (u, v) for k >= 1.
"""

# Copyright (C) 2014 Jan Blechta
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
# First added:  2014-01-27
# Last changed: 2014-01-29

# Begin demo

from dolfin import *

# Create mesh
mesh = UnitSquareMesh(32, 32)

# Define finite elements spaces and build mixed space
DRT = FiniteElement("DRT", mesh.ufl_cell(), 2)
CG  = FiniteElement("CG", mesh.ufl_cell(), 3)
W = FunctionSpace(mesh, DRT * CG)

# Define trial and test functions
(sigma, u) = TrialFunctions(W)
(tau, v) = TestFunctions(W)

# Define source functions
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("sin(5.0*x[0])", degree=2)

# Define variational form
a = (dot(sigma, tau) + dot(grad(u), tau) + dot(sigma, grad(v)))*dx
L = - f*v*dx - g*v*ds

# Define Dirichlet BC
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS
bc = DirichletBC(W.sub(1), 0.0, boundary)

# Compute solution
w = Function(W)
solve(a == L, w, bc)
(sigma, u) = w.split()

# Plot sigma and u
plot(sigma)
plot(u)
interactive()
