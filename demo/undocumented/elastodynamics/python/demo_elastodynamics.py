"""This demo program solves an elastodynamics problem."""

# Copyright (C) 2010 Garth N. Wells
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
# Modified by Anders Logg 2008-2011
#
# First added:  2010-04-30
# Last changed: 2012-11-12

from __future__ import print_function
from dolfin import *

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

def update(u, u0, v0, a0, beta, gamma, dt):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u.vector(), u0.vector()
    v0_vec, a0_vec = v0.vector(), a0.vector()

    # Update acceleration and velocity

    # a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
    a_vec = (1.0/(2.0*beta))*( (u_vec - u0_vec - v0_vec*dt)/(0.5*dt*dt) - (1.0-2.0*beta)*a0_vec )

    # v = dt * ((1-gamma)*a0 + gamma*a) + v0
    v_vec = dt*((1.0-gamma)*a0_vec + gamma*a_vec) + v0_vec

    # Update (u0 <- u0)
    v0.vector()[:], a0.vector()[:] = v_vec, a_vec
    u0.vector()[:] = u.vector()

# External load
class Traction(Expression):

    def __init__(self, dt, t, old, **kwargs):
        self.t   = t
        self.dt  = dt
        self.old = old

    def eval(self, values, x):

        # 'Shift' time for n-1 values
        t_tmp = self.t
        if self.old and t > 0.0:
            t_tmp -= self.dt

        cutoff_t = 10.0*1.0/32.0;
        weight = t_tmp/cutoff_t if t_tmp < cutoff_t else 1.0

        values[0] = 1.0*weight
        values[1] = 0.0

    def value_shape(self):
        return (2,)

# Sub domain for clamp at left end
def left(x, on_boundary):
    return x[0] < 0.001 and on_boundary

# Sub domain for rotation at right end
def right(x, on_boundary):
    return x[0] > 0.99 and on_boundary

# Load mesh and define function space
mesh = Mesh("../dolfin_fine.xml.gz")

# Define function space
V = VectorFunctionSpace(mesh, "CG", 1)

# Test and trial functions
u = TrialFunction(V)
r = TestFunction(V)

E  = 1.0
nu = 0.0
mu    = E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

# Mass density andviscous damping coefficient
rho = 1.0
eta = 0.25

# Time stepping parameters
alpha_m = 0.2
alpha_f = 0.4
beta    = 0.36
gamma   = 0.7
dt      = 1.0/32.0
t       = 0.0
T       = 10*dt

# Some useful factors
factor_m1  = rho*(1.0-alpha_m)/(beta*dt*dt)
factor_m2  = rho*(1.0-alpha_m)/(beta*dt)
factor_m3  = rho*(1.0-alpha_m-2.0*beta)/(2.0*beta)

factor_d1  = eta*(1.0-alpha_f)*gamma/(beta*dt)
factor_d2  = eta*((1.0-alpha_f)*gamma-beta)/beta
factor_d3  = eta*(gamma-2.0*beta)*(1.0-alpha_f)*dt/(2.0*beta)

# Fields from previous time step (displacement, velocity, acceleration)
u0 = Function(V)
v0 = Function(V)
a0 = Function(V)

# External forces (body and applied tractions
f  = Constant((0.0, 0.0))
p  = Traction(dt, t, False, degree=1)
p0 = Traction(dt, t, True, degree=1)

# Create mesh function over the cell facets
boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_subdomains.set_all(0)
force_boundary = AutoSubDomain(right)
force_boundary.mark(boundary_subdomains, 3)

# Define measure for boundary condition integral
dss = ds(subdomain_data=boundary_subdomains)

# Stress tensor
def sigma(r):
    return 2.0*mu*sym(grad(r)) + lmbda*tr(sym(grad(r)))*Identity(len(r))

# Forms
a = factor_m1*inner(u, r)*dx + factor_d1*inner(u, r)*dx \
   +(1.0-alpha_f)*inner(sigma(u), grad(r))*dx

L =  factor_m1*inner(r, u0)*dx + factor_m2*inner(r, v0)*dx \
   + factor_m3*inner(r, a0)*dx \
   + factor_d1*inner(r, u0)*dx + factor_d2*inner(r, v0)*dx \
   + factor_d3*inner(r, a0)*dx \
   - alpha_f*inner(grad(r), sigma(u0))*dx \
   + inner(r, f)*dx + (1.0-alpha_f)*inner(r, p)*dss(3) + alpha_f*inner(r, p0)*dss(3)

# Set up boundary condition at left end
zero = Constant((0.0, 0.0))
bc = DirichletBC(V, zero, left)

# FIXME: This demo needs some improved commenting

# Time-stepping
u = Function(V)
vtk_file = File("elasticity.pvd")
while t <= T:

    t += dt
    print("Time: ", t)

    p.t = t
    p0.t = t

    solve(a == L, u, bc)
    update(u, u0, v0, a0, beta, gamma, dt)

    # Save solution to VTK format
    vtk_file << u

# Plot solution
plot(u, mode="displacement", interactive=True)
