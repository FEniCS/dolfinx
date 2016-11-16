# Copyright (C) 2015 Anders Logg
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
# First added:  2015-11-11
# Last changed: 2015-11-24
#
# This demo program solves the Stokes equations on a domain defined
# by three overlapping and non-matching meshes.

from dolfin import *

class InflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class OutflowBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)

class NoslipBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], 1.0))

if MPI.size(mpi_comm_world()) > 1:
    info("Sorry, this demo does not (yet) run in parallel.")
    exit(0)

# Create meshes
mesh_0 = UnitSquareMesh(16, 16)
mesh_1 = RectangleMesh(Point(0.2, 0.2), Point(0.6, 0.6), 8, 8)
mesh_2 = RectangleMesh(Point(0.4, 0.4), Point(0.8, 0.8), 8, 8)

# Build multimesh
multimesh = MultiMesh()
multimesh.add(mesh_0)
multimesh.add(mesh_1)
multimesh.add(mesh_2)
multimesh.build()

# FIXME: Tensor algebra not supported for multimesh function spaces

# Create function space
P2 = VectorElement("Lagrange", triangle, 2)
P1 = FiniteElement("Lagrange", triangle, 1)
TH = P2 * P1
W  = MultiMeshFunctionSpace(multimesh, TH)

# Define trial and test functions and right-hand side
u = TrialFunction(W)
v = TestFunction(W)
f = Constant((0, 0))

# Define trial and test functions and right-hand side
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Define facet normal and mesh size
n = FacetNormal(multimesh)
h = 2.0*Circumradius(multimesh)

# Parameters
alpha = 4.0

def tensor_jump(v, n):
    return outer(v('+'), n('+')) + outer(v('-'), n('-'))

def a_h(v, w):
    return inner(grad(v), grad(w))*dX \
         - inner(avg(grad(v)), tensor_jump(w, n))*dI \
         - inner(avg(grad(w)), tensor_jump(v, n))*dI \
         + alpha/avg(h) * inner(jump(v), jump(w))*dI

def b_h(v, q):
    return -div(v)*q*dX + jump(v, n)*avg(q)*dI

def l_h(v, q, f):
    return inner(f, v)*dX

def s_O(v, w):
    return inner(jump(grad(v)), jump(grad(w)))*dO

def s_C(v, q, w, r):
    return h*h*inner(-div(grad(v)) + grad(q), -div(grad(w)) - grad(r))*dC

def l_C(v, q, f):
    return h*h*inner(f, -div(grad(v)) - grad(q))*dC

# Define bilinear form
a = a_h(u, v) + b_h(v, p) + b_h(u, q) + s_O(u, v) + s_C(u, p, v, q)

# Define linear form
L  = l_h(v, q, f) + l_C(v, q, f)

# Assemble linear system
A = assemble_multimesh(a)
b = assemble_multimesh(L)

# Create boundary values
inflow_value = Expression(("sin(x[1]*DOLFIN_PI)", "0.0"), degree=2)
outflow_value = Constant(0)
noslip_value = Constant((0, 0))

# Create subdomains for boundary conditions
inflow_boundary = InflowBoundary()
outflow_boundary = OutflowBoundary()
noslip_boundary = NoslipBoundary()

# Create subspaces for boundary conditions
V = MultiMeshSubSpace(W, 0)
Q = MultiMeshSubSpace(W, 1)

# Create boundary conditions
bc0 = MultiMeshDirichletBC(V, noslip_value,  noslip_boundary)
bc1 = MultiMeshDirichletBC(V, inflow_value,  inflow_boundary)
bc2 = MultiMeshDirichletBC(Q, outflow_value, outflow_boundary)

# Apply boundary conditions
bc0.apply(A, b)
bc1.apply(A, b)
bc2.apply(A, b)

# Compute solution
w = MultiMeshFunction(W)
solve(A, w.vector(), b)

# FIXME: w.part(i).split() not working for extracted parts
# FIXME: since they are only dolfin::Functions

# Extract solution components
u0 = w.part(0).sub(0)
u1 = w.part(1).sub(1)
u2 = w.part(2).sub(0)
p0 = w.part(0).sub(1)
p1 = w.part(1).sub(0)
p2 = w.part(2).sub(1)

# Save to file
File("u0.pvd") << u0
File("u1.pvd") << u1
File("u2.pvd") << u2
File("p0.pvd") << p0
File("p1.pvd") << p1
File("p2.pvd") << p2

# Plot solution
plot(W.multimesh())
plot(u0, title="u_0")
plot(u1, title="u_1")
plot(u2, title="u_2")
plot(p0, title="p_0")
plot(p1, title="p_1")
plot(p2, title="p_2")
interactive()
