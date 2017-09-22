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
# First added:  2015-11-05
# Last changed: 2015-11-17
#
# This demo program solves Poisson's equation on a domain defined by
# three overlapping and non-matching meshes. The solution is computed
# on a sequence of rotating meshes to test the multimesh
# functionality.

from dolfin import *

if has_pybind11():
    print("Not supported in pybind11")
    exit()

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

def solve_poisson(t, x1, y1, x2, y2):
    "Compute solution for given mesh configuration"

    # Create meshes
    r = 0.5
    mesh_0 = RectangleMesh(Point(-r, -r), Point(r, r), 16, 16)
    mesh_1 = RectangleMesh(Point(x1 - r, y1 - r), Point(x1 + r, y1 + r), 8, 8)
    mesh_2 = RectangleMesh(Point(x2 - r, y2 - r), Point(x2 + r, y2 + r), 8, 8)
    mesh_1.rotate(70*t)
    mesh_2.rotate(-70*t)

    # Build multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.add(mesh_2)
    multimesh.build()

    # Create function space
    V = MultiMeshFunctionSpace(multimesh, "Lagrange", 1)

    # Define trial and test functions and right-hand side
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1)

    # Define facet normal and mesh size
    n = FacetNormal(multimesh)
    h = 2.0*Circumradius(multimesh)
    h = (h('+') + h('-')) / 2

    # Set parameters
    alpha = 10.0
    beta = 1.0

    # Define bilinear form
    a = dot(grad(u), grad(v))*dX \
      - dot(avg(grad(u)), jump(v, n))*dI \
      - dot(avg(grad(v)), jump(u, n))*dI \
      + alpha/h * jump(u)*jump(v)*dI \
      + beta/h**2 * dot(jump(u), jump(v))*dO

    # Define linear form
    L = f*v*dX

    # Assemble linear system
    A = assemble_multimesh(a)
    b = assemble_multimesh(L)

    # Apply boundary condition
    zero = Constant(0)
    boundary = DirichletBoundary()
    bc = MultiMeshDirichletBC(V, zero, boundary)
    bc.apply(A, b)

    # Remove inactive dofs
    V.lock_inactive_dofs(A, b)

    # Compute solution
    u = MultiMeshFunction(V)
    solve(A, u.vector(), b)

    return u

if MPI.size(mpi_comm_world()) > 1:
    info("Sorry, this demo does not (yet) run in parallel.")
    exit(0)

# Parameters
T = 10.0
N = 100
dt = T / N

# Create files for output
f0 = XDMFFile("output/u0.xdmf")
f1 = XDMFFile("output/u1.xdmf")
f2 = XDMFFile("output/u2.xdmf")

# Iterate over configurations
for n in range(N):
    info("Computing solution, step %d / %d." % (n + 1, N))

    # Compute coordinates for meshes
    t = dt*n
    x1 = sin(t)*cos(2*t)
    y1 = cos(t)*cos(2*t)
    x2 = cos(t)*cos(2*t)
    y2 = sin(t)*cos(2*t)

    # Compute solution
    u = solve_poisson(t, x1, y1, x2, y2)

    # Save to file
    f0.write(u.part(0), t)
    f1.write(u.part(1), t)
    f2.write(u.part(2), t)

# Close files
f0.close()
f1.close()
f2.close()
