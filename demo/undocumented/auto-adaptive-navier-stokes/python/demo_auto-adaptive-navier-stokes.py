# Copyright (C) 2010 Marie E. Rognes
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
# First added:  2010
# Last changed: 2011-10-04

from dolfin import *

class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS) or \
               (on_boundary and abs(x[0] - 1.5) < 0.1 + DOLFIN_EPS)

class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 4.0 - DOLFIN_EPS

# Use compiler optimizations
parameters["form_compiler"]["cpp_optimize"] = True

# Allow approximating values for points that may be generated outside
# of domain (because of numerical inaccuracies)
parameters["allow_extrapolation"] = True
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

# Material parameters
nu = Constant(0.02)

# Mesh
mesh = Mesh("../channel_with_flap.xml.gz")

# Define function spaces (Taylor-Hood)
V = VectorElement("CG", mesh.ufl_cell(), 2)
Q = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, V * Q)

# Define unknown and test function(s)
(v, q) = TestFunctions(W)
w = Function(W)
(u, p) = (as_vector((w[0], w[1])), w[2])

# Prescribed pressure
p0 = Expression("(4.0 - x[0])/4.0", degree=2)

# Define variational forms
n = FacetNormal(mesh)
a = (nu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx()
a = a + inner(grad(u)*u, v)*dx()
L = - p0*dot(v, n)*ds()
F = a - L

# Define boundary conditions
bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), Noslip())

# Create boundary subdomains
outflow = Outflow()
outflow_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
outflow_markers.set_all(1)
outflow.mark(outflow_markers, 0)

# Define new measure with associated subdomains
ds = Measure('ds', domain=mesh, subdomain_data=outflow_markers)

# Define goal
M = u[0]*ds(0)

# Define error tolerance (with respect to goal)
tol = 1.e-05

# If no more control is wanted, do:
# solve(F == 0, w, bc, tol=tol, M=M)

# Compute Jacobian form
J = derivative(F, w)

# Define variational problem
pde = NonlinearVariationalProblem(F, w, bc, J)

# Define solver
solver = AdaptiveNonlinearVariationalSolver(pde, M)

# Set reference value
solver.parameters["reference"] = 0.40863917;

# Solve to given tolerance
solver.solve(tol)

# Show solver summary
solver.summary();

# Show all timings
list_timings(TimingClear_clear, [TimingType_wall])

# Extract solutions on coarsest and finest mesh:
(u0, p0) = w.root_node().split()
(u1, p1) = w.leaf_node().split()
plot(p0, title="Pressure on initial mesh")
plot(p1, title="Pressure on final mesh")
interactive()
