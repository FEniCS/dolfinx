# Copyright (C) 2016 Jorgen Dokken
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
# First added:  2015-08-11
# Last changed: 2015-08-11
#
# This demo program assembles a MultiMeshForm for MultiMeshFunctions

from dolfin import *

class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


def functional_poisson():
    "Compute solution for given mesh configuration"

    # Create meshes
    mesh_0 = RectangleMesh(Point(0, 0), Point(0.5, 1), 20, 20)
    mesh_1 = RectangleMesh(Point(0.5, 0), Point(1, 1), 21, 21)
    mesh_2 = RectangleMesh(Point(0.2, 0.2), Point(0.4, 0.4), 100, 100)
    # Build multimesh
    multimesh = MultiMesh()
    multimesh.add(mesh_0)
    multimesh.add(mesh_1)
    multimesh.add(mesh_2)
    multimesh.build()
    # Create finite element and functionspace
    element = FiniteElement("Lagrange", triangle, 1)

    V = MultiMeshFunctionSpace(multimesh, element)

    # Define trial and test functions and right-hand side
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6)

    # Define facet normal and mesh size
    n = FacetNormal(multimesh)
    h = 2.0*Circumradius(multimesh)
    h = (h('+') + h('-')) / 2

    # Set parameters
    alpha = 4.0
    beta = 4.0

    # Define bilinear form
    a = dot(grad(u), grad(v))*dX \
      - dot(avg(grad(u)), jump(v, n))*dI \
      - dot(avg(grad(v)), jump(u, n))*dI \
      + alpha/h*jump(u)*jump(v)*dI \
      + beta*dot(jump(grad(u)), jump(grad(v)))*dO

    # Define linear form
    L = f*v*dX

    # Assemble linear system
    A = assemble_multimesh(a)
    b = assemble_multimesh(L)

    # Apply boundary condition
    # zero = Constant(0)
    u0 = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]")
    boundary = DirichletBoundary()
    bc = MultiMeshDirichletBC(V, u0, boundary)
    bc.apply(A, b)

    # Compute solution
    u = MultiMeshFunction(V)
    solve(A, u.vector(), b)
    #plot(V.multimesh())
    plot(u.part(0), title="u_0")
    plot(u.part(1), title="u_1")
    plot(u.part(2), title="u_2")
    interactive()
    
    uflform = u*dX
    # from IPython import embed; embed()
    A = assemble_multimesh(uflform)
    print 'Area of MultiMesh: ', A

def org_prob():
    # Create mesh and define function space
    mesh = UnitSquareMesh(40, 40)
    V = FunctionSpace(mesh, 'Lagrange', 1)
    # Define boundary conditions
    u0 = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]')
    def u0_boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u0, u0_boundary)
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    a = inner(nabla_grad(u), nabla_grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution and mesh
    print assemble(u*dx)



    
if __name__ == '__main__':
    functional_poisson()
    org_prob()
