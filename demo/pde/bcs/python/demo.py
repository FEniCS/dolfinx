"""This demo illustrates how to set boundary conditions for meshes
that include boundary indicators. The mesh used in this demo was
generated with VMTK (http://villacamozzi.marionegri.it/~luca/vmtk/)."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-05-23 -- 2008-05-23"
__copyright__ = "Copyright (C) 2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and finite element
mesh = Mesh("../../../../data/meshes/aneurysm.xml.gz")
element = FiniteElement("Lagrange", "tetrahedron", 1)

# Define variational problem
v = TestFunction(element)
u = TrialFunction(element)
f = Function(element, mesh, 0.0)
a = dot(grad(v), grad(u))*dx
L = v*f*dx

# Define boundary condition values
u0 = Function(mesh, 0.0)
u1 = Function(mesh, 1.0)
u2 = Function(mesh, 2.0)
u3 = Function(mesh, 3.0)

# Define boundary conditions
bc0 = DirichletBC(u0, mesh, 0)
bc1 = DirichletBC(u1, mesh, 1)
bc2 = DirichletBC(u2, mesh, 2)
bc3 = DirichletBC(u3, mesh, 3)

# Solve PDE and plot solution
pde = LinearPDE(a, L, mesh, [bc0, bc1, bc2, bc3])
u = pde.solve()
plot(u, interactive=True)
