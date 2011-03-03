"""This demo illustrates how to set boundary conditions for meshes
that include boundary indicators. The mesh used in this demo was
generated with VMTK (http://villacamozzi.marionegri.it/~luca/vmtk/)."""

__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-05-23 -- 2008-12-13"
__copyright__ = "Copyright (C) 2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh and define function space
mesh = Mesh("aneurysm.xml.gz")
V = FunctionSpace(mesh, "CG", 1)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Constant(0.0)
a = dot(grad(v), grad(u))*dx
L = v*f*dx

# Define boundary condition values
u0 = Constant(0.0)
u1 = Constant(1.0)
u2 = Constant(2.0)
u3 = Constant(3.0)

# Define boundary conditions
bc0 = DirichletBC(V, u0, 0)
bc1 = DirichletBC(V, u1, 1)
bc2 = DirichletBC(V, u2, 2)
bc3 = DirichletBC(V, u3, 3)

# Compute solution
problem = VariationalProblem(a, L, [bc0, bc1, bc2, bc3])
u = problem.solve()

# Plot solution
plot(u, interactive=True)
