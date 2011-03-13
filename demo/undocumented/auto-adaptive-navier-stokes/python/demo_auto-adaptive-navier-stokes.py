__author__ = """Marie E. Rognes."""
__copyright__ = "Copyright (C) 2010 Marie Rognes"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *
import time

class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS) or \
               (on_boundary and abs(x[0] - 1.5) < 0.1 + DOLFIN_EPS)

class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > 4.0 - DOLFIN_EPS

parameters["allow_extrapolation"] = True;

# Material parameters
nu = Constant(0.02)

# Mesh
mesh = Mesh("channel_with_flap.xml")

# Define function spaces (Taylor-Hood)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# Define unknown and test function(s)
(v, q) = TestFunctions(W)
w_h = Function(W)
(u_h, p_h) = (as_vector((w_h[0], w_h[1])), w_h[2])

# Prescribed pressure
p0 = Expression("(4.0 - x[0])/4.0")

# Define variational forms
n = FacetNormal(mesh)
a = (nu*inner(grad(u_h), grad(v)) - div(v)*p_h + q*div(u_h))*dx
a = a + inner(grad(u_h)*u_h, v)*dx
L = - p0*dot(v, n)*ds
F = a - L

dw = TrialFunction(W)
dF = derivative(F, w_h, dw) # FIXME

# Define boundary conditions
bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), Noslip())

# Define variational problem (with new notation)
pde = VariationalProblem(F, dF, bc)

# Define goal and reference
M = u_h[0]*ds(0)
outflow = Outflow()
outflow_markers = MeshFunction("uint", mesh, mesh.topology().dim() - 1)
outflow_markers.set_all(1)
outflow.mark(outflow_markers, 0)
M.exterior_facet_domains = outflow_markers;

pde.parameters["adaptivity"]["reference"] = 0.40863917;
pde.parameters["adaptivity"]["plot_mesh"] = False;

# Solve to given tolerance
tol = 1.e-05
pde.solve(w_h, tol, M)
