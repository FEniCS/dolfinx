
"""This demo program solves the equations of static linear elasticity
for a gear clamped at two of its ends and twisted 30 degrees."""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2009-10-07"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

# Load mesh and define function space
mesh = Mesh("../../../../data/meshes/gear.xml.gz")
V = VectorFunctionSpace(mesh, "CG", 1)

# Sub domain for clamp at left end
def left(x, on_boundary):
    return x[0] < 0.5 and on_boundary

# Dirichlet boundary condition for rotation at right end
class Rotation(Expression):

    def eval(self, values, x):

        # Center of rotation
        y0 = 0.5
        z0 = 0.219

        # Angle of rotation (30 degrees)
        theta = 0.5236

        # New coordinates
        y = y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta)
        z = z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta)

        # Clamp at right end
        values[0] = 0.0
        values[1] = y - x[1]
        values[2] = z - x[2]

    def dim(self):
        return 3

# Sub domain for rotation at right end
def right(x, on_boundary):
    return x[0] > 0.9 and on_boundary

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Constant((0, 0, 0))

E  = 10.0
nu = 0.3

mu    = E / (2*(1 + nu))
lmbda = E*nu / ((1 + nu)*(1 - 2*nu))

def epsilon(v):
    return 0.5*(grad(v) + grad(v).T)

def sigma(v):
    return 2.0*mu*epsilon(v) + lmbda*tr(epsilon(v))*Identity(v.cell().d)

a = inner(epsilon(v), sigma(u))*dx
L = inner(v, f)*dx

# Set up boundary condition at left end
c = Constant((0, 0, 0))
bcl = DirichletBC(V, c, left)

# Set up boundary condition at right end
r = Rotation()
bcr = DirichletBC(V, r, right)

# Set up boundary conditions
bcs = [bcl, bcr]

# Set up PDE and solve
problem = VariationalProblem(a, L, bcs)
problem.parameters["symmetric"] = True
u = problem.solve()

# Save solution to VTK format
vtk_file = File("elasticity.pvd")
vtk_file << u

# Plot solution
#plot(u, mode="displacement", interactive=True)
