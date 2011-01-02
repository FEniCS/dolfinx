
"""This demo program solves the equations of static linear elasticity
for a gear clamped at two of its ends and twisted 30 degrees."""

__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2010-09-05"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *

# Form compiler options
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

set_log_level(0)

# Load mesh and define function space
mesh = Mesh("gear.xml.gz")
mesh.order()
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

    def value_shape(self):
        return (3,)

# Sub domain for rotation at right end
def right(x, on_boundary):
    return x[0] > 0.9 and on_boundary

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Constant((0.0, 0.0, 0.0))

E  = 10.0
nu = 0.3

mu    = E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(v.cell().d)

a = inner(grad(v), sigma(u))*dx
L = inner(v, f)*dx

# Set up boundary condition at left end
c = Constant((0.0, 0.0, 0.0))
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
File("elasticity.pvd") << u

# Save colored mesh partitions in VTK format if running in parallel
if MPI.num_processes() > 1:
    File("partitions.pvd") << CellFunction("uint", mesh, MPI.process_number())

# Plot solution
#plot(u, mode="displacement", interactive=True)
