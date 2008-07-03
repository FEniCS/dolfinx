# This demo program solves the equations of static
# linear elasticity for a gear clamped at two of its
# ends and twisted 30 degrees.
#
# Original implementation: ../cpp/main.cpp by Johan Jansson and Anders Logg
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2007-12-03"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Load mesh and create finite element
mesh = Mesh("../../../../data/meshes/gear.xml.gz")
element = VectorElement("Lagrange", "tetrahedron", 1)

# Dirichlet boundary condition for clamp at left end
class Clamp(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)

    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0

    def rank(self):
        return 1

    def dim(self, i):
        return 3

# Sub domain for clamp at left end
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < 0.5 and on_boundary)

# Dirichlet boundary condition for rotation at right end
class Rotation(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)

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

    def rank(self):
        return 1

    def dim(self, i):
        return 3

# Sub domain for rotation at right end
class Right(SubDomain):
    def inside(self, x, on_boundary):
      return bool(x[0] > 0.9 and on_boundary)

# Initialise source function
f = Function(element, mesh, (0.0, 0.0, 0.0))

# Define variational problem
# Test and trial functions
v = TestFunction(element)
u = TrialFunction(element)

E  = 10.0
nu = 0.3

mu    = E / (2*(1 + nu))
lmbda = E*nu / ((1 + nu)*(1 - 2*nu))

def epsilon(v):
    return 0.5*(grad(v) + transp(grad(v)))

def sigma(v):
    return 2.0*mu*epsilon(v) + lmbda*mult(trace(epsilon(v)), Identity(len(v)))

a = dot(grad(v), sigma(u))*dx
L = dot(v, f)*dx

# Set up boundary condition at left end
c = Clamp(element, mesh)
left = Left()
bcl = DirichletBC(c, mesh, left)

# Set up boundary condition at right end
r = Rotation(element, mesh)
right = Right()
bcr = DirichletBC(r, mesh, right)

# Set up boundary conditions
bcs = [bcl, bcr]

# Set up PDE and solve
pde = LinearPDE(a, L, mesh, bcs)
sol = pde.solve()

# Save solution to VTK format
vtk_file = File("elasticity.pvd")
vtk_file << sol

# Save solution to XML format
xml_file = File("elasticity.xml")
xml_file << sol


# Plot solution
plot(sol, mode="displacement")
interactive()
