# This program illustrates the use of the DOLFIN nonlinear solver for solving 
# problems of the form F(u) = 0. The user must provide functions for the 
# function (Fu) and update of the (approximate) Jacobian.  
#
# This simple program solves a nonlinear variant of Poisson's equation
#
#     - div (1+u^2) grad u(x, y) = f(x, y)
#
# on the unit square with source f given by
#
#     f(x, y) = t * x * sin(y)
#
# and boundary conditions given by
#
#     u(x, y)     = t  for x = 0
#     du/dn(x, y) = 0  otherwise
#
# where t is pseudo time.
#
# This is equivalent to solving: 
# F(u) = (grad(v), (1-u^2)*grad(u)) - f(x,y) = 0
#
# Original implementation: ../cpp/main.cpp by Garth N. Wells.
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-15 -- 2007-12-07"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# FIXME: Not working, see notice below
import sys
print "This demo is not working, please fix me"
sys.exit(1)

# Create mesh and create finite element
mesh = UnitSquare(64, 64)
element = FiniteElement("Lagrange", "triangle", 1)

# Right-hand side
#class Source(Function, TimeDependent):
class Source(Function):
    def __init__(self, element, mesh, t):
        Function.__init__(self, element, mesh)
#        TimeDependent.__init__(self, t)
        self.t = t

    def time(self):
        return self.t

    def eval(self, values, x):
        values[0] = self.time()*x[0]*sin(x[1])

# Dirichlet boundary condition
#class DirichletBoundaryCondition(Function, TimeDependent):
class DirichletBoundaryCondition(Function):
    def __init__(self, element, mesh, t):
        Function.__init__(self, element, mesh)
#        TimeDependent.__init__(self, t)
        self.t = t

    def time(self):
        return self.t
  
    def eval(self, values, x):
        values[0] = self.time()*1.0

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary)

# User defined nonlinear problem 
class MyNonlinearProblem(NonlinearProblem):
    def __init__(self, element, mesh, bc, u0, f):
        NonlinearProblem.__init__(self)

        # Define variational problem
        v = TestFunction(element)
        u = TrialFunction(element)

        a = (1.0 + u0*u0)*dot(grad(v), grad(u))*dx + 2.0*u0*u*dot(grad(v), grad(u0))*dx
        L = v*f*dx - (1.0 + u0*u0)*dot(grad(v), grad(u0))*dx

        # Attach members
        self.a = a
        self.L = L
        self.mesh = mesh
        self.bc = bc
        self.compiled_form = jit(a)[0]
 
    # User defined assemble of Jacobian and residual vector 
    def form(self, A, b, x):
        dolfin_set("output destination", "silent")
        # Copy paste from assemble.py
        # Assemble A
        # Compile form
        (compiled_form, module, form_data) = jit(self.a)

        # Extract coefficients
        coefficients = ArrayFunctionPtr()
        for c in form_data[0].coefficients:
            coefficients.push_back(c.f)

        # Create dummy arguments (not yet supported)
        cell_domains = MeshFunction("uint")
        exterior_facet_domains = MeshFunction("uint")
        interior_facet_domains = MeshFunction("uint")

        # Assemble compiled form
        cpp_assemble(A, compiled_form, mesh, coefficients,
                     cell_domains, exterior_facet_domains, interior_facet_domains,
                     True)

#        A = assemble(self.a, self.mesh)
#        b = assemble(self.L, self.mesh)

        # Assemble b
        (compiled_form, module, form_data) = jit(self.L)

        # Extract coefficients
        coefficients = ArrayFunctionPtr()
        for c in form_data[0].coefficients:
            coefficients.push_back(c.f)

        # Create dummy arguments (not yet supported)
        cell_domains = MeshFunction("uint")
        exterior_facet_domains = MeshFunction("uint")
        interior_facet_domains = MeshFunction("uint")

        # Assemble compiled form
        cpp_assemble(b, compiled_form, mesh, coefficients,
                     cell_domains, exterior_facet_domains, interior_facet_domains,
                     True)

#        print "b: ", b.disp()
        self.bc.apply(A, b, x, self.compiled_form)
#        print "bc"
        dolfin_set("output destination", "terminal")

# Pseudo time
t = 0.0

# Create source function
f = Source(element, mesh, t)

# Dirichlet boundary conditions
dirichlet_boundary = DirichletBoundary()
g = DirichletBoundaryCondition(element, mesh, t)
bc = DirichletBC(g, mesh, dirichlet_boundary)

x = Vector()
v = TestFunction(element)
u = TrialFunction(element)

compiled_form = jit(v*u*dx)[0]
u0 = Function(element, mesh, x, compiled_form)

# Create user-defined nonlinear problem
nonlinear_problem = MyNonlinearProblem(element, mesh, bc, u0, f)

# Create nonlinear solver (using GMRES linear solver) and set parameters
# nonlinear_solver = NewtonSolver(gmres)
nonlinear_solver = NewtonSolver()
# nonlinear_solver.set("Newton maximum iterations", 50)
# nonlinear_solver.set("Newton relative tolerance", 1e-10)
# nonlinear_solver.set("Newton absolute tolerance", 1e-10)

# Solve nonlinear problem in a series of steps
dt = 1.0
T  = 3.0

while( f.t < T):
    f.t += dt
    g.t += dt
    nonlinear_solver.solve(nonlinear_problem, x)

# Plot solution
plot(u0)

# Save function to file
file = File("nonlinear_poisson.pvd")
file << u0



