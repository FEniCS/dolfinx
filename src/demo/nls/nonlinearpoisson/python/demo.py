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
__date__ = "2007-11-15 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

#
# THIS DEMO IS CURRENTLY NOT WORKING
#
# ERROR:
# Starting Newton solve.
# Traceback (most recent call last):
#   File "demo.py", line 135, in <module>
#     nonlinear_solver.solve(nonlinear_problem, x)
#   File "/home/oelgaard/fenics/dolfin/local/lib/python2.5/site-packages/dolfin/dolfin.py", line 6107, in solve
#     return _dolfin.NewtonSolver_solve(*args)
# RuntimeError: *** Error: Nonlinear problem update for F(u) and J  has not been supplied by user.

# Create mesh and create finite element
mesh = UnitSquare(64, 64)
element = FiniteElement("Lagrange", "triangle", 1)

# Right-hand side
class Source(Function, TimeDependent):
    def __init__(self, element, mesh, t):
        Function.__init__(self, element, mesh)
        TimeDependent.__init__(self, t)

    def eval(self, values, x):
        values[0] = time()*x[0]*sin(x[1])

# Dirichlet boundary condition
class DirichletBoundaryCondition(Function, TimeDependent):
    def __init__(self, element, mesh, t):
        Function.__init__(self, element, mesh)
        TimeDependent.__init__(self, t)
  
    def eval(self, values, x):
        values[0] = 1.0*time()

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary)

# User defined nonlinear problem 
class MyNonlinearProblem(NonlinearProblem):
    def __init__(self, element, mesh, x, dirichlet_boundary, g, f, u0):
        NonlinearProblem.__init__(self)

        # Define variational problem
        v = TestFunction(element)
        u = TrialFunction(element)

        a = (1.0 + u0*u0)*dot(grad(v), grad(u))*dx + 2.0*u0*u*dot(grad(v), grad(u0))*dx
        L = v*f*dx - (1.0 + u0*u0)*dot(grad(v), grad(u0))*dx

        # Define Dirichlet boundary conditions
        bc = DirichletBC(g, mesh, dirichlet_boundary)

        # Attach members
        self.a = a
        self.L = L
        self.compiled_form = jit(a)[0]
        self.mesh = mesh
        self.bc = bc

#       // Initialise solution vector u
#        u0.init(element, mesh, x)
 
    # User defined assemble of Jacobian and residual vector 
    def form(self, A, b, x):
        print "form"
        set("output destination", "silent")
        A = assemble(self.a, self.mesh)
        b = assemble(self.L, self.mesh)
        self.bc.apply(A, b, x, self.compiled_form)
        set("output destination", "terminal")

    def F(b, x):
        print x
    def J(A, x):
        print x

# Pseudo time
t = 0.0

# Create source function
f = Source(element, mesh, t)

# Dirichlet boundary conditions
dirichlet_boundary = DirichletBoundary()
g = DirichletBoundaryCondition(element, mesh, t)

x = Vector()
v = TestFunction(element)
u = TrialFunction(element)
u0 = Function(element, mesh)

a = (1.0 + u0*u0)*dot(grad(v), grad(u))*dx + 2.0*u0*u*dot(grad(v), grad(u0))*dx
compiled_form = jit(a)[0]

u1 = Function(element, mesh, x, compiled_form)

# # Create user-defined nonlinear problem
nonlinear_problem = MyNonlinearProblem(element, mesh, u1.vector(), dirichlet_boundary, g, f, u1)

# Create nonlinear solver (using GMRES linear solver) and set parameters
# nonlinear_solver = NewtonSolver(gmres)
nonlinear_solver = NewtonSolver()

# Solve nonlinear problem in a series of steps
dt = 1.0
T  = 3.0

while( t < T):
    t += dt
    nonlinear_solver.solve(nonlinear_problem, x)

# Plot solution
plot(u)

# Save function to file
file = File("nonlinear_poisson.pvd")
file << u



