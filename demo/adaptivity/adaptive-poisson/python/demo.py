"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = exp(-100(x^2 + y^2))

and homogeneous Dirichlet boundary conditions.

Note that we use a simplified error indicator, ignoring
edge (jump) terms and the size of the interpolation constant.
"""

__author__ = "Rolv Erlend Bredesen <rolv@simula.no>"
__date__ = "2008-04-03 -- 2009-10-08"
__copyright__ = "Copyright (C) 2008 Rolv Erlend Bredesen"
__license__  = "GNU LGPL Version 2.1"

# Modified by Anders Logg, 2008.

from dolfin import *
from numpy import array, sqrt
from math import pow

# This demo does not run in parallel
not_working_in_parallel("This demo")

# This demo does not work without CGAL
if not has_cgal():
    print "DOLFIN must be compiled with CGAL to run this demo."
    exit(0)

TOL = 5e-4           # Error tolerance
REFINE_RATIO = 0.50  # Refine 50 % of the cells in each iteration
MAX_ITER = 20        # Maximal number of iterations

# Create initial mesh
mesh = UnitSquare(4, 4)
source_str = "exp(-100.0*(pow(x[0], 2) + pow(x[1], 2)))"
source = eval("lambda x: " + source_str)
# Adaptive algorithm
for level in xrange(MAX_ITER):

    # Define variational problem
    V = FunctionSpace(mesh, "CG", 1)
    v = TestFunction(V)
    u = TrialFunction(V)
    f = Expression(source_str)
    a = dot(grad(v), grad(u))*dx
    L = v*f*dx

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, DomainBoundary())

    # Compute solution
    problem = VariationalProblem(a, L, bc)
    u = problem.solve()

    # Compute error indicators
    h = array([c.diameter() for c in cells(mesh)])
    K = array([c.volume() for c in cells(mesh)])
    R = array([abs(source([c.midpoint().x(), c.midpoint().y()])) for c in cells(mesh)])
    gamma = h*R*sqrt(K)

    # Compute error estimate
    E  = sqrt(sum([g*g for g in gamma]))
    print "Level %d: E = %g (TOL = %g)" % (level, E, TOL)

    # Check convergence
    if E < TOL:
        print "Success, solution converged after %d iterations" % level
        break

    # Mark cells for refinement
    cell_markers = MeshFunction("bool", mesh, mesh.topology().dim())
    gamma_0 = sorted(gamma, reverse=True)[int(len(gamma)*REFINE_RATIO)]
    for c in cells(mesh):
        cell_markers[c] = gamma[c.index()] > gamma_0

    # Refine mesh
    mesh.refine(cell_markers)

    # Plot mesh
    plot(mesh)

# Hold plot
interactive()
