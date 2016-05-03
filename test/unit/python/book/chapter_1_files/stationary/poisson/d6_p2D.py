"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
As d5_p2D.py, but with a more complicated solution, error computations
and convergence studies.
"""

from __future__ import print_function
from dolfin import *
import sys

def compute(nx, ny, degree):
    # Create mesh and define function space
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'Lagrange', degree)

    # Define boundary conditions

    def u0_boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, Constant(0.0), u0_boundary)

    # Exact solution
    omega = 1.0
    u_e = Expression('sin(omega*pi*x[0])*sin(omega*pi*x[1])',
                     omega=omega, degree=degree+1)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    #f = Function(V,
    #    '2*pow(pi,2)*pow(omega,2)*sin(omega*pi*x[0])*sin(omega*pi*x[1])',
    #    {'omega': omega})
    f = 2*pi**2*omega**2*u_e
    a = inner(nabla_grad(u), nabla_grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    problem = LinearVariationalProblem(a, L, u, bc)
    solver =  LinearVariationalSolver(problem)
    solver.solve()
    #plot(u, title='Numerical')

    # Compute gradient
    #gradu = project(grad(u), VectorFunctionSpace(mesh, 'Lagrange', degree))

    # Compute error norm

    # Function - Expression
    error = (u - u_e)**2*dx
    E1 = sqrt(assemble(error))

    # Explicit interpolation of u_e onto the same space as u:
    u_e_V = interpolate(u_e, V)
    error = (u - u_e_V)**2*dx
    E2 = sqrt(assemble(error))

    # Explicit interpolation of u_e to higher-order elements,
    # u will also be interpolated to the space Ve before integration
    Ve = FunctionSpace(mesh, 'Lagrange', 5)
    u_e_Ve = interpolate(u_e, Ve)
    error = (u - u_e_Ve)**2*dx
    E3 = sqrt(assemble(error))

    # errornorm interpolates u and u_e to a space with
    # given degree, and creates the error field by subtracting
    # the degrees of freedom, then the error field is integrated
    # TEMPORARY BUG - doesn't accept Expression for u_e
    #E4 = errornorm(u_e, u, normtype='l2', degree=3)
    # Manual implementation
    def errornorm(u_e, u, Ve):
        u_Ve = interpolate(u, Ve)
        u_e_Ve = interpolate(u_e, Ve)
        e_Ve = Function(Ve)
        # Subtract degrees of freedom for the error field
        e_Ve.vector()[:] = u_e_Ve.vector().array() - u_Ve.vector().array()
        # More efficient computation (avoids the rhs array result above)
        #e_Ve.assign(u_e_Ve)                      # e_Ve = u_e_Ve
        #e_Ve.vector().axpy(-1.0, u_Ve.vector())  # e_Ve += -1.0*u_Ve
        error = e_Ve**2*dx(Ve.mesh())
        return sqrt(assemble(error)), e_Ve
    E4, e_Ve = errornorm(u_e, u, Ve)

    # Infinity norm based on nodal values
    u_e_V = interpolate(u_e, V)
    E5 = abs(u_e_V.vector().array() - u.vector().array()).max()
    print('E2:', E2)
    print('E3:', E3)
    print('E4:', E4)
    print('E5:', E5)

    # H1 seminorm
    error = inner(grad(e_Ve), grad(e_Ve))*dx
    E6 = sqrt(assemble(error))

    # Collect error measures in a dictionary with self-explanatory keys
    errors = {'u - u_e': E1,
              'u - interpolate(u_e,V)': E2,
              'interpolate(u,Ve) - interpolate(u_e,Ve)': E3,
              'error field': E4,
              'infinity norm (of dofs)': E5,
              'grad(error field)': E6}

    return errors

# Perform experiments
degree = int(sys.argv[1])
h = []  # element sizes
E = []  # errors
# Changed this line so unit tests run faster
for nx in [4, 8, 16]:
#for nx in [4, 8, 16, 32, 64, 128, 264]:
    h.append(1.0/nx)
    E.append(compute(nx, nx, degree))  # list of dicts

# Convergence rates
from math import log as ln  # log is a dolfin name too
error_types = list(E[0].keys())
for error_type in sorted(error_types):
    print('\nError norm based on', error_type)
    for i in range(1, len(E)):
        Ei   = E[i][error_type]  # E is a list of dicts
        Eim1 = E[i-1][error_type]
        r = ln(Ei/Eim1)/ln(h[i]/h[i-1])
        print('h=%8.2E E=%8.2E r=%.2f' % (h[i], Ei, r))
