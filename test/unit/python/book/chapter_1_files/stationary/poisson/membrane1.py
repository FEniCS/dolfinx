"""
FEniCS program for the deflection w(x,y) of a membrane:
-Laplace(w) = p = Gaussian function, in a unit circle,
with w = 0 on the boundary.
"""

from __future__ import print_function
from dolfin import *
import numpy

# This demo needs to be updated for the removal
# of CircleMesh from DOLFIN
exit(0)

# Set pressure function:
T = 10.0  # tension
A = 1.0   # pressure amplitude
R = 0.3   # radius of domain
theta = 0.2
x0 = 0.6*R*cos(theta)
y0 = 0.6*R*sin(theta)
sigma = 0.025
#sigma = 50  # large value for verification
n = 40   # approx no of elements in radial direction
mesh = CircleMesh(Point(0.0, 0.0), 1.0, 1.0/n)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary condition w=0

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.0), boundary)

# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
a = inner(nabla_grad(w), nabla_grad(v))*dx
f = Expression('4*exp(-0.5*(pow((R*x[0] - x0)/sigma, 2)) '
                     '-0.5*(pow((R*x[1] - y0)/sigma, 2)))',
               R=R, x0=x0, y0=y0, sigma=sigma, degree=2)
L = f*v*dx

# Compute solution
w = Function(V)
problem = LinearVariationalProblem(a, L, w, bc)
solver  = LinearVariationalSolver(problem)
solver.parameters['linear_solver'] = 'cg'
solver.parameters['preconditioner'] = 'ilu'
solver.solve()

# Plot scaled solution, mesh and pressure
#plot(mesh, title='Mesh over scaled domain')
#plot(w, title='Scaled deflection')
f = interpolate(f, V)
#plot(f, title='Scaled pressure')

# Find maximum real deflection
max_w = w.vector().array().max()
max_D = A*max_w/(8*pi*sigma*T)
print('Maximum real deflection is', max_D)

# Verification for "flat" pressure (large sigma)
if sigma >= 50:
    w_exact = Expression('1 - x[0]*x[0] - x[1]*x[1]', degree=2)
    w_e = interpolate(w_exact, V)
    w_e_array = w_e.vector().array()
    w_array = w.vector().array()
    diff_array = numpy.abs(w_e_array - w_array)
    print('Verification of the solution, max difference is %.4E' % \
          diff_array.max())

    # Create finite element field over V and fill with error values
    difference = Function(V)
    difference.vector()[:] = diff_array
    #plot(difference, title='Error field for sigma=%g' % sigma)

# Should be at the end
interactive()
