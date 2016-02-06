"""As membrane1.py, but with more Viper visualization."""

from __future__ import print_function
from dolfin import *

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
               '     - 0.5*(pow((R*x[1] - y0)/sigma, 2)))',
               R=R, x0=x0, y0=y0, sigma=sigma, degree=2)
L = f*v*dx

# Compute solution
w = Function(V)
problem = LinearVariationalProblem(a, L, w, bc)
solver  = LinearVariationalSolver(problem)
solver.parameters['linear_solver'] = 'cg'
solver.parameters['preconditioner'] = 'ilu'
solver.solve()

# Find maximum real deflection
max_w = w.vector().array().max()
max_D = A*max_w/(8*pi*sigma*T)
print('Maximum real deflection is', max_D)

# Demonstrate some visualization

# Cannot do plot(w) first and then grab viz object!
import time
viz_w = plot(w,
             wireframe=False,
             title='Scaled membrane deflection',
             rescale=False,
             axes=True,
             )

viz_w.elevate(-65) # tilt camera -65 degrees (latitude dir)
viz_w.set_min_max(0, 0.5*max_w)
viz_w.update(w)    # bring settings above into action
viz_w.write_png('membrane_deflection.png')
viz_w.write_ps('membrane_deflection', format='eps')

f = interpolate(f, V)
viz_f = plot(f, title='Scaled pressure')
viz_f.elevate(-65)
viz_f.update(f)
viz_f.write_png('pressure.png')
viz_f.write_ps('pressure', format='eps')

viz_m = plot(mesh, title='Finite element mesh')

#time.sleep(15)

# Should be at the end
interactive()
