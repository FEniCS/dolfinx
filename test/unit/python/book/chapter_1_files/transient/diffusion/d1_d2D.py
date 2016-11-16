"""
FEniCS tutorial demo program: Diffusion equation with Dirichlet
conditions and a solution that will be exact at all nodes.
"""

from __future__ import print_function
from dolfin import *
import numpy

# Create mesh and define function space
nx = ny = 2
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
alpha = 3; beta = 1.2
u0 = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                alpha=alpha, beta=beta, t=0, degree=1)

class Boundary(SubDomain):  # define the Dirichlet boundary
    def inside(self, x, on_boundary):
        return on_boundary

boundary = Boundary()
bc = DirichletBC(V, u0, boundary)

# Initial condition
u_1 = interpolate(u0, V)
#u_1 = project(u0, V)  # will not result in exact solution!

dt = 0.3      # time step

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)
a = u*v*dx + dt*inner(nabla_grad(u), nabla_grad(v))*dx
L = (u_1 + dt*f)*v*dx

A = assemble(a)   # assemble only once, before the time stepping
b = None          # necessary for memory saving assemeble call

# Compute solution
u = Function(V)   # the unknown at a new time level
T = 1.9           # total simulation time
t = dt
while t <= T:
    print('time =', t)
    b = assemble(L, tensor=b)
    u0.t = t
    bc.apply(A, b)
    solve(A, u.vector(), b)

    # Verify
    u_e = interpolate(u0, V)
    maxdiff = numpy.abs(u_e.vector().array() - u.vector().array()).max()
    print('Max error, t=%.2f: %-10.3f' % (t, maxdiff))

    t += dt
    u_1.assign(u)
