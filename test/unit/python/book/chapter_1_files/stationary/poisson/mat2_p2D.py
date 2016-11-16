"""
FEniCS tutorial demo: Poisson problem in 2D with 2 materials.
"""

from __future__ import print_function
from dolfin import *
import sys, math, numpy

nx = 4;  ny = 6
mesh = UnitSquareMesh(nx, ny)

# Define a MeshFunction over two subdomains
subdomains = MeshFunction('size_t', mesh, 2)

class Omega0(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] <= 0.5 else False

class Omega1(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] >= 0.5 else False
# note: it is essential to use <= and >= in the comparisons

# Mark subdomains with numbers 0 and 1
subdomain0 = Omega0()
subdomain0.mark(subdomains, 0)
subdomain1 = Omega1()
subdomain1.mark(subdomains, 1)

V0 = FunctionSpace(mesh, 'DG', 0)
k = Function(V0)

print('mesh:', mesh)
print('subdomains:', subdomains)
print('k:', k)

# Loop over all cell numbers, find corresponding
# subdomain number and fill cell value in k
k_values = [1.5, 50]  # values of k in the two subdomains
for cell_no in range(len(subdomains.array())):
    subdomain_no = subdomains.array()[cell_no]
    k.vector()[cell_no] = k_values[subdomain_no]

# Much more efficient vectorized code
# (subdomains.array() has elements of type uint32, which
# must be transformed to plain int for numpy.choose to work)
help = numpy.asarray(subdomains.array(), dtype=numpy.int32)
k.vector()[:] = numpy.choose(help, k_values)

print('k degree of freedoms:', k.vector().array())

#plot(subdomains, title='subdomains')

V = FunctionSpace(mesh, 'Lagrange', 1)

# Define Dirichlet conditions for y=0 boundary

tol = 1E-14   # tolerance for coordinate comparisons
class BottomBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < tol

Gamma_0 = DirichletBC(V, Constant(0), BottomBoundary())

# Define Dirichlet conditions for y=1 boundary

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1) < tol

Gamma_1 = DirichletBC(V, Constant(1), TopBoundary())

bcs = [Gamma_0, Gamma_1]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
a = k*inner(nabla_grad(u), nabla_grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Compare numerical and exact solution
u_e_expr = Expression('x[1] <= 0.5 ? 2*x[1]*k1/(k0+k1) : '\
                      '((2*x[1]-1)*k0 + k1)/(k0+k1)',
                      k0=k_values[0], k1=k_values[1], degree=3)
u_e = interpolate(u_e_expr, V)
u_e_array = u_e.vector().array()
u_nodal_values = u.vector()
u_array = u_nodal_values.array()
import numpy
error = numpy.abs(u_e_array - u_array)
print('max error:', error.max())

#coor = mesh.coordinates()
#for i in range(len(u_array)):
#    print 'u(%8g,%8g) = %g, error: %.4E' % \
#        (coor[i][0], coor[i][1], u_array[i],
#         u_exact(coor[i][1]) - u_array[i])

#interactive()
