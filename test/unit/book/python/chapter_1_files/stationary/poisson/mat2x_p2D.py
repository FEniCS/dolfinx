"""
As mat2_p2D.py, but the boundary conditions are different.
Here, u=0 on x=0 and u=1 on x=1, while du/dn=0 on y=0 and y=1.
This yields a solution u(x,y)=x, regardless of the values of k.
"""

from dolfin import *
import sys, math, numpy

mesh = UnitSquare(4, 6)

# Define a MeshFunction over two subdomains
subdomains = MeshFunction('size_t', mesh, 2)

class Omega0(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] <= 0.5 else False

class Omega1(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] >= 0.5 else False

# Mark subdomains with numbers 0 and 1
subdomain0 = Omega0()
subdomain0.mark(subdomains, 0)
subdomain1 = Omega1()
subdomain1.mark(subdomains, 1)

V0 = FunctionSpace(mesh, 'DG', 0)
k = Function(V0)

print 'mesh:', mesh
print 'subdomains:', subdomains
print 'k:', k

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

print 'k degree of freedoms:', k.vector().array()

#plot(subdomains, title='subdomains')

V = FunctionSpace(mesh, 'Lagrange', 1)

# Define Dirichlet conditions for x=0 boundary

u_L = Constant(0)

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0]) < tol

Gamma_0 = DirichletBC(V, u_L, LeftBoundary())

# Define Dirichlet conditions for x=1 boundary

u_R = Constant(1)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14   # tolerance for coordinate comparisons
        return on_boundary and abs(x[0] - 1) < tol

Gamma_1 = DirichletBC(V, u_R, RightBoundary())

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

# Dump solution to the screen
u_nodal_values = u.vector()
u_array = u_nodal_values.array()
coor = mesh.coordinates()
for i in range(len(u_array)):
    print 'u(%8g,%8g) = %g' % (coor[i][0], coor[i][1], u_array[i])

#interactive()
