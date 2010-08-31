__author__ = "Anders Logg (logg@simula.no)"
__date__ = "2008-10-17 -- 2009-10-07"
__copyright__ = "Copyright (C) 2008 Anders Logg"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

# Create mesh
mesh = UnitSquare(16, 16)

# Define function spaces
P1 = FunctionSpace(mesh, "CG", 1)
P0 = FunctionSpace(mesh, "DG", 0)

# Define test and trial functions
v1 = TestFunction(P1)
w1 = TrialFunction(P1)
w0 = TestFunction(P0)

# Define functions
u  = Function(P1)
z  = Function(P1)
f  = Constant(1.0)
p  = Function(P0)
u0 = Expression("x[0]*(1.0 - x[0])*x[1]*(1.0 - x[1])")

# Dirichlet boundary
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Boundary condition
bc = DirichletBC(P1, u0, DirichletBoundary())

# Goal functional
J = (u - u0)*(u - u0)*dx

# Forward problem
problem = (inner(grad(v1), p*grad(w1))*dx, v1*f*dx)

# Adjoint problem
adjoint = (inner(grad(w1), p*grad(v1))*dx, -2*v1*(u - u0)*dx)

# Update of parameter
gradient = -inner(grad(z), w0*grad(u))*dx
px = p.vector()

# Set initial value for parameter
px[:] = 1.0

# Iterate until convergence
for i in range(100):

    # Solve forward problem
    A = assemble(problem[0])
    b = assemble(problem[1])
    bc.apply(A, b)
    solve(A, u.vector(), b)

    # Solve adjoint problem
    A = assemble(adjoint[0])
    b = assemble(adjoint[1])
    bc.apply(A, b)
    solve(A, z.vector(), b)

    # Update parameter
    dp = assemble(gradient)
    dp *= 100.0
    px -= dp

    # Print value of functional
    jval = assemble(J, mesh = mesh)
    print "J = ", jval
    print u.vector().max()

# Plot solution and parameter
plot(u,  title="Solution",  rescale=True)
plot(z,  title="Adjoint",   rescale=True)
plot(p,  title="Parameter", rescale=True)
plot(u0, mesh=mesh, title="Target", rescale=True)

# Hold plot
interactive()
