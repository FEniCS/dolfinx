""" This demo implements a Poisson equations solver
based on the demo "dolfin/demo/pde/poisson/python/demo.py"
in Dolfin using Epetra matrices, the AztecOO CG solver and ML 
AMG preconditioner 
"""

__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2008-04-24"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"



from dolfin import *

dolfin_set("linear algebra backend", "Epetra")

# Create mesh and finite element
mesh = UnitSquare(20,20)
V = FunctionSpace(mesh, "Lagrange", 1)


# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and x[0] < DOLFIN_EPS)

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Function(V,"500.0 * exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")

a = u*v*dx + dot(grad(v), grad(u))*dx
L = v*f*dx 

# Define boundary condition
u0 = Constant(mesh, 0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Create linear system
A = assemble(a)
b = assemble(L)
A, b = assemble_system(a, L, bc, mesh) 

# Create solution vector (also used as start vector) 
U = Function(V)

solve(A, U.vector(), b, cg, ilu)

# plot the solution
plot(U)
interactive()

# Save solution to file
file = File("poisson.pvd")
file << U



