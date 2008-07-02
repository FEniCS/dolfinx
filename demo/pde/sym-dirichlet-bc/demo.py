
from dolfin import *

#dolfin_set("linear algebra backend", "uBlas")
dolfin_set("linear algebra backend", "PETSc")

# Create mesh and finite element
mesh = UnitSquare(50,50)
element = FiniteElement("Lagrange", "triangle", 3)

# Source term
class Solution(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        values[0] = sin(10*x[0]) 

# Source term
class Source(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        values[0] = 100*sin(10*x[0]) 


# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary) 

# Define variational problem
v = TestFunction(element)
u = TrialFunction(element)
f = Source(element, mesh)

a = dot(grad(v), grad(u))*dx
L = v*f*dx 

# Define boundary condition
u0 = Solution(element, mesh) 
boundary = DirichletBoundary()
bc = DirichletBC(u0, mesh, boundary)

A, b = assemble_system(a, L, bc, mesh)

file = File("A.m")
file << A

file = File("b.m")
file << b

x = b.copy()
x.zero()
solve(A, x, b)

# plot the solution
U = Function(element, mesh, x)
plot(U)
interactive()


# Save solution to file
#file = File("poisson.pvd")
#file << u

# Hold plot
#interactive()
