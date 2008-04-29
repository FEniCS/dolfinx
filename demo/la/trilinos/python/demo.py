""" This demo implements a Poisson equations solver
based on the demo "dolfin/demo/pde/poisson/python/demo.py"
in Dolfin using Epetra matrices, the AztecOO CG solver and ML 
AMG preconditioner 
"""

__author__ = "Kent-Andre Mardal (kent-and@simula.no)"
__date__ = "2008-04-24"
__copyright__ = "Copyright (C) 2008 Kent-Andre Mardal"



# Test for Trilinos:
try:
    from PyTrilinos import Epetra, AztecOO, TriUtils, ML 
except:
    print "You Need to have PyTrilinos with Epetra, AztecOO, TriUtils and ML installed for this demo to run",
    print "Exiting."
    exit()

from dolfin import *
try:
    dolfin.EpetraMatrix
except:
    print "PyDOLFIN has not been configured with Trilinos. Exiting."
    exit()


# Create mesh and finite element
mesh = UnitSquare(20,20)
element = FiniteElement("Lagrange", "triangle", 1)

# Source term
class Source(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] = 500.0*exp(-(dx*dx + dy*dy)/0.02)

# Neumann boundary condition
class Flux(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)
    def eval(self, values, x):
        if x[0] > DOLFIN_EPS:
            values[0] = 25.0*sin(5.0*DOLFIN_PI*x[1])
        else:
            values[0] = 0.0

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and x[0] < DOLFIN_EPS)

# Define variational problem
v = TestFunction(element)
u = TrialFunction(element)
f = Source(element, mesh)
g = Flux(element, mesh)

a = dot(grad(v), grad(u))*dx
L = v*f*dx + v*g*ds

# Create backend
backend = EpetraFactory.instance()

# Assemble matrices
A = assemble(a, mesh, backend=backend)
b = assemble(L, mesh, backend=backend) 

# Define boundary condition
u0 = Function(mesh, 0.0)
boundary = DirichletBoundary()
bc = DirichletBC(u0, mesh, boundary)
bc.apply(A, b, a)

# Create solution vector (also used as start vector) 
x = b.copy()
x.zero()

# Sets up the parameters for ML using a python dictionary
MLList = {"max levels"        : 3, 
          "output"            : 10,
          "smoother: type"    : "ML symmetric Gauss-Seidel",
          "aggregation: type" : "Uncoupled",
          "ML validate parameter list" : False
}

# Create the preconditioner 
Prec = ML.MultiLevelPreconditioner(A.mat(), False)
Prec.SetParameterList(MLList)
Prec.ComputePreconditioner()

# Create solver and solve system 
Solver = AztecOO.AztecOO(A.mat(), x.vec(), b.vec())
Solver.SetPrecOperator(Prec)
Solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_cg)
Solver.SetAztecOption(AztecOO.AZ_output, 16)
Solver.Iterate(1550, 1e-5)

# plot the solution
U = Function(element, mesh, x)
plot(U)
interactive()

# Save solution to file
file = File("poisson.pvd")
file << U



