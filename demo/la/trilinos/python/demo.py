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

dolfin_set("linear algebra backend", "Epetra")
try:
    EpetraMatrix
except:
    print "PyDOLFIN has not been configured with Trilinos. Exiting."
    exit()

# Create mesh and finite element
mesh = UnitSquare(20,20)
V = FunctionSpace(mesh, "CG", 1)

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] < DOLFIN_EPS

# Define variational problem
v = TestFunction(V)
u = TrialFunction(V)
f = Function(V,"500.0 * exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")


a = dot(grad(v), grad(u))*dx
L = v*f*dx 


# Define boundary condition
u0 = Constant(mesh, 0.0)
bc = DirichletBC(V, u0, DirichletBoundary())

# Create linear system
A, b = assemble_system(a, L, bc) 

# Solution   
U = Function(V)

# Fetch underlying epetra objects 

A_epetra = cpp.down_cast_EpetraMatrix(A).mat() 
b_epetra = cpp.down_cast_EpetraVector(b).vec() 
x_epetra = cpp.down_cast_EpetraVector(U.vector()).vec() 

# Sets up the parameters for ML using a python dictionary
MLList = {"max levels"        : 3, 
          "output"            : 10,
          "smoother: type"    : "ML symmetric Gauss-Seidel",
          "aggregation: type" : "Uncoupled",
          "ML validate parameter list" : False
}

# Create the preconditioner 
Prec = ML.MultiLevelPreconditioner(A_epetra, False)
Prec.SetParameterList(MLList)
Prec.ComputePreconditioner()

# Create solver and solve system 
Solver = AztecOO.AztecOO(A_epetra, x_epetra, b_epetra)
Solver.SetPrecOperator(Prec)
Solver.SetAztecOption(AztecOO.AZ_solver, AztecOO.AZ_cg)
Solver.SetAztecOption(AztecOO.AZ_output, 16)
Solver.Iterate(1550, 1e-5)

# Plot the solution 
plot(U)
interactive()

# Save solution to file
file = File("poisson.pvd")
file << U



