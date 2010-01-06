"""This demo illustrates how to use of DOLFIN for solving the Cahn-Hilliard 
equation, which is a time-dependent nonlinear PDE """

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2009-06-20"
__copyright__ = "Copyright (C) 2009 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

import random
from dolfin import *

class InitialConditions(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.63 + 0.02*(0.5 - random.random())
    def dim(self):
        return 2


class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.reset_sparsity = True
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A, reset_sparsity=self.reset_sparsity)
        self.reset_sparsity = False

#------------------------------------------------------------------------------
# Create mesh and define function space
lmbda  = 1.0e-02  # surface parameter
factor = 100      # chemical free energy multiplier
dt     = 5.0e-06  # time step
theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson   

parameters.optimize = True

# Define function spaces
mesh = UnitSquare(96, 96)
V = FunctionSpace(mesh, "CG", 1)
ME = V + V

# Define test and trial functions
q, v  = TestFunctions(ME)
du    = TrialFunction(ME)

# Define functions
u   = Function(ME)  # current solution 
u0  = Function(ME)  # solution from previous converged step 

# Split mixed functions
dk, dc = split(du) 
k,  c  = split(u)
k0, c0 = split(u0)

# Potential mu = \phi,c (chemical free-energy \phi = c^2*(1-c)^2)
mu = factor*(2.0*c*(1.0-c)*(1.0-c) - 2.0*c*c*(1.0-c))

# k^(n+theta)
k_mid = (1.0-theta)*k0 + theta*k

L1 = q*c*dx - q*c0*dx + dt*dot(grad(q), grad(k_mid))*dx
L2 = v*k*dx - v*mu*dx - lmbda*dot(grad(v), grad(c))*dx

L = L1 + L2
a = derivative(L, u, du)
#------------------------------------------------------------------------------

# Create intial conditions and interpolate
u_init = InitialConditions()
u.interpolate(u_init)
u0.interpolate(u_init)

# Create nonlinear problem and Newton solver
problem = CahnHilliardEquation(a, L)
solver = NewtonSolver("lu")
solver.parameters["convergence_criterion"] = "incremental"
solver.parameters["relative_tolerance"] = 1e-6

# Output file
file = File("output.pvd", "compressed")

t = 0.0
T = 80*dt
while (t < T):
    t += dt
    u0.vector()[:] = u.vector()[:]
    solver.solve(problem, u.vector())
    file << u.split()[1]

plot(u.split()[1])
interactive()
