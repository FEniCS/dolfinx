"""This demo illustrates how to use of DOLFIN for solving the Cahn-Hilliard 
equation, which is a time-dependent nonlinear PDE """

__author__ = "Garth N. Wells (gnw20@cam.ac.uk)"
__date__ = "2009-06-20"
__copyright__ = "Copyright (C) 2009 Garth N. Wells"
__license__  = "GNU LGPL Version 2.1"

import random
from dolfin import *

class InitialConditions(Function):
    def __init__(self, V):
        Function.__init__(V)
        random.seed(2)
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.63 + 0.02*(0.5 - random.random())

class CahnHilliardEquation(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.a = form.Form(a)
        self.L = form.Form(L)
    def F(self, b, x):
        cpp.assemble(b, self.L)
    def J(self, A, x):
        cpp.assemble(A, self.a)

#------------------------------------------------------------------------------
# Create mesh and define function space
lmbda    = 1.0e-02  # surface parameter
muFactor = 100      # chemical free energy multiplier
dt       = 5.0e-06
theta    = 0.5 

mesh = UnitSquare(32, 32)
V = FunctionSpace(mesh, "CG", 1)
ME = V + V

q, v  = TestFunctions(ME)
du    = TrialFunction(ME)

u   = Function(ME)  # current solution 
u0  = Function(ME)  # solution from previous converged step 

# Split mixed functions
dk, dc = split(du) 
k,  c  = split(u)
k0, c0 = split(u0)

# Potential mu = \phi,c (chemical free-energy \phi = c^2*(1-c)^2)
mu = muFactor*(2.0*c*(1.0-c)*(1.0-c) - 2.0*c*c*(1.0-c))

# k^(n+theta)
k_mid = (1.0-theta)*k0 + theta*k

L1 = q*c*dx - q*c0*dx + dt*dot(grad(q), grad(k_mid))*dx
L2 = v*k*dx - v*mu*dx - lmbda*dot(grad(v), grad(c))*dx

a1 = derivative(L1, u, du)
a2 = derivative(L2, u, du)

L = L1 + L2
a = a1 + a2
#------------------------------------------------------------------------------

u_init = InitialConditions(ME)
u.interpolate(u_init)
u0.interpolate(u)

problem = CahnHilliardEquation(a, L)
newton_solver = NewtonSolver("lu")

file = File("output.pvd")

t = 0.0
T = 2*dt
while (t < T):
    t += dt
    u0.interpolate(u)
    newton_solver.solve(problem, u.vector())
    file << u

# FIXME: Why does the below not work?
#plot(c)

# FIXME: Why does the below not work?
#conc = u[1]
#file = File("output.pvd")
#file << conc
