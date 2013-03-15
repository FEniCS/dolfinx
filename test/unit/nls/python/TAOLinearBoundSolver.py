"""Unit tests for the TAOLinearBoundSolver interface"""
# Copyright (C) 2013 Corrado Maurini
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  03/09/2012
# Last changed: 03/15/2012

# Begin demo  
# Corrado Maurini 
#
# This example solve the bound constrained minimization problem  
# in the domain (x,y) in [0,Lx]x[0,Ly]
#
# min F(u) with  0<=u<=1 and u(0,y)= 0, u(Lx,y) = 1
# 
# where F(u) is the quadratic functionaldefined by the form 
#
# F(u) = 3./4.*(ell/2.*inner(grad(u), grad(u))+ 2./ell*usol)*dx
#
# An analytical is available: 
# u(x,y) = 0 for 0<x<1-ell,  u(x,y) = (x-(1-ell))^2 for 1-ell<x<Lx
# and the value of the functional at the solution usol is F(usol)=Ly
# for any value of ell, with 0<ell<Lx. 

from dolfin import *
import unittest

try:
  parameters["linear_algebra_backend"] = "PETSc"
except RuntimeError:
  import sys; sys.exit(0)

# Create mesh and define function space
Lx = 1; Ly = .1
mesh = RectangleMesh(0,0,Lx,Ly,100,10)
V    = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundaries
def left(x,on_boundary):
    return on_boundary and x[0]==0.

def rigth(x,on_boundary):
    return on_boundary and x[0]==1.
    
# Define boundary conditions
zero = Constant(0.0)
one = Constant(1.0)
bc_l = DirichletBC(V, zero, left)
bc_r = DirichletBC(V, one, rigth)
bc=[bc_l,bc_r]

# Define variational problem
usol = Function(V)
u = TrialFunction(V)
v = TestFunction(V)
cv = Constant(3./4.)
ell = Constant(.5) # This should be smaller than Lx
F = cv*(ell/2.*inner(grad(usol), grad(usol))*dx + 2./ell*usol*dx)
# Weak form
a  = cv*ell*inner(grad(u), grad(v))*dx
L  = -cv*2*v/ell*dx

# Assemble the linear system
A=PETScMatrix()
b=PETScVector()
A, b = assemble_system(a, L, bc)

# Define the upper and lower bounds
upperbound = interpolate(Constant(1.), V) # 
lowerbound = interpolate(Constant(0.), V) 
xu = upperbound.vector()
xl = lowerbound.vector() 

# Take the PETScVector of the solution function
xsol=usol.vector()  

if has_tao():
    class TAOLinearBoundSolverTester(unittest.TestCase):

        def test_tao_linear_bound_solver(self):
            "Test TAOLinearBoundSolver"
            solver=TAOLinearBoundSolver("tao_tron","gmres")
            solver.solve(A,xsol,b,xl,xu)
            # Test that F(usol) = Ly
            self.assertAlmostEqual(assemble(F), Ly, 5)
    
if __name__ == "__main__":

    # Turn off DOLFIN output
    set_log_active(False)

    print ""
    print "Testing DOLFIN TAOLinearBoundSolver interface"
    print "----------------------------------------"
    unittest.main()