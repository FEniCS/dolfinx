"""
  This example demonstrates use of the TAO package to
  solve a bound constrained minimization problem.  
  This example is based on the problem DPJB from the MINPACK-2 test suite.  
  and is the demo for bound constrained quadratic problem in the TAO package  
  This pressure journal  bearing problem is an example of elliptic variational 
  problem defined over a two dimensional rectangle.  
  By discretizing the domain into triangular elements, the pressure surrounding
  the journal bearing is defined as the minimum of a quadratic function 
  whose variables are bounded below by zero.
  
  For a detailed problem description see pgg 33-34 of
  http://ftp.mcs.anl.gov/pub/tech_reports/reports/P153.pdf
  
"""

# Copyright (C) 2007-2011 Anders Logg
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
# Last changed: 03/09/2012

# Begin demo  
# Corrado Maurini 
#
from dolfin import *
import unittest

try:
  parameters["linear_algebra_backend"] = "PETSc"
except RuntimeError:
  import sys; sys.exit(0)
  
# Create mesh and define function space
b=10
eps=0.1
mesh = Rectangle(0,0,2*pi,2*b,100,500)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x,on_boundary):
    return on_boundary

# Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
wq = Expression("pow(1 + eps * cos(x[0]),3)",eps=eps)
wl = Expression("eps * sin(x[0])",eps=eps)

a = wq*inner(grad(u), grad(v))*dx
L = wl*v*dx

# Assemble the linear system
A=PETScMatrix()
b=PETScVector()
A=assemble(a)
b=assemble(L)
bc.apply(A)
bc.apply(b)

# Define the upper and lower bounds
upperbound = interpolate(Constant(10.), V) # set a large upper-bound, which is never reached
lowerbound = interpolate(Constant(0.), V) 
xu=upperbound.vector() # or xu=down_cast(upperbound.vector())
xl=lowerbound.vector() # or xl=down_cast(lowerbound.vector())

# Define the function to store the solution and the related PETScVector
usol=Function(V);
xsol=usol.vector() # or xsol=down_cast(usol.vector())

# Create the TAOLinearBoundSolver 
solver=TAOLinearBoundSolver("tao_gpcg")

# Set some parameters
solver.parameters["krylov_solver"]["absolute_tolerance"]=1e-8
solver.parameters["krylov_solver"]["relative_tolerance"]=1e-8
solver.solve(A,xsol,b,xl,xu)


if __name__ == "__main__":
  # Turn off DOLFIN output
  set_log_active(False)

  print ""
  print "Testing DOLFIN nls/PETScSNESSolver interface"
  print "--------------------------------------------"
  unittest.main()