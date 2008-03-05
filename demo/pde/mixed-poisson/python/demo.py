# This demo program solves the mixed formulation of
# Poisson's equation:
#
#     sigma + grad(u) = 0
#          div(sigma) = f
#
# The corresponding weak (variational problem)
#
#     <tau, sigma> - <div(tau), u> = 0       for all tau
#                  <w, div(sigma)> = <w, f>  for all w
#
# is solved using BDM (Brezzi-Douglas-Marini) elements
# of degree q (tau, sigma) and DG (discontinuous Galerkin)
# elements of degree q - 1 for (w, u).
#
# Original implementation: ../cpp/main.cpp by Anders Logg and Marie Rognes
#
__author__ = "Kristian B. Oelgaard (k.b.oelgaard@tudelft.nl)"
__date__ = "2007-11-14 -- 2007-11-28"
__copyright__ = "Copyright (C) 2007 Kristian B. Oelgaard"
__license__  = "GNU LGPL Version 2.1"

from dolfin import *

#
# THIS DEMO WORKS, BUT ...
# when extracting subfunctions SWIG issues the following warning:
# swig/python detected a memory leak of type 'SubFunction *', no destructor found.
#

# Create elements and mesh
q = 1
BDM = FiniteElement("Brezzi-Douglas-Marini", "triangle", q)
DG  = FiniteElement("Discontinuous Lagrange", "triangle", q - 1)
mixed_element = BDM + DG

mesh = UnitSquare(16, 16)

# Source term
class Source(Function):
    def __init__(self, element, mesh):
        Function.__init__(self, element, mesh)

    def eval(self, values, x):
        dx = x[0] - 0.5
        dy = x[1] - 0.5
        values[0] =  500.0*exp(-(dx*dx + dy*dy)/0.02)

(tau, w) = TestFunctions(mixed_element)
(sigma, u) = TrialFunctions(mixed_element)
f = Source(DG, mesh)


a = (dot(tau, sigma) - div(tau)*u + w*div(sigma))*dx
L = w*f*dx

# Define PDE
pde = LinearPDE(a, L, mesh)

# Solve PDE and get sub-functions
(sigma, u) = pde.solve().split()

# Plot solution
plot(sigma)
plot(u)

# Save solution to file
f0 = File("sigma.xml")
f1 = File("u.xml")
f0 << sigma
f1 << u

# Save solution to pvd format
f3 = File("sigma.pvd")
f4 = File("u.pvd")
f3 << sigma
f4 << u



