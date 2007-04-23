from ffc.compiler.compiler import *

# Copyright (c) 2005 Johan Jansson (johanjan@math.chalmers.se)
# Licensed under the GNU LGPL Version 2.1
#
# First added:  2005
# Last changed: 2006-03-28
#
# The bilinear form for classical linear elasticity (Navier)
# Compile this form with FFC: ffc Elasticity.form.

K1 = FiniteElement("Vector Lagrange", "tetrahedron", 1)
K2 = FiniteElement("Vector Lagrange", "tetrahedron", 1)

K = K1 + K2

(v_0, v_1) = TestFunctions(K)
(U1_0, U1_1) = TrialFunctions(K)
(U0_0, U0_1) = Functions(K)

f = Function(K2)

# Dimension of domain
d = K1.shapedim()

def epsilon(u):
    return 0.5 * (grad(u) + transp(grad(u)))

def E(e):
    Ee = mult(7.7, e) + mult(11.5, mult(trace(e), Identity(d)))
    #Ee = mult(7.7, e)
    
    return Ee
        
sigma = E(epsilon(U0_0))

a = (dot(U1_0, v_0) + dot(U1_1, v_1)) * dx
L = (dot(U0_1, v_0) - dot(sigma, epsilon(v_1)) + dot(f, v_1)) * dx
