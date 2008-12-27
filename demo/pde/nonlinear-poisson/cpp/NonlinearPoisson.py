from ffc import *

# Reserved variables for forms
(a, L, M) = (None, None, None)

# Reserved variable for element
element = None

# Copyright (C) 2005 Garth N. Wells.
# Licensed under the GNU LGPL Version 2.1.
#
# The linearised bilinear form a(v, U) and linear form L(v) for
# the nonlinear equation - div (1+u^2) grad u = f
#
# Compile this form with FFC: ffc -l dolfin NonlinearPoisson.form.

element = FiniteElement("Lagrange", "triangle", 1)

v = TestFunction(element)
u = TrialFunction(element)
f = Function(element)
U = Function(element)
a = v.dx(i)*(1.0 + U*U)*u.dx(i)*dx + v.dx(i)*(2.0*U*u)*U.dx(i)*dx
L = v.dx(i)*(1.0 + U*U)*U.dx(i)*dx - v*f*dx

compile([a, L, M, element], "NonlinearPoisson", options={'language': 'dolfin', 'blas': False, 'form_postfix': True, 'precision': '15', 'cache_dir': None, 'cpp optimize': False, 'split_implementation': False, 'quadrature_points': False, 'output_dir': '.', 'representation': 'tensor', 'shared_ptr': False, 'optimize': False}, global_variables=globals())
