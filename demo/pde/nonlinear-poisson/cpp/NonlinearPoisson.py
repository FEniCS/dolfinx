from ufl import *
from ufl.log import set_level
from ffc.compiler.compiler import compile

# Set debug level
set_level(20)

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

compile([a, L, M, element], "NonlinearPoisson", {'log_level': 20, 'language': 'dolfin', 'format': 'dolfin', 'form_postfix': True, 'quadrature_order': 'auto', 'precision': '15', 'cpp optimize': False, 'split_implementation': False, 'cache_dir': None, 'output_dir': '.', 'representation': 'auto', 'optimize': True}, globals())
