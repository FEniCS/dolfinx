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
# Modified by Harish Narayanan, 2009.
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
L = inner(grad(v), (1 + U**2)*grad(U))*dx - v*f*dx
a = derivative(L, U, u)


compile([a, L, M, element], "NonlinearPoisson", {'log_level': 20, 'format': 'dolfin', 'form_postfix': True, 'quadrature_order': 'auto', 'precision': '15', 'cpp optimize': False, 'cache_dir': None, 'split': False, 'representation': 'auto', 'optimize': True, 'quadrature_rule': None, 'output_dir': '.'}, globals())
