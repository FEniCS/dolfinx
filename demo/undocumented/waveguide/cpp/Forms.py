from ufl import *
from ufl.log import set_level
from ffc.compiler.compiler import compile

# Set debug level
set_level(20)

# Reserved variables for forms
(a, L, M) = (None, None, None)

# Reserved variable for element
element = None

# Copyright (C) 2008 Evan Lezar.
#
# Forms for solving the transverse magnetic (TM) cutoff modes of a waveguide
#
# Compile this form with FFC: ffc -l dolfin Forms.form

element = FiniteElement("Nedelec 1st kind H(curl)", "triangle", 2)

v = TestFunction(element)
u = TrialFunction(element)

def curl_t(w):
    return Dx(w[1], 0) - Dx(w[0], 1)

a = curl_t(v)*curl_t(u)*dx
L = inner(v, u)*dx

compile([a, L, M, element], "Forms", {'log_level': 20, 'format': 'dolfin', 'form_postfix': True, 'quadrature_order': 'auto', 'precision': '15', 'cpp optimize': False, 'split_implementation': False, 'cache_dir': None, 'output_dir': '.', 'representation': 'auto', 'optimize': True}, globals())
