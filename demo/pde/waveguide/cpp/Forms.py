from ffc import *

# Reserved variables for forms
(a, L, M) = (None, None, None)

# Reserved variable for element
element = None

# Copyright (C) 2008 Evan Lezar.
#
# Forms for solving the transverse magnetic (TM) cutoff modes of a waveguide
#
# Compile this form with FFC: ffc -l dolfin Forms.form

element = FiniteElement("Nedelec", "triangle", 2)

v = TestFunction(element)
u = TrialFunction(element)


a = dot(curl_t(v), curl_t(u))*dx
L = dot(v, u)*dx
compile([a, L, M, element], "Forms", {'language': 'dolfin', 'blas': False, 'form_postfix': True, 'precision': '15', 'cpp optimize': False, 'split_implementation': False, 'quadrature_points': False, 'output_dir': '.', 'representation': 'tensor', 'cache_dir': None, 'optimize': False})
