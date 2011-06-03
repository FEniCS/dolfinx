# Copyright (C) 2008 Evan Lezar
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
# Forms for solving the transverse magnetic (TM) cutoff modes of a waveguide
#
# Compile this form with FFC: ffc -l dolfin Forms.form

from ufl import *
from ufl.log import set_level
from ffc.compiler.compiler import compile

# Set debug level
set_level(20)

# Reserved variables for forms
(a, L, M) = (None, None, None)

# Reserved variable for element
element = None

element = FiniteElement("Nedelec 1st kind H(curl)", "triangle", 2)

v = TestFunction(element)
u = TrialFunction(element)

def curl_t(w):
    return Dx(w[1], 0) - Dx(w[0], 1)

a = curl_t(v)*curl_t(u)*dx
L = inner(v, u)*dx

compile([a, L, M, element], "Forms", {'log_level': 20, 'format': 'dolfin', 'form_postfix': True, 'quadrature_order': 'auto', 'precision': '15', 'cpp optimize': False, 'split_implementation': False, 'cache_dir': None, 'output_dir': '.', 'representation': 'auto', 'optimize': True}, globals())
