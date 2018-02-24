# -*- coding: utf-8 -*-
# Copyright (C) 2011-2017 Anders Logg and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import dolfin.cpp as cpp
from dolfin.fem.form import Form


# class LinearVariationalProblem(cpp.fem.LinearVariationalProblem):

#     def __init__(self, a, L, u, bcs=None, form_compiler_parameters=None):
#         """Create linear variational problem a(u, v) = L(v).

#         An optional argument bcs may be passed to specify boundary
#         conditions.

#         Another optional argument form_compiler_parameters may be
#         specified to pass parameters to the form compiler.

#         """

#         if bcs is None:
#             bcs = []
#         elif not isinstance(bcs, (list, tuple)):
#             bcs = [bcs]

#         # Store input UFL forms and solution Function
#         self.a_ufl = a
#         self.L_ufl = L
#         self.u_ufl = u

#         # Store form compiler parameters
#         form_compiler_parameters = form_compiler_parameters or {}
#         self.form_compiler_parameters = form_compiler_parameters

#         # Wrap forms (and check if linear form L is empty)
#         if L.empty():
#             L = cpp.fem.Form(1, 0)
#         else:
#             L = Form(L, form_compiler_parameters=form_compiler_parameters)
#         a = Form(a, form_compiler_parameters=form_compiler_parameters)

#         # Initialize C++ base class
#         cpp.fem.LinearVariationalProblem.__init__(self, a, L, u._cpp_object, bcs)


# class NonlinearVariationalProblem(cpp.fem.NonlinearVariationalProblem):

#     def __init__(self, F, u, bcs=None, J=None, form_compiler_parameters=None):
#         """Create nonlinear variational problem F(u; v) = 0.

#         Optional arguments bcs and J may be passed to specify boundary
#         conditions and the Jacobian J = dF/du.

#         Another optional argument form_compiler_parameters may be
#         specified to pass parameters to the form compiler.

#         """

#         if bcs is None:
#             bcs = []
#         elif not isinstance(bcs, (list, tuple)):
#             bcs = [bcs]

#         # Store input UFL forms and solution Function
#         self.F_ufl = F
#         self.J_ufl = J
#         self.u_ufl = u

#         # Store form compiler parameters
#         form_compiler_parameters = form_compiler_parameters or {}
#         self.form_compiler_parameters = form_compiler_parameters

#         # Wrap forms
#         F = Form(F, form_compiler_parameters=form_compiler_parameters)
#         if J is not None:
#             J = Form(J, form_compiler_parameters=form_compiler_parameters)

#         # Initialize C++ base class
#         cpp.fem.NonlinearVariationalProblem.__init__(self, F, u._cpp_object, bcs, J)
