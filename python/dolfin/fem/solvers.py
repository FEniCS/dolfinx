# Copyright (C) 2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

# import dolfin.cpp as cpp
# from dolfin.fem.form import Form


# __all__ = ["LocalSolver"]


# class LocalSolver(cpp.fem.LocalSolver):

#     def __init__(self, a, L=None, solver_type=cpp.fem.LocalSolver.SolverType.LU):
#         """Create a local (cell-wise) solver for a linear variational problem
#         a(u, v) = L(v).

#         """

#         # Store input UFL forms and solution Function
#         self.a_ufl = a
#         self.L_ufl = L

#         # Wrap as DOLFIN forms
#         a = Form(a)
#         if L is None:
#             # Initialize C++ base class
#             cpp.fem.LocalSolver.__init__(self, a, solver_type)
#         else:
#             if L.empty():
#                 L = cpp.fem.Form(1, 0)
#             else:
#                 L = Form(L)

#         # Initialize C++ base class
#         cpp.fem.LocalSolver.__init__(self, a, L, solver_type)
