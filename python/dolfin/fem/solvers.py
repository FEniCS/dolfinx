# -*- coding: utf-8 -*-

# Copyright (C) 2011 Anders Logg
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

import dolfin.cpp as cpp
from dolfin.fem.form import Form

__all__ = ["LocalSolver"]


class LocalSolver(cpp.fem.LocalSolver):

    def __init__(self, a, L=None, solver_type=cpp.fem.LocalSolver.SolverType.LU):
        """Create a local (cell-wise) solver for a linear variational problem
        a(u, v) = L(v).

        """

        # Store input UFL forms and solution Function
        self.a_ufl = a
        self.L_ufl = L

        # Wrap as DOLFIN forms
        a = Form(a)
        if L is None:
            # Initialize C++ base class
            cpp.fem.LocalSolver.__init__(self, a, solver_type)
        else:
            if L.empty():
                L = cpp.fem.Form(1, 0)
            else:
                L = Form(L)

        # Initialize C++ base class
        cpp.fem.LocalSolver.__init__(self, a, L, solver_type)
