# -*- coding: utf-8 -*-
"""
This module provides a DOLFINErrorControlGenerator for handling UFL
forms defined over DOLFIN objects.
"""

# Copyright (C) 2011 Marie E. Rognes
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

from ffc.errorcontrol.errorcontrolgenerators import ErrorControlGenerator

from dolfin import Function, FunctionSpace
from dolfin.fem.formmanipulations import tear, increase_order


class DOLFINErrorControlGenerator(ErrorControlGenerator):
    """
    This class provides a realization of
    ffc.errorcontrol.errorcontrolgenerators.ErrorControlGenerator for
    use with UFL forms defined over DOLFIN objects
    """

    def __init__(self, F, M, u):
        """
        *Arguments*

            F (tuple or Form)
               tuple of (bilinear, linear) forms or linear form

            M (Form)
               functional or linear form

            u (Coefficient)
              The coefficient considered as the unknown.
        """
        ErrorControlGenerator.__init__(self, __import__("dolfin"), F, M, u)

    def initialize_data(self):
        """
        Extract required objects for defining error control
        forms. This will be stored and reused.
        """
        # Developer's note: The UFL-FFC-DOLFIN--PyDOLFIN toolchain for
        # error control is quite fine-tuned. In particular, the order
        # of coefficients in forms is (and almost must be) used for
        # their assignment. This means that the order in which these
        # coefficients are defined matters and should be considered
        # fixed.

        # Primal trial element space
        self._V = self.u.function_space()

        # Primal test space == Dual trial space
        Vhat = self.weak_residual.arguments()[0].function_space()

        # Discontinuous version of primal trial element space
        self._dV = tear(self._V)

        # Extract cell and geometric dimension
        mesh = self._V.mesh()
        dim = mesh.topology().dim()

        # Function representing improved dual
        E = increase_order(Vhat)
        self._Ez_h = Function(E)

        # Function representing cell bubble function
        B = FunctionSpace(mesh, "B", dim + 1)
        self._b_T = Function(B)
        self._b_T.vector()[:] = 1.0

        # Function representing strong cell residual
        self._R_T = Function(self._dV)

        # Function representing cell cone function
        C = FunctionSpace(mesh, "DG", dim)
        self._b_e = Function(C)

        # Function representing strong facet residual
        self._R_dT = Function(self._dV)

        # Function for discrete dual on primal test space
        self._z_h = Function(Vhat)

        # Piecewise constants for assembling indicators
        self._DG0 = FunctionSpace(mesh, "DG", 0)
