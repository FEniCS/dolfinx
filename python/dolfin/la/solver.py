# -*- coding: utf-8 -*-
"""This module provides a small Python layer for solving linear
sytems.

"""

# Copyright (C) 2011-2017 Ander Logg and Garth N. Wells
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


def solve(A, x, b, method="default", preconditioner="default"):
    """Solve linear system Ax = b.

    A linear system Ax = b may be solved by calling solve(A, x, b),
    where A is a matrix and x and b are vectors. Optional arguments
    may be passed to specify the solver method and preconditioner.
    Some examples are given below:

    .. code-block:: python

        solve(A, x, b)
        solve(A, x, b, "lu")
        solve(A, x, b, "gmres", "ilu")

    Possible values for the solver method and preconditioner depend
    on which linear algebra backend is used and how that has been
    configured.

    To list all available LU methods, run the following command:

    .. code-block:: python

        list_lu_solver_methods()

    To list all available Krylov methods, run the following command:

    .. code-block:: python

        list_krylov_solver_methods()

    To list all available preconditioners, run the following command:

    .. code-block:: python

        list_krylov_solver_preconditioners()

    To list all available solver methods, including LU methods, Krylov
    methods and, possibly, other methods, run the following command:

    .. code-block:: python

        list_linear_solver_methods()

    """

    return cpp.la.solve(A, x, b, method, preconditioner)
