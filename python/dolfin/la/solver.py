# -*- coding: utf-8 -*-
# Copyright (C) 2011-2017 Anders Logg and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""This module provides a small Python layer for solving linear
sytems.

"""

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
