# -*- coding: utf-8 -*-
# Copyright (C) 2011-2017 Anders Logg and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Simpler interface for solving linear systems"""

from dolfin import cpp


def solve(A, x, b):
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

    """

    solver = cpp.la.PETScKrylovSolver(cpp.MPI.comm_world)
    solver.set_operator(A)
    solver.solve(x, b)
    return x
