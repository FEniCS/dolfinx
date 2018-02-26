# Copyright (C) 2009-2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""This module provides a simple way to compute the interpolation of a
:py:class:`Function <dolfin.functions.function.Function>` or
:py:class:`Expression <dolfin.functions.expression.Expression>` onto a
finite element space.

"""


from dolfin.function.function import Function


def interpolate(v, V):
    """Return interpolation of a given function into a given finite
    element space.

    *Arguments*
        v
            a :py:class:`Function <dolfin.functions.function.Function>` or
            an :py:class:`Expression <dolfin.functions.expression.Expression>`
        V
            a :py:class:`FunctionSpace (standard, mixed, etc.)
            <dolfin.functions.functionspace.FunctionSpace>`

    *Example of usage*

        .. code-block:: python

            v = Expression("sin(pi*x[0])")
            V = FunctionSpace(mesh, "Lagrange", 1)
            Iv = interpolate(v, V)

    """

    # Check arguments
    # if not isinstance(V, cpp.functionFunctionSpace):
    #     cpp.dolfin_error("interpolation.py",
    #                      "compute interpolation",
    #                      "Illegal function space for interpolation, not a FunctionSpace (%s)" % str(v))

    # Compute interpolation
    Pv = Function(V)

    if hasattr(v, "_cpp_object"):
        Pv.interpolate(v._cpp_object)
    else:
        Pv.interpolate(v)

    return Pv
