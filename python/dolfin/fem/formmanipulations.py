# -*- coding: utf-8 -*-
# Copyright (C) 2010-2012 Marie E. Rognes
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
from dolfin import function


def adjoint(form: ufl.Form, reordered_arguments=None) -> ufl.Form:
    """Compute adjoint of a bilinear form by changing the ordering (count)
    of the test and trial functions.

    The functions wraps ``ufl.adjoint``, and by default UFL will create new
    ``Argument`` s. To specify the ``Argument`` s rather than creating new ones,
    pass a tuple of ``Argument`` s as ``reordered_arguments``.
    See the documentation for ``ufl.adjoint`` for more details.

    """

    if reordered_arguments is not None:
        return ufl.adjoint(form, reordered_arguments=reordered_arguments)

    # Extract form arguments
    arguments = form.arguments()
    if len(arguments) != 2:
        raise RuntimeError(
            "Cannot compute adjoint of form, form is not bilinear")
    if any(arg.part() is not None for arg in arguments):
        raise RuntimeError(
            "Cannot compute adjoint of form, parts not supported")

    # Create new Arguments in the same spaces (NB: Order does not matter
    # anymore here because number is absolute)
    v1 = function.Argument(arguments[1].function_space(),
                           arguments[0].number(), arguments[0].part())
    v0 = function.Argument(arguments[0].function_space(),
                           arguments[1].number(), arguments[1].part())

    # Return form with swapped arguments as new arguments
    return ufl.adjoint(form, reordered_arguments=(v1, v0))


def derivative(form: ufl.Form, u, du,
               coefficient_derivatives=None) -> ufl.Form:
    """Compute derivative of from about u (coefficient) in the direction
    of du (Argument)

    """
    return ufl.derivative(form, u, du, coefficient_derivatives)


def increase_order(V: function.FunctionSpace) -> function.FunctionSpace:
    """For a given function space, return the same space, but with
    polynomial degree increase by 1.

    """
    e = ufl.algorithms.elementtransformations.increase_order(V.ufl_element())
    return function.FunctionSpace(V.mesh, e)


def change_regularity(V: function.FunctionSpace,
                      family: str) -> function.FunctionSpace:
    """For a given function space, return the corresponding space with
    the finite elements specified by 'family'. Possible families are
    the families supported by the form compiler

    """
    e = ufl.algorithms.elementtransformations.change_regularity(
        V.ufl_element(), family)
    return function.FunctionSpace(V.mesh, e)


def tear(V: function.FunctionSpace) -> function.FunctionSpace:
    """For a given function space, return the corresponding discontinuous
    space

    """
    return change_regularity(V, "DG")
