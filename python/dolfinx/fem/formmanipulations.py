# Copyright (C) 2010-2012 Marie E. Rognes
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
from dolfinx.fem import function


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
    v1 = function.Argument(arguments[1].function_space,
                           arguments[0].number(), arguments[0].part())
    v0 = function.Argument(arguments[0].function_space,
                           arguments[1].number(), arguments[1].part())

    # Return form with swapped arguments as new arguments
    return ufl.adjoint(form, reordered_arguments=(v1, v0))
