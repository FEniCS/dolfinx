# Copyright (C) 2010-2012 Marie E. Rognes
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import ufl
import ufl.algorithms.elementtransformations
from dolfin.function.functionspace import FunctionSpace
from dolfin.function.function import Function
from dolfin.function.argument import Argument

__all__ = ["derivative", "adjoint", "increase_order", "tear"]


def adjoint(form, reordered_arguments=None):

    # Call UFL directly if new arguments are provided directly
    if reordered_arguments is not None:
        return ufl.adjoint(form, reordered_arguments=reordered_arguments)

    # Extract form arguments
    arguments = form.arguments()
    if any(arg.part() is not None for arg in arguments):
        raise RuntimeError("Compute adjoint of form, parts not supported")

    if not (len(arguments) == 2):
        raise RuntimeError("Compute adjoint of form, form is not bilinear")

    # Define new Argument(s) in the same spaces (NB: Order does not
    # matter anymore here because number is absolute)
    v_1 = Argument(arguments[1].function_space(), arguments[0].number(),
                   arguments[0].part())
    v_0 = Argument(arguments[0].function_space(), arguments[1].number(),
                   arguments[1].part())

    # Call ufl.adjoint with swapped arguments as new arguments
    return ufl.adjoint(form, reordered_arguments=(v_1, v_0))


adjoint.__doc__ = ufl.adjoint.__doc__


def derivative(form, u, du=None, coefficient_derivatives=None):
    if du is None:
        # Get existing arguments from form and position the new one
        # with the next argument number
        form_arguments = form.arguments()

        number = max([-1] + [arg.number() for arg in form_arguments]) + 1

        if any(arg.part() is not None for arg in form_arguments):
            raise RuntimeError("Compute derivative of form, cannot automatically create new Argument using parts, please supply one")
        part = None

        if isinstance(u, Function):
            V = u.function_space()
            du = Argument(V, number, part)
        elif isinstance(u, (list, tuple)) and all(isinstance(w, Function) for w in u):
            raise RuntimeError("Taking derivative of form w.r.t. a tuple of Coefficients. Take derivative w.r.t. a single Coefficient on a mixed space instead.")
        else:
            raise RuntimeError("Computing derivative of form w.r.t. '{}'. Supply Function as a Coefficient".format(u))

    return ufl.derivative(form, u, du, coefficient_derivatives)


derivative.__doc__ = ufl.derivative.__doc__
derivative.__doc__ += """

    A tuple of Coefficients in place of a single Coefficient is not
    supported in DOLFIN. Supply rather a Function on a mixed space in
    place of a Coefficient.
    """


def increase_order(V):
    """For a given function space, return the same space, but with a
    higher polynomial degree

    """
    mesh = V.mesh()
    element = ufl.algorithms.elementtransformations.increase_order(V.ufl_element())
    constrained_domain = V.dofmap().constrained_domain
    return FunctionSpace(mesh, element, constrained_domain=constrained_domain)


def change_regularity(V, family):
    """For a given function space, return the corresponding space with
    the finite elements specified by 'family'. Possible families are
    the families supported by the form compiler

    """
    mesh = V.mesh()
    element = ufl.algorithms.elementtransformations.change_regularity(V.ufl_element(), family)
    constrained_domain = V.dofmap().constrained_domain
    return FunctionSpace(mesh, element, constrained_domain=constrained_domain)


def tear(V):
    """
    For a given function space, return the corresponding discontinuous
    space
    """
    return change_regularity(V, "DG")
