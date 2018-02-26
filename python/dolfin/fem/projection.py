# Copyright (C) 2008-2011 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""This module provides a simple way to compute the projection of a
:py:class:`Function <dolfin.functions.function.Function>` or an
:py:class:`Expression <dolfin.functions.expression.Expression>` onto a
finite element space.

"""


import ufl
import dolfin.cpp as cpp
import dolfin
from dolfin.function.argument import TestFunction, TrialFunction
from dolfin.function.function import Function
from dolfin.fem.assembling import assemble_system
from dolfin.function.functionspace import (FunctionSpace,
                                           VectorFunctionSpace, TensorFunctionSpace)

__all__ = ['project']


def project(v, V=None, bcs=None, mesh=None,
            function=None,
            solver_type="lu",
            preconditioner_type="default",
            form_compiler_parameters=None):
    """Return projection of given expression *v* onto the finite element
    space *V*.

    *Arguments*
        v
            a :py:class:`Function <dolfin.functions.function.Function>` or
            an :py:class:`Expression <dolfin.functions.expression.Expression>`
        bcs
            Optional argument :py:class:`list of DirichletBC
            <dolfin.fem.bcs.DirichletBC>`
        V
            Optional argument :py:class:`FunctionSpace
            <dolfin.functions.functionspace.FunctionSpace>`
        mesh
            Optional argument :py:class:`mesh <dolfin.cpp.Mesh>`.
        solver_type
            see :py:func:`solve <dolfin.fem.solving.solve>` for options.
        preconditioner_type
            see :py:func:`solve <dolfin.fem.solving.solve>` for options.
        form_compiler_parameters
            see :py:class:`Parameters <dolfin.cpp.Parameters>` for more
            information.

    *Example of usage*

        .. code-block:: python

            v = Expression("sin(pi*x[0])")
            V = FunctionSpace(mesh, "Lagrange", 1)
            Pv = project(v, V)

        This is useful for post-processing functions or expressions
        which are not readily handled by visualization tools (such as
        for example discontinuous functions).

    """

    # Try figuring out a function space if not specified
    if V is None:
        # Create function space based on Expression element if trying
        # to project an Expression
        if isinstance(v, dolfin.function.expression.Expression):
            if mesh is not None and isinstance(mesh, cpp.mesh.Mesh):
                V = FunctionSpace(mesh, v.ufl_element())
            # else:
            #     cpp.dolfin_error("projection.py",
            #                      "perform projection",
            #                      "Expected a mesh when projecting an Expression")
        else:
            # Otherwise try extracting function space from expression
            V = _extract_function_space(v, mesh)

    # Check arguments

    # Ensure we have a mesh and attach to measure
    if mesh is None:
        mesh = V.mesh()
    dx = ufl.dx(mesh)

    # Define variational problem for projection
    w = TestFunction(V)
    Pv = TrialFunction(V)
    a = ufl.inner(w, Pv) * dx
    L = ufl.inner(w, v) * dx

    # Assemble linear system
    A, b = assemble_system(a, L, bcs=bcs,
                           form_compiler_parameters=form_compiler_parameters)

    # Solve linear system for projection
    if function is None:
        function = Function(V)
    cpp.la.solve(A, function.vector(), b, solver_type, preconditioner_type)

    return function


def _extract_function_space(expression, mesh):
    """Try to extract a suitable function space for projection of given
    expression.

    """

    # Get mesh from expression
    if mesh is None:
        domain = expression.ufl_domain()
        if domain is not None:
            mesh = domain.ufl_cargo()

    # Extract mesh from functions
    if mesh is None:
        # (Not sure if this code is relevant anymore, the above code
        # should cover this)
        # Extract functions
        functions = ufl.algorithms.extract_coefficients(expression)
        for f in functions:
            if isinstance(f, Function):
                mesh = f.function_space().mesh()
                if mesh is not None:
                    break

    if mesh is None:
        raise RuntimeError("Unable to project expression, cannot find a suitable mesh.")

    # Create function space
    shape = expression.ufl_shape
    if shape == ():
        V = FunctionSpace(mesh, "Lagrange", 1)
    elif len(shape) == 1:
        V = VectorFunctionSpace(mesh, "Lagrange", 1, dim=shape[0])
    elif len(shape) == 2:
        V = TensorFunctionSpace(mesh, "Lagrange", 1, shape=shape)
    else:
        raise RuntimeError("Unhandled rank, shape is {}.".format((shape,)))

    return V
