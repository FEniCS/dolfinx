# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""FIXME: add description"""

import types
import typing

import ufl
from dolfin import cpp, fem, function


class AutoSubDomain(cpp.mesh.SubDomain):
    "Wrapper class for creating a SubDomain from an inside() function."

    def __init__(self, inside_function: types.FunctionType):
        "Create SubDomain subclass for given inside() function"

        # Check that we get a function
        if not isinstance(inside_function, types.FunctionType):
            raise RuntimeError(
                "bcs.py", "auto-create subdomain",
                "Expecting a function (not %s)" % str(type(inside_function)))
        self.inside_function = inside_function

        # Check the number of arguments
        if inside_function.__code__.co_argcount not in (1, 2):
            raise RuntimeError(
                "bcs.py", "auto-create subdomain",
                "Expecting a function of the form inside(x) or inside(x, on_boundary)"
            )
        self.num_args = inside_function.__code__.co_argcount

        super().__init__()

    def inside(self, x, on_boundary):
        "Return true for points inside the subdomain"

        if self.num_args == 1:
            return self.inside_function(x)
        else:
            return self.inside_function(x, on_boundary)


class DirichletBC(cpp.fem.DirichletBC):
    def __init__(
            self,
            V: typing.Union[function.FunctionSpace,
                            cpp.function.FunctionSpace],
            value: typing.Union[ufl.Coefficient, cpp.function.GenericFunction,
                                list, tuple, float, int],
            domain: typing.Union[cpp.mesh.SubDomain, types.FunctionType],
            method: cpp.fem.DirichletBC.Method = cpp.fem.DirichletBC.Method.
            topological,
            check_midpoint: typing.Optional[bool] = None):
        """Representation of Dirichlet boundary conditions which are imposed on
        linear systems.

        """

        # FIXME: Handle (mesh function, index) marker type? If yes, use
        # tuple domain=(mf, index) to not have variable arguments

        # Extract cpp function space
        try:
            _V = V._cpp_object
        except AttributeError:
            _V = V

        # Construct bc value
        if isinstance(value, ufl.Coefficient):
            _value = value.cpp_object()
        elif isinstance(value, cpp.function.GenericFunction):
            _value = value
        else:
            _value = cpp.function.Constant(value)

        # Construct domain
        if isinstance(domain, types.FunctionType):
            _domain = AutoSubDomain(domain)
            self.sub_domain = _domain
        else:
            _domain = domain

        if not check_midpoint:
            check_midpoint = True

        super().__init__(_V, _value, _domain, method, check_midpoint)
