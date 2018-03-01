# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""FIXME: add description"""

import types
import ufl
import dolfin.cpp as cpp
from dolfin.function.constant import Constant
from dolfin.function.functionspace import FunctionSpace
from dolfin.fem.projection import project


class AutoSubDomain(cpp.mesh.SubDomain):
    "Wrapper class for creating a SubDomain from an inside() function."

    def __init__(self, inside_function):
        "Create SubDomain subclass for given inside() function"

        # Check that we get a function
        if not isinstance(inside_function, types.FunctionType):
            raise RuntimeError("bcs.py",
                               "auto-create subdomain",
                               "Expecting a function (not %s)" %
                               str(type(inside_function)))
        self.inside_function = inside_function

        # Check the number of arguments
        if inside_function.__code__.co_argcount not in (1, 2):
            raise RuntimeError("bcs.py",
                               "auto-create subdomain",
                               "Expecting a function of the form inside(x) or inside(x, on_boundary)")
        self.num_args = inside_function.__code__.co_argcount

        super().__init__()

    def inside(self, x, on_boundary):
        "Return true for points inside the subdomain"

        if self.num_args == 1:
            return self.inside_function(x)
        else:
            return self.inside_function(x, on_boundary)


class DirichletBC(cpp.fem.DirichletBC):
    def __init__(self, *args, **kwargs):

        # FIXME: the logic in this function is really messy and
        # unclear

        # Copy constructor
        if len(args) == 1:
            if not isinstance(args[0], cpp.fem.DirichletBC):
                raise RuntimeError(
                    "Expecting a DirichleBC as only argument for copy constructor")

            # Initialize base class
            cpp.fem.DirichletBC.__init__(self, args[0])
            return

        # Get FunctionSpace
        if not isinstance(args[0], FunctionSpace):
            raise RuntimeError("First argument must be of type FunctionSpace")

        # FIXME: correct the below comment
        # Case: boundary value specified as float, tuple or similar
        # if len(args) >= 2 and not isinstance(args[1], (cpp.function.GenericFunction):
        if len(args) >= 2:
            # Check if we have a UFL expression or a concrete type
            if not hasattr(args[1], "_cpp_object"):
                if isinstance(args[1], ufl.classes.Expr):
                    # FIXME: This should really be interpolaton (project is expensive)
                    expr = project(args[1], args[0])
                else:
                    expr = Constant(args[1])
                args = args[:1] + (expr,) + args[2:]

        # Get boundary condition field (the condition that is applied)
        if isinstance(args[1], float) or isinstance(args[1], int):
            u = cpp.function.Constant(float(args[1]))
        elif isinstance(args[1], ufl.Coefficient):
            u = args[1].cpp_object()
        elif isinstance(args[1], cpp.function.GenericFunction):
            u = args[1]
        else:
            raise RuntimeError("Second argument must be convertiable to a GenericFunction: ",
                               args[1], type(args[1]))
        args = args[:1] + (u,) + args[2:]

        args = (args[0]._cpp_object,) + args[1:]

        # Case: Special sub domain 'inside' function provided as a
        # function
        if len(args) >= 3 and isinstance(args[2], types.FunctionType):
            # Note: using self below to avoid a problem where the user
            # function attached to AutoSubDomain get prematurely
            # destroyed. Maybe a pybind11 bug? Was the same with SWIG...
            self.sub_domain = AutoSubDomain(args[2])
            args = args[:2] + (self.sub_domain,) + args[3:]

        # FIXME: for clarity, can the user provided function case be
        # handled here too?
        # Create SubDomain object
        if isinstance(args[2], cpp.mesh.SubDomain):
            self.sub_domain = args[2]
            args = args[:2] + (self.sub_domain,) + args[3:]
        elif isinstance(args[2], str):
            self.sub_domain = CompiledSubDomain(args[2])
            args = args[:2] + (self.sub_domain,) + args[3:]
        elif isinstance(args[2], cpp.mesh.MeshFunctionSizet):
            pass
        else:
            raise RuntimeError("Invalid argument")

        # Add kwargs
        if isinstance(args[-1], str):
            method = args[-1]
        else:
            method = kwargs.pop("method", "topological")
            args += (method,)
        check_midpoint = kwargs.pop("check_midpoint", None)
        if check_midpoint is not None:
            args += (check_midpoint,)

        if (len(kwargs) > 0):
            raise RuntimeError("Invalid keyword arguments", kwargs)

        super().__init__(*args)
