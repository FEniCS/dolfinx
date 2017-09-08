# -*- coding: utf-8 -*-
"""FIXME: Add description"""

# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# Distributed under the terms of the GNU Lesser Public License (LGPL),
# either version 3 of the License, or (at your option) any later
# version.


__all__ = ["UserExpression"]

import hashlib
from functools import reduce
import types
import numpy
import ufl
from ufl import product
from ufl.utils.indexflattening import (flatten_multiindex,
                                       shape_to_strides)
import dolfin.cpp as cpp
import dolfin.function.jit as jit


def _select_element(family, cell, degree, value_shape):
    """Select finite element type for cases where user has not provided a
    complete ufl.FiniteElement

    """
    if family is None:
        if degree == 0:
            family = "Discontinuous Lagrange"
        else:
            family = "Lagrange"

    if len(value_shape) == 0:
        element = ufl.FiniteElement(family, cell, degree)
    elif len(value_shape) == 1:
        element = ufl.VectorElement(family, cell, degree, dim=value_shape[0])
    else:
        element = ufl.TensorElement(family, cell, degree, shape=value_shape)

    return element


class _InterfaceExpression(cpp.function.Expression):
    """A DOLFIN C++ Expression to which user eval functions are attached.

    """

    def __init__(self, user_expression, value_shape):
        self.user_expression = user_expression

        # Wrap eval functions
        def wrapped_eval(self, values, x):
            self.user_expression.eval(values, x)
        def wrapped_eval_cell(self, values, x, cell):
            self.user_expression.eval_cell(values, x, cell)

        # Attach user-provided Python eval functions (if they exist in
        # the user expression class) to the C++ class
        if hasattr(user_expression, 'eval'):
            self.eval = types.MethodType(wrapped_eval, self)
        elif hasattr(user_expression, 'eval_cell'):
            self.eval_cell = types.MethodType(wrapped_eval_cell, self)

        # Create C++ Expression object
        cpp.function.Expression.__init__(self, value_shape)


class BaseExpression(ufl.Coefficient):
    def __init__(self, cell=None, element=None, domain=None, name=None,
                 label=None):

        # Some messy cell/domain handling for compatibility, will be
        # straightened out later
        if domain is None:
            ufl_domain = None
        else:
            if isinstance(domain, ufl.domain.AbstractDomain):
                ufl_domain = domain
            else:
                # Probably getting a Mesh here, from existing dolfin
                # tests. Will be the same later anyway.
                ufl_domain = domain.ufl_domain()

            if cell is None:
                cell = ufl_domain.ufl_cell()

        # Initialise base class
        ufl_function_space = ufl.FunctionSpace(ufl_domain, element)
        ufl.Coefficient.__init__(self, ufl_function_space, count=self.id())

        name = name or "f_" + str(ufl.Coefficient.count(self))
        label = label or "User defined expression"
        self._cpp_object.rename(name, label)

    #def __call__(self, x):
    #    return self._cpp_object(x)
    def __call__(self, *args, **kwargs):
        # GNW: This function is copied from the old DOLFIN Python
        # code. It is far too complicated. There is no need to provide
        # so many ways of doing the same thing.
        #
        # Deprecate as many options as possible

        if len(args) == 0:
            raise TypeError("expected at least 1 argument")

        # Test for ufl restriction
        if len(args) == 1 and isinstance(args[0], str):
            if args[0] in ('+', '-'):
                return ufl.Coefficient.__call__(self, *args)

        # Test for ufl mapping
        if len(args) == 2 and isinstance(args[1], dict) and self in args[1]:
            return ufl.Coefficient.__call__(self, *args)

        # Some help variables
        value_size = product(self.ufl_element().value_shape())

        # If values (return argument) is passed, check the type and length
        values = kwargs.get("values", None)
        if values is not None:
            if not isinstance(values, numpy.ndarray):
                raise TypeError("expected a NumPy array for 'values'")
            if len(values) != value_size or not numpy.issubdtype(values.dtype, 'd'):
                raise TypeError("expected a double NumPy array of length"\
                                " %d for return values." % value_size)
            values_provided = True
        else:
            values_provided = False
            values = numpy.zeros(value_size, dtype='d')

        # Get dim if element has any domains
        cell = self.ufl_element().cell()
        dim = None if cell is None else cell.geometric_dimension()

        # Assume all args are x argument
        x = args

        # If only one x argument has been provided, unpack it if it's
        # an iterable
        if len(x) == 1:
            if isinstance(x[0], cpp.geometry.Point):
                if dim is not None:
                    x = [x[0][i] for i in range(dim)]
                else:
                    x = [x[0][i] for i in range(3)]
            elif hasattr(x[0], '__iter__'):
                x = x[0]

        # Convert it to an 1D numpy array
        try:
            x = numpy.fromiter(x, 'd')
        except (TypeError, ValueError, AssertionError) as e:
            raise TypeError("expected scalar arguments for the coordinates")

        if len(x) == 0:
            raise TypeError("coordinate argument too short")

        if dim is None:
            # Disabled warning as it breaks py.test due to excessive
            # output, and that code that is warned about is still
            # officially supported. See
            # https://bitbucket.org/fenics-project/dolfin/issues/355/
            # warning("Evaluating an Expression without knowing the right geometric dimension, assuming %d is correct." % len(x))
            pass
        else:
            if len(x) != dim:
                raise TypeError("expected the geometry argument to be of "\
                                "length %d" % dim)

        # The actual evaluation
        self._cpp_object.eval(values, x)

        # If scalar return statement, return scalar value.
        if value_size == 1 and not values_provided:
            return values[0]

        return values

    def id(self):
        return self._cpp_object.id()

    def value_rank(self):
        return self._cpp_object.value_rank()

    def value_dimension(self, i):
        return self._cpp_object.value_dimension(i)

    def name(self):
        return self._cpp_object.name()

    def label(self):
        return self._cpp_object.label()

    def __str__(self):
        return self._cpp_object.name()

    def cpp_object(self):
        return self._cpp_object

    def compute_vertex_values(self, mesh):
        return self._cpp_object.compute_vertex_values(mesh)


class UserExpression(BaseExpression):
    """Base class for user-defined Python Expression classes, where the
    user overloads eval or eval_cell

    """

    def __init__(self, *args, **kwargs):

        # Extract data
        element = kwargs.pop("element", None)
        degree = kwargs.pop("degree", 2)
        cell = kwargs.pop("cell", None)
        domain = kwargs.pop("domain", None)
        name = kwargs.pop("name", None)
        label = kwargs.pop("label", None)
        mpi_comm = kwargs.pop("mpi_comm", None)
        if (len(kwargs) > 0):
            raise RuntimeError("Invalid keyword argument")

        # Deduce element type if not provided
        if element is None:
            if hasattr(self, "value_shape"):
                value_shape = self.value_shape()
            else:
                print("WARNING: user expression has not supplied value_shape method or an element. Assuming scalar element.")
                value_shape = ()

            element = _select_element(family=None, cell=cell, degree=degree,
                                      value_shape=value_shape)
        else:
            value_shape = element.value_shape()

        self._cpp_object = _InterfaceExpression(self, value_shape)
        BaseExpression.__init__(self, cell=cell, element=element, domain=domain,
                                name=name, label=label)


class Expression(BaseExpression):
    """JIT Expressions"""

    def __init__(self, cpp_code=None, *args, **kwargs):

        # Developer note: class attributes must be prefixed by "_",
        # otherwise problems occur due to __setattr__, which is used
        # to pass parameters through to JIT complied expression.

        # Remove arguments that are used in Expression creation
        element = kwargs.pop("element", None)
        degree = kwargs.pop("degree", None)
        cell = kwargs.pop("cell", None)
        domain = kwargs.pop("domain", None)
        name = kwargs.pop("name", None)
        label = kwargs.pop("label", None)
        mpi_comm = kwargs.pop("mpi_comm", None)

        # Save properties for checking later
        self._properties = kwargs
        for k in self._properties:
            if not isinstance(k, str):
                raise KeyError("Invalid key:", k)

        if cpp_code is not None:
            self._cpp_object = jit.compile_expression(cpp_code, self._properties)

        if element and degree:
            raise RuntimeError("Cannot specify an element and a degree for Expressions.")

        # Deduce element type if not provided
        if element is None:
            if degree is None:
                raise KeyError("Must supply element or degree")
            value_shape = tuple(self.value_dimension(i)
                                for i in range(self.value_rank()))
            if domain is not None and cell is None:
                cell = domain.ufl_cell()
            element = _select_element(family=None, cell=cell, degree=degree,
                                      value_shape=value_shape)

        # FIXME: The below is invasive and fragile. Fix multistage so
        #        the below is note required.
        # Store C++ code and user parameters because they are used by
        # the the multistage module.
        self._user_parameters = kwargs
        self._cppcode = cpp_code

        BaseExpression.__init__(self, cell=cell, element=element, domain=domain,
                                name=name, label=label)

    # FIXME: below comment seems obsolete?
    # This is added dynamically in the intialiser to allow checking of
    # eval in user classes.
    def __getattr__(self, name):
        "Pass attributes through to (JIT compiled) Expression object"
        if name in self._properties.keys():
            if isinstance(self._properties[name], (float, int)):
                return self._cpp_object.get_property(name)
            else:
                return self._cpp_object.get_generic_function(name)
        else:
            raise AttributeError

    def __setattr__(self, name, value):
        # FIXME: this messes up setting attributes
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif name in self._properties.keys():
            self._cpp_object.set_property(name, value)
