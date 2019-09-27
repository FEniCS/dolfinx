# -*- coding: utf-8 -*-
# Copyright (C) 2009-2019 Johan Hake, Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing
from functools import singledispatch

import numpy as np
from petsc4py import PETSc

import ufl
from dolfin import common, cpp, function, functionspace


class Function(ufl.Coefficient):
    """A finite element function that is represented by a function
    space (domain, element and dofmap) and a vector holding the
    degrees-of-freedom

    """

    def __init__(self,
                 V: functionspace.FunctionSpace,
                 x: typing.Optional[PETSc.Vec] = None,
                 name: typing.Optional[str] = None):
        """Initialize finite element Function."""

        # Create cpp Function
        if x is not None:
            self._cpp_object = cpp.function.Function(V._cpp_object, x)
        else:
            self._cpp_object = cpp.function.Function(V._cpp_object)

        # Initialize the ufl.FunctionSpace
        super().__init__(V.ufl_function_space(), count=self._cpp_object.id)

        # Set name
        if name is None:
            self.name = "f_{}".format(self.count())
        else:
            self.name = name

        # Store DOLFIN FunctionSpace object
        self._V = V

    @property
    def function_space(self) -> function.functionspace.FunctionSpace:
        """Return the FunctionSpace"""
        return self._V

    @property
    def value_rank(self) -> int:
        return self._cpp_object.value_rank

    def value_dimension(self, i) -> int:
        return self._cpp_object.value_dimension(i)

    def value_shape(self):
        return self._cpp_object.value_shape

    def ufl_evaluate(self, x, component, derivatives):
        """Function used by ufl to evaluate the Expression"""
        # FIXME: same as dolfin.expression.Expression version. Find way
        # to re-use.
        assert derivatives == ()  # TODO: Handle derivatives

        if component:
            shape = self.ufl_shape
            assert len(shape) == len(component)
            value_size = ufl.product(shape)
            index = ufl.utils.indexflattening.flatten_multiindex(
                component, ufl.utils.indexflattening.shape_to_strides(shape))
            values = np.zeros(value_size)
            # FIXME: use a function with a return value
            self(*x, values=values)
            return values[index]
        else:
            # Scalar evaluation
            return self(*x)

    def eval(self, x: np.ndarray, cells: np.ndarray, u=None) -> np.ndarray:
        """Evaluate Function at points x, where x has shape (num_points, gdim)"""

        # Make sure input coordinates are a NumPy array
        x = np.asarray(x, dtype=np.float)
        assert x.ndim < 2
        num_points = x.shape[0] if x.ndim > 1 else 1
        x = np.reshape(x, (num_points, -1))
        if x.shape[1] != self.geometric_dimension():
            raise ValueError("Wrong geometric dimension for coordinate(s).")

        # Allocate memory for return value if not provided
        if u is None:
            value_size = ufl.product(self.ufl_element().value_shape())
            if common.has_petsc_complex:
                u = np.empty((num_points, value_size), dtype=np.complex128)
            else:
                u = np.empty((num_points, value_size))

        self._cpp_object.eval(x, [cells], u)
        if num_points == 1:
            u = np.reshape(u, (-1, ))
        return u

    def interpolate(self, u) -> None:
        """Interpolate an expression"""

        @singledispatch
        def _interpolate(u):
            try:
                self._cpp_object.interpolate(u)
            except TypeError:
                self._cpp_object.interpolate(u._cpp_object)

        @_interpolate.register(int)
        def _(u, verbose=False):
            self._cpp_object.interpolate_ptr(u)

        _interpolate(u)

    def compute_point_values(self):
        return self._cpp_object.compute_point_values()

    def copy(self):
        """Return a copy of the Function. The FunctionSpace is shared and the
        degree-of-freedom vector is copied.

        """
        return function.Function(self.function_space(),
                                 self._cpp_object.vector.copy())

    @property
    def vector(self):
        """Return the vector holding Function degrees-of-freedom."""
        return self._cpp_object.vector

    @property
    def name(self) -> str:
        """Name of the Function."""
        return self._cpp_object.name

    @name.setter
    def name(self, name):
        self._cpp_object.name = name

    @property
    def id(self) -> int:
        """Return object id index."""
        return self._cpp_object.id

    def __str__(self):
        """Return a pretty print representation of it self."""
        return self.name

    def sub(self, i: int):
        """Return a sub function.

        The sub functions are numbered from i = 0..N-1, where N is the
        total number of sub spaces.

        """
        return Function(
            self._V.sub(i), self.vector, name="{}-{}".format(str(self), i))

    def split(self):
        """Extract any sub functions.

        A sub function can be extracted from a discrete function that
        is in a mixed, vector, or tensor FunctionSpace. The sub
        function resides in the subspace of the mixed space.

        """
        num_sub_spaces = self.function_space.num_sub_spaces()
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        return tuple(self.sub(i) for i in range(num_sub_spaces))

    def collapse(self):
        u_collapsed = self._cpp_object.collapse()
        V_collapsed = functionspace.FunctionSpace(None, self.ufl_element(),
                                                  u_collapsed.function_space)
        return Function(V_collapsed, u_collapsed.vector)


# # TODO: Update this message to clarify dolfin.FunctionSpace vs
# # ufl.FunctionSpace
# _ufl_dolfin_difference_message = """\ When constructing an Argument, TestFunction or TrialFunction, you
# must to provide a FunctionSpace and not a FiniteElement.  The
# FiniteElement class provided by ufl only represents an abstract finite
# element space and is only used in standalone .ufl files, while the
# FunctionSpace provides a full discrete function space over a given
# mesh and should be used in dolfin programs in Python.  """

class Argument(ufl.Argument):
    """Representation of an argument to a form"""

    def __init__(self, V: functionspace.FunctionSpace, number: int, part: int = None):
        """Create a UFL/DOLFIN Argument"""
        ufl.Argument.__init__(self, V.ufl_function_space(), number, part)
        self._V = V

    def function_space(self):
        """Return the FunctionSpace"""
        return self._V

    def __eq__(self, other: 'Argument'):
        """Extending UFL __eq__ here to distinguish test and trial functions
        in different function spaces with same ufl element.

        """
        return (isinstance(other, Argument)
                and self.number() == other.number()
                and self.part() == other.part() and self._V == other._V)

    def __hash__(self):
        return ufl.Argument.__hash__(self)


def TestFunction(V: functionspace.FunctionSpace, part: int = None):
    """Create a test function argument to a form"""
    return Argument(V, 0, part)


def TrialFunction(V: functionspace.FunctionSpace, part: int = None):
    """UFL value: Create a trial function argument to a form."""
    return Argument(V, 1, part)


def TestFunctions(V: functionspace.FunctionSpace):
    """Create a TestFunction in a mixed space, and return a
    tuple with the function components corresponding to the
    subelements.

    """
    return ufl.split(TestFunction(V))


def TrialFunctions(V: functionspace.FunctionSpace):
    """Create a TrialFunction in a mixed space, and return a
    tuple with the function components corresponding to the
    subelements.

    """
    return ufl.split(TrialFunction(V))
