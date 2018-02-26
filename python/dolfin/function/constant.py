# Copyright (C) 2008-2015 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""Create a constant-valued function with given value."""

import ufl
import dolfin.cpp as cpp
import numpy


class Constant(ufl.Coefficient):

    def __init__(self, value, cell=None, name=None):
        """
        Create constant-valued function with given value.

        *Arguments*
            value
                The value may be either a single scalar value, or a
                tuple/list of values for vector-valued functions, or
                nested lists or a numpy array for tensor-valued
                functions.
            cell
                Optional argument. A :py:class:`Cell
                <ufl.Cell>` which defines the geometrical
                dimensions the Constant is defined for.
            name
                Optional argument. A str which overrules the default
                name of the Constant.

        The data type Constant represents a constant value that is
        unknown at compile-time. Its values can thus be changed
        without requiring re-generation and re-compilation of C++
        code.

        *Examples of usage*

            .. code-block:: python

                p = Constant(pi/4)              # scalar
                C = Constant((0.0, -1.0, 0.0))  # constant vector

        """

        # TODO: Either take mesh instead of cell, or drop cell and let
        # grad(c) be undefined.
        if cell is not None:
            cell = ufl.as_cell(cell)
        ufl_domain = None

        array = numpy.array(value)
        rank = len(array.shape)
        floats = list(map(float, array.flat))

        # Create UFL element and initialize constant
        if rank == 0:
            ufl_element = ufl.FiniteElement("Real", cell, 0)
            self._cpp_object = cpp.function.Constant(floats[0])
        elif rank == 1:
            ufl_element = ufl.VectorElement("Real", cell, 0, dim=len(floats))
            self._cpp_object = cpp.function.Constant(floats)
        else:
            ufl_element = ufl.TensorElement("Real", cell, 0, shape=array.shape)
            self._cpp_object = cpp.function.Constant(list(array.shape), floats)

        # Initialize base classes
        ufl_function_space = ufl.FunctionSpace(ufl_domain, ufl_element)
        ufl.Coefficient.__init__(self, ufl_function_space, count=self.id())

        # Set name as given or automatic
        name = name or "f_%d" % self.count()
        self.rename(name, "a Constant")

    def assign(self, x):
        if isinstance(x, Constant):
            x = x._cpp_object
        return self._cpp_object.assign(x)

    def cell(self):
        return self.ufl_element().cell()

    def compute_vertex_values(self, mesh):
        return self._cpp_object.compute_vertex_values(mesh)

    def values(self):
        return self._cpp_object.values()

    def value_size(self):
        return self._cpp_object.value_size()

    def id(self):
        return self._cpp_object.id()

    def name(self):
        return self._cpp_object.name()

    def rename(self, name, s):
        self._cpp_object.rename(name, s)

    def __float__(self):
        # Overriding UFL operator in this particular case.
        if self.ufl_shape:
            raise TypeError("Cannot convert nonscalar constant to float.")
        return float(self._cpp_object)

    def str(self, verbose):
        return self._cpp_object.str(verbose)

    def __str__(self):
        return self.name()

    def cpp_object(self):
        return self._cpp_object
