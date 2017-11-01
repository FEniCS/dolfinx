# -*- coding: utf-8 -*-
"""This module handles the Function class in Python.

"""
# Copyright (C) 2009-2014 Johan Hake
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import ufl
from ufl.classes import ComponentTensor, Sum, Product, Division
from ufl.utils.indexflattening import shape_to_strides, flatten_multiindex
import dolfin.cpp as cpp
import dolfin.la as la
from dolfin.function.functionspace import FunctionSpace
from dolfin.function.expression import Expression
from dolfin.function.constant import Constant


def _assign_error():
    raise RuntimeError("Expected only linear combinations of Functions in the same FunctionSpaces")


def _check_mul_and_division(e, linear_comb, scalar_weight=1.0, multi_index=None):
    """
    Utility func for checking division and multiplication of a Function
    with scalars in linear combinations of Functions
    """
    from ufl.constantvalue import ScalarValue
    from ufl.classes import ComponentTensor, MultiIndex, Indexed
    from ufl.algebra import Division, Product, Sum
    # ops = e.ufl_operands

    # FIXME: What should be checked!?
    # martinal: This code has never done anything sensible,
    #   but I don't know what it was supposed to do so I can't fix it.
    # same_multi_index = lambda x, y: (x.ufl_free_indices == y.ufl_free_indices \
    #                        and x.ufl_index_dimensions == y.ufl_index_dimensions)

    assert isinstance(scalar_weight, float)

    # Split passed expression into scalar and expr
    if isinstance(e, Product):
        for i, op in enumerate(e.ufl_operands):
            if isinstance(op, ScalarValue) or \
               (isinstance(op, Constant) and op.value_size() == 1):
                scalar = op
                expr = e.ufl_operands[1 - i]
                break
        else:
            _assign_error()

        scalar_weight *= float(scalar)
    elif isinstance(e, Division):
        expr, scalar = e.ufl_operands
        if not (isinstance(scalar, ScalarValue) or
                isinstance(scalar, Constant) and scalar.value_rank() == 1):
            _assign_error()
        scalar_weight /= float(scalar)
    else:
        _assign_error()

    # If a CoefficientTensor is passed we expect the expr to be either
    # a Function or another ComponentTensor, where the latter wil
    # result in a recursive call
    if multi_index is not None:
        assert isinstance(multi_index, MultiIndex)
        assert isinstance(expr, Indexed)

        # Unpack Indexed and check equality with passed multi_index
        expr, multi_index2 = expr.ufl_operands
        assert isinstance(multi_index2, MultiIndex)
        # if not same_multi_index(multi_index, multi_index2):
        #    _assign_error()

    if isinstance(expr, Function):
        linear_comb.append((expr, scalar_weight))

    elif isinstance(expr, (ComponentTensor, Product, Division, Sum)):
        # If componentTensor we need to unpack the MultiIndices
        if isinstance(expr, ComponentTensor):
            expr, multi_index = expr.ufl_operands
            # if not same_multi_index(multi_index, multi_index2):
            #    _error()

        if isinstance(expr, (Product, Division)):
            linear_comb = _check_mul_and_division(expr, linear_comb, scalar_weight, multi_index)
        elif isinstance(expr, Sum):
            linear_comb = _check_and_extract_functions(expr, linear_comb, scalar_weight, multi_index)
        else:
            _assign_error()
    else:
        _assign_error()

    return linear_comb


def _check_and_extract_functions(e, linear_comb=None, scalar_weight=1.0,
                                 multi_index=None):
    """
    Utility func for extracting Functions and scalars in linear
    combinations of Functions
    """
    from ufl.classes import ComponentTensor, Sum, Product, Division
    linear_comb = linear_comb or []

    # First check u
    if isinstance(e, Function):
        linear_comb.append((e, scalar_weight))
        return linear_comb

    # Second check a*u*b, u/a/b, a*u/b where a and b are scalars
    elif isinstance(e, (Product, Division)):
        linear_comb = _check_mul_and_division(e, linear_comb, scalar_weight, multi_index)
        return linear_comb

    # Third check a*u*b, u/a/b, a*u/b where a and b are scalars and u
    # is a Tensor
    elif isinstance(e, ComponentTensor):
        e, multi_index = e.ufl_operands
        linear_comb = _check_mul_and_division(e, linear_comb, scalar_weight, multi_index)
        return linear_comb

    # If not Product or Division we expect Sum
    elif isinstance(e, Sum):
        for op in e.ufl_operands:
            linear_comb = _check_and_extract_functions(op, linear_comb,
                                                       scalar_weight, multi_index)

    else:
        _assign_error()

    return linear_comb


def _check_and_contract_linear_comb(expr, self, multi_index):
    """
    Utility func for checking and contracting linear combinations of
    Functions
    """
    linear_comb = _check_and_extract_functions(expr, multi_index=multi_index)
    funcs = []
    weights = []
    funcspace = None
    for func, weight in linear_comb:
        funcspace = funcspace or func.function_space()
        if func not in funcspace:
            _assign_error()
        try:
            # Check if the exact same Function is already present
            ind = funcs.index(func)
            weights[ind] += weight
        except Exception:
            funcs.append(func)
            weights.append(weight)

    # Check that rhs does not include self
    for ind, func in enumerate(funcs):
        if func == self:
            # If so make a copy
            funcs[ind] = self.copy(deepcopy=True)
            break

    return list(zip(funcs, weights))


class Function(ufl.Coefficient):

    def __init__(self, *args, **kwargs):
        """Initialize Function."""

        if isinstance(args[0], Function):
            other = args[0]
            if len(args) == 1:
                # Copy constructor used to be here
                raise RuntimeError("Use 'Function.copy(deepcopy=True)' for copying.")
            elif len(args) == 2:
                i = args[1]
                if not isinstance(i, int):
                    raise TypeError("Invalid subfunction number %s" % (i,))
                num_sub_spaces = other.function_space().num_sub_spaces()
                if num_sub_spaces == 1:
                    raise RuntimeError("No subfunctions to extract")
                if not i < num_sub_spaces:
                    raise RuntimeError("Can only extract subfunctions "
                                       "with i = 0..%d" % num_sub_spaces)
                self._cpp_object = cpp.function.Function(other._cpp_object, i)
                ufl.Coefficient.__init__(self, self.function_space().ufl_function_space(),
                                         count=self._cpp_object.id())
            else:
                raise TypeError("expected one or two arguments when "
                                "instantiating from another Function")
        elif isinstance(args[0], cpp.function.Function):
            raise RuntimeError("Construction from a cpp function not implemented yet")
        elif isinstance(args[0], FunctionSpace):
            V = args[0]

            # If initialising from a FunctionSpace
            if len(args) == 1:
                # If passing only the FunctionSpace
                self._cpp_object = cpp.function.Function(V._cpp_object)
            elif len(args) == 2:
                if isinstance(args[1], cpp.la.GenericVector):
                    self._cpp_object = cpp.function.Function(V._cpp_object, args[1])
                elif isinstance(args[1], cpp.function.Function):
                    self._cpp_object = args[1]
                elif isinstance(args[1], str):
                    # Read from xml filename in string
                    self._cpp_object = cpp.function.Function(V._cpp_object, args[1])
                else:
                    raise RuntimeError("Don't know what to do with ", type(args[1]))
            else:
                raise RuntimeError("Don't know what to do yet")

            # Initialize the ufl.FunctionSpace
            ufl.Coefficient.__init__(self, V.ufl_function_space(), count=self._cpp_object.id())

        else:
            raise TypeError("Expected a FunctionSpace or a Function as argument 1")

        # Set name as given or automatic
        name = kwargs.get("name") or "f_%d" % self.count()
        self.rename(name, "a Function")

    def function_space(self):
        "Return the FunctionSpace"
        return FunctionSpace(self._cpp_object.function_space())

    def value_rank(self):
        return self._cpp_object.value_rank()

    def value_dimension(self, i):
        return self._cpp_object.value_dimension(i)

    def value_shape(self):
        return self._cpp_object.value_shape

    def ufl_evaluate(self, x, component, derivatives):
        """Function used by ufl to evaluate the Expression"""
        # FIXME: same as dolfin.expression.Expression version. Find
        # way to re-use.
        assert derivatives == ()   # TODO: Handle derivatives

        if component:
            shape = self.ufl_shape
            assert len(shape) == len(component)
            value_size = ufl.product(shape)
            index = flatten_multiindex(component, shape_to_strides(shape))
            values = np.zeros(value_size)
            # FIXME: use a function with a return value
            self(*x, values=values)
            return values[index]
        else:
            # Scalar evaluation
            return self(*x)

    def __call__(self, *args, **kwargs):
        # GNW: This function is copied from the old DOLFIN Python
        # code. It is far too complicated. There is no need to provide
        # so many ways of doing the same thing.
        #
        # Deprecate as many options as possible, and maybe share with
        # dolfin.expression.Expresssion.

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
        value_size = ufl.product(self.ufl_element().value_shape())

        # If values (return argument) is passed, check the type and length
        values = kwargs.get("values", None)
        if values is not None:
            if not isinstance(values, np.ndarray):
                raise TypeError("expected a NumPy array for 'values'")
            if len(values) != value_size or \
               not np.issubdtype(values.dtype, 'd'):
                raise TypeError("expected a double NumPy array of length"
                                " %d for return values." % value_size)
            values_provided = True
        else:
            values_provided = False
            values = np.zeros(value_size, dtype='d')

        # Get the geometric dimension we live in
        dim = self.ufl_domain().geometric_dimension()

        # Assume all args are x argument
        x = args

        # If only one x argument has been provided, unpack it if it's
        # an iterable
        if len(x) == 1:
            if isinstance(x[0], cpp.geometry.Point):
                x = [x[0][i] for i in range(dim)]
            elif hasattr(x[0], '__iter__'):
                x = x[0]

        # Convert it to an 1D numpy array
        try:
            x = np.fromiter(x, 'd')
        except (TypeError, ValueError, AssertionError):
            raise TypeError("expected scalar arguments for the coordinates")

        if len(x) == 0:
            raise TypeError("coordinate argument too short")

        if len(x) != dim:
            raise TypeError("expected the geometry argument to be of "
                            "length %d" % dim)

        # The actual evaluation
        self._cpp_object.eval(values, x)

        # If scalar return statement, return scalar value.
        if value_size == 1 and not values_provided:
            return values[0]

        return values

    # def _assign(self, u):
    #    if isinstance(u, cpp.function.FunctionAXPY):
    #        self._cpp_object._assign(u)

    def eval_cell(self, u, x, cell):
        return self._cpp_object.eval(u, x, cell)

    def eval(self, u, x):
        return self._cpp_object.eval(u, x)

    def extrapolate(self, u):
        if isinstance(u, ufl.Coefficient):
            self._cpp_object.extrapolate(u._cpp_object)
        else:
            self._cpp_object.extrapolate(u)

    def interpolate(self, u):
        if isinstance(u, ufl.Coefficient):
            self._cpp_object.interpolate(u._cpp_object)
        else:
            self._cpp_object.interpolate(u)

    def compute_vertex_values(self, mesh=None):
        if mesh is not None:
            return self._cpp_object.compute_vertex_values(mesh)
        else:
            return self._cpp_object.compute_vertex_values()

    def set_allow_extrapolation(self, value):
        self._cpp_object.set_allow_extrapolation(value)

    def get_allow_extrapolation(self):
        return self._cpp_object.get_allow_extrapolation()

    def copy(self, deepcopy=False):
        # See https://bitbucket.org/fenics-project/dolfin/issues/702
        if deepcopy:
            return Function(self.function_space(), self._cpp_object.vector().copy())
        return Function(self.function_space(), self._cpp_object.vector())

    def vector(self):
        return self._cpp_object.vector()

    def assign(self, rhs):
        """
        Assign either a Function or linear combination of Functions.

        *Arguments*
            rhs (_Function_)
                A Function or a linear combination of Functions. If a linear
                combination is passed all Functions need to be in the same
                FunctionSpaces.
        """

        if isinstance(rhs, (cpp.function.Function, cpp.function.Expression, cpp.function.FunctionAXPY)):
            # Avoid self assignment
            if self == rhs:
                return
            self._cpp_object._assign(rhs)
        elif isinstance(rhs, (Constant, Function, Expression)):
            # Avoid self assignment
            if self == rhs:
                return
            self._cpp_object._assign(rhs._cpp_object)
        elif isinstance(rhs, (Sum, Product, Division, ComponentTensor)):
            if isinstance(rhs, ComponentTensor):
                rhs, multi_index = rhs.ufl_operands
            else:
                multi_index = None
            linear_comb = _check_and_contract_linear_comb(rhs, self, multi_index)
            assert(linear_comb)

            # If the assigned Function lives in a different FunctionSpace
            # we cannot operate on this function directly
            same_func_space = linear_comb[0][0] in self.function_space()
            func, weight = linear_comb.pop()

            # Assign values from first func
            if not same_func_space:
                self._cpp_object._assign(func._cpp_object)
                vector = self.vector()
            else:
                vector = self.vector()
                vector[:] = func.vector()

            # If first weight is not 1 scale
            if weight != 1.0:
                vector *= weight

            # AXPY the other functions
            for func, weight in linear_comb:
                if weight == 0.0:
                    continue
                vector.axpy(weight, func.vector())

        else:
            raise RuntimeError("Expected a Function or linear combinations of Functions in the same FunctionSpace")

    def __float__(self):
        # FIXME: this could be made simple on the C++ (in particular,
        # with dolfin::Scalar)
        if self.ufl_shape != ():
            raise RuntimeError("Cannot convert nonscalar function to float.")
        elm = self.ufl_element()
        if elm.family() != "Real":
            raise RuntimeError("Cannot convert spatially varying function to float.")
        # FIXME: This could be much simpler be exploiting that the
        # vector is ghosted
        # Gather value directly from vector in a parallel safe way
        vec = self.vector()
        indices = np.zeros(1, dtype=la.la_index_dtype())
        values = vec.gather(indices)
        return float(values[0])

    def name(self):
        return self._cpp_object.name()

    def rename(self, name, s):
        self._cpp_object.rename(name, s)

    def id(self):
        return self._cpp_object.id()

    def __str__(self):
        """Return a pretty print representation of it self."""
        return self.name()

    def root_node(self):
        u = self._cpp_object.root_node()
        return Function(FunctionSpace(u.function_space()), u.vector())

    def leaf_node(self):
        u = self._cpp_object.leaf_node()
        return Function(FunctionSpace(u.function_space()), u.vector())

    def cpp_object(self):
        return self._cpp_object

    def sub(self, i, deepcopy=False):
        """
        Return a sub function.

        The sub functions are numbered from i = 0..N-1, where N is the
        total number of sub spaces.

        *Arguments*
            i : int
                The number of the sub function

        """
        if not isinstance(i, int):
            raise TypeError("expects an 'int' as first argument")
        num_sub_spaces = self.function_space().num_sub_spaces()
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        if not i < num_sub_spaces:
            raise RuntimeError("Can only extract subfunctions with i = 0..%d"
                               % num_sub_spaces)

        # Create and instantiate the Function
        if deepcopy:
            return Function(self.function_space().sub(i),
                            self.cpp_object().sub(i),
                            name='%s-%d' % (str(self), i))
        else:
            return Function(self, i, name='%s-%d' % (str(self), i))

    def split(self, deepcopy=False):
        """Extract any sub functions.

        A sub function can be extracted from a discrete function that
        is in a mixed, vector, or tensor FunctionSpace. The sub
        function resides in the subspace of the mixed space.

        *Arguments*
            deepcopy
                Copy sub function vector instead of sharing

        """

        num_sub_spaces = self.function_space().num_sub_spaces()
        if num_sub_spaces == 1:
            raise RuntimeError("No subfunctions to extract")
        return tuple(self.sub(i, deepcopy) for i in range(num_sub_spaces))
