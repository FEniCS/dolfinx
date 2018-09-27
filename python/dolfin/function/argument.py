# -*- coding: utf-8 -*-
# Copyright (C) 2009-2018 Johan Hake, Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Interface for UFL and DOLFN form arguments"""

import typing

import ufl
from dolfin import function

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

    def __init__(self,
                 V: function.FunctionSpace,
                 number: int,
                 part: int = None):
        """Create a UFL/DOLFIN Argument"""
        ufl.Argument.__init__(self, V.ufl_function_space(), number, part)
        self._V = V

    def function_space(self):
        """Return the FunctionSpace"""
        return self._V

    def __eq__(self, other):
        """Extending UFL __eq__ here to distinguish test and trial functions
        in different function spaces with same ufl element.

        """
        return (isinstance(other, Argument)
                and self.number() == other.number()
                and self.part() == other.part() and self._V == other._V)

    def __hash__(self):
        return ufl.Argument.__hash__(self)


def TestFunction(V: function.FunctionSpace, part: int = None):
    """Create a test function argument to a form"""
    return Argument(V, 0, part)


def TrialFunction(V: function.FunctionSpace, part: int = None):
    """UFL value: Create a trial function argument to a form."""
    return Argument(V, 1, part)


def Arguments(V: function.FunctionSpace, number: int):
    """Create an Argument in a mixed space, and return a tuple with the
    function components corresponding to the subelements.

    """
    return ufl.split(Argument(V, number))


def TestFunctions(V: function.FunctionSpace):
    """Create a TestFunction in a mixed space, and return a
    tuple with the function components corresponding to the
    subelements.

    """
    return ufl.split(TestFunction(V))


def TrialFunctions(V: function.FunctionSpace):
    """Create a TrialFunction in a mixed space, and return a
    tuple with the function components corresponding to the
    subelements.

    """
    return ufl.split(TrialFunction(V))
