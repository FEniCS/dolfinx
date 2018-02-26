# Copyright (C) 2009-2017 Johan Hake, Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

"""This module handles the Function class in Python.

"""

import ufl
from .functionspace import FunctionSpace

__all__ = ["TestFunction", "TrialFunction", "Argument",
           "TestFunctions", "TrialFunctions"]

# --- Subclassing of ufl.{Basis, Trial, Test}Function ---

# TODO: Update this message to clarify dolfin.FunctionSpace vs
# ufl.FunctionSpace
_ufl_dolfin_difference_message = """\ When constructing an Argument, TestFunction or TrialFunction, you
must to provide a FunctionSpace and not a FiniteElement.  The
FiniteElement class provided by ufl only represents an abstract finite
element space and is only used in standalone .ufl files, while the
FunctionSpace provides a full discrete function space over a given
mesh and should be used in dolfin programs in Python.  """


class Argument(ufl.Argument):
    """UFL value: Representation of an argument to a form.

    This is the overloaded PyDOLFIN variant.
    """
    def __init__(self, V, number, part=None):

        # Check argument
        if not isinstance(V, FunctionSpace):
            if isinstance(V, (ufl.FiniteElementBase, ufl.FunctionSpace)):
                raise TypeError(_ufl_dolfin_difference_message)
            else:
                raise TypeError("Illegal argument for creation of Argument, not a FunctionSpace: " + str(V))
            raise TypeError("Illegal argument for creation of Argument, not a FunctionSpace: " + str(V))

        # Initialize UFL Argument
        ufl.Argument.__init__(self, V.ufl_function_space(), number, part)

        self._V = V

    def function_space(self):
        "Return the FunctionSpace"
        return self._V

    def __eq__(self, other):
        """Extending UFL __eq__ here to distinguish test and trial functions
        in different function spaces with same ufl element.

        """
        return (isinstance(other, Argument) and
                self.number() == other.number() and
                self.part() == other.part() and
                self._V == other._V)

    def __hash__(self):
        return ufl.Argument.__hash__(self)


def TestFunction(V, part=None):
    """UFL value: Create a test function argument to a form.

    This is the overloaded PyDOLFIN variant.
    """
    return Argument(V, 0, part)


def TrialFunction(V, part=None):
    """UFL value: Create a trial function argument to a form.

    This is the overloaded PyDOLFIN variant.

    """
    return Argument(V, 1, part)


# --- TestFunctions and TrialFunctions ---

def Arguments(V, number):
    """UFL value: Create an Argument in a mixed space, and return a
    tuple with the function components corresponding to the subelements.

    This is the overloaded PyDOLFIN variant.

    """
    return ufl.split(Argument(V, number))


def TestFunctions(V):
    """UFL value: Create a TestFunction in a mixed space, and return a
    tuple with the function components corresponding to the
    subelements.

    This is the overloaded PyDOLFIN variant.

    """
    return ufl.split(TestFunction(V))


def TrialFunctions(V):
    """UFL value: Create a TrialFunction in a mixed space, and return a
    tuple with the function components corresponding to the
    subelements.

    This is the overloaded PyDOLFIN variant.

    """
    return ufl.split(TrialFunction(V))
