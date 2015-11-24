# -*- coding: utf-8 -*-
""" Module to extract scalar expression factors for each test function component."""
# Copyright (C) 2014 Martin Sandve Alnæs
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
#
# Modified by Patrick Farrell, 2014
#
# First added:  2014-11-28
# Last changed: 2014-11-28

from ufl import *
from ufl.corealg.multifunction import MultiFunction
from ufl.corealg.map_dag import map_expr_dag
from ufl.classes import Argument, MultiIndex, Indexed, FixedIndex
from ufl.log import error as ufl_error

__all__ = ["extract_tested_expressions"]

# TODO: "factorization" is not the right term

class ScalarFactorizer(MultiFunction):
    def __init__(self):
        MultiFunction.__init__(self)
        self._one = as_ufl(1.0)
        self._arg = None

    def _argument(self, component, expr):
        if self._arg is None:
            self._arg = expr
        elif self._arg is not expr:
            ufl_error("Expecting only one Argument in this algorithm implementation.")
        return { component: self._one }

    def argument(self, e):
        if e.ufl_shape != ():
            ufl_error("Nonscalar argument {}.".format(str(e)))
        return self._argument(0, e)

    def terminal(self, t):
        if t.ufl_shape != ():
            ufl_error("Nonscalar terminal {}.".format(str(t)))
        return t

    def indexed(self, e):
        if e.ufl_shape != ():
            ufl_error("Nonscalar indexed {}.".format(str(e)))

        v, i = e.ufl_operands

        if v._ufl_typecode_ == Argument._ufl_typecode_:
            if len(i) != 1 or not isinstance(i[0], FixedIndex):
                ufl_error("Expecting only vector valued Arguments in this algorithm implementation.")
            return self._argument(int(i[0]), v)

        return e

    def operator(self, e, *ops):
        if e.ufl_shape != ():
            ufl_error("Nonscalar operator {}.".format(str(e)))
        if any(isinstance(op, dict) for op in ops):
            ufl_error("Handler for operator {} assumes no Arguments among operands.".format(e._ufl_handler_name_))
        return e

    def sum(self, e, a, b):
        n_a = len(a) if isinstance(a, dict) else 0
        n_b = len(b) if isinstance(b, dict) else 0
        if n_a > 0 and n_b > 0:
            c = {}
            keys = set(a.keys()) | set(b.keys())
            for k in keys:
                av = a.get(k)
                bv = b.get(k)
                if av is None:
                    # Case: Only b contains a term with test function component k
                    c[k] = bv
                elif bv is None:
                    # Case: Only a contains a term with test function component k
                    c[k] = av
                else:
                    # Case: Both a and b contains a term with test function component k
                    c[k] = av + bv
            return c
        elif n_a or n_b:
            ufl_error("Cannot add Argument-dependent expression with non-Argument-dependent expression.")
        else:
            return e

    def product(self, e, a, b):
        a_is_dict = isinstance(a, dict)
        b_is_dict = isinstance(b, dict)
        if a_is_dict and b_is_dict:
            ufl_error("Expecting only one Argument in this algorithm. Products of Arguments are not allowed.")
        elif a_is_dict:
            c = {}
            for k,v in a.items():
                c[k] = v*b
            return c
        elif b_is_dict:
            c = {}
            for k,v in b.items():
                c[k] = v*a
            return c
        else:
            return e

    def division(self, e, a, b):
        if isinstance(b, dict):
            ufl_error("Cannot divide by Arguments.")
        if isinstance(a, dict):
            c = {}
            for k,v in a.items():
                c[k] = v / b
            return c
        else:
            return e

def extract_tested_expressions(expr):
    """Extract scalar expression factors for each test function component.

    This is for internal usage and has several undocumented limitations.
    """
    func = ScalarFactorizer()
    e = map_expr_dag(func, expr, compress=False)
    return e, func._arg
