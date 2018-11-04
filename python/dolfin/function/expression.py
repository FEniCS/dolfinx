# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import cpp


class Expression(cpp.function.Expression):

    def __init__(self, eval_func, shape=()):
        """Initialise Expression

        Initialises Expression from evaluate function and value shape.

        Parameters
        ---------
        eval_func:
            Function accepting the following sets of arguments:
            1. (values, x)
                `values` is NumPy array with shape=(number of points, value size)
                and has to be filled with custom values,
                `x` is NumPy array with shape=(number of points, geometrical dimension)
                and represents array of points in physical space at which the Expression
                is being evaluated.
            2. (values, x, cell)
                The same as previously, but with information about cell
                where the Expression is being evaluated.
        shape: list
            Value shape
        """
        # Without this, undefined behaviour might happen due pybind docs
        cpp.function.Expression.__init__(self, shape)
        self.shape = shape
        self.eval_func = eval_func

    def eval_cell(self, values, x, cell):
        self.eval_func(values, x, cell)

    def eval(self, values, x):
        self.eval_func(values, x)
