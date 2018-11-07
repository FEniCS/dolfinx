# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from typing import Callable

import numpy

from dolfin import cpp


class Expression(cpp.function.Expression):
    def __init__(self,
                 eval_func: int,
                 shape: tuple = ()):
        """Initialise Expression

        Initialises Expression from address of C function and value shape.

        Parameters
        ---------
        eval_func:
            Address of compiled C function.
            C function must accept the following sets of arguments:
            1. (values_p, x_p, cells_p)
                `values_p` is a pointer to a row major 2D
                C-style array of `PetscScalar`. The array has shape=(number of points, value size)
                and has to be filled with custom values in the function body,
                `x_p` is a pointer to a row major C-style 2D
                array of `double`. The array has shape=(number of points, geometrical dimension)
                and represents array of points in physical space at which the Expression
                is being evaluated,
                `cells_p` is a pointer to a 1D C-style array of `int`. It is an array
                of indices of cells where points are evaluated. Value -1 represents
                cell-independent eval function.
        shape: tuple
            Value shape
        """
        self.shape = shape
        # Without this, undefined behaviour might happen due pybind docs
        super().__init__(shape)
        self.set_eval(eval_func)
