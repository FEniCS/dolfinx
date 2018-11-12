# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import numba
from petsc4py import PETSc

from dolfin import cpp


def numba_eval(func):
    """Decorator for JITable eval method

    Parameters
    ----------
    func: Callable(None, (numpy.array, numpy.array, numpy.array))
        Evaluate method to JIT.

    Returns
    -------
    Address of JITed method.
    """
    scalar_type = numba.typeof(PETSc.ScalarType())
    c_signature = numba.types.void(
        numba.types.CPointer(scalar_type),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.int32),
        numba.types.intc, numba.types.intc, numba.types.intc,
        numba.types.intc)

    jitted_func = numba.jit(nopython=True)(func)

    @numba.cfunc(c_signature, nopython=True)
    def eval(values, x, cell_idx, num_points, value_size, gdim, num_cells):
        np_values = numba.carray(values, (num_points, value_size), dtype=scalar_type)
        np_x = numba.carray(x, (num_points, gdim), dtype=numba.types.double)
        np_cell_idx = numba.carray(cell_idx, (num_cells,), dtype=numba.types.int32)

        jitted_func(np_values, np_x, np_cell_idx)

    return eval.address


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
            1. (values_p, x_p, cells_p, num_points, value_size, gdim, num_cells)
                `values_p` is a pointer to a row major 2D
                C-style array of `PetscScalar`. The array has shape=(number of points, value size)
                and has to be filled with custom values in the function body,
                `x_p` is a pointer to a row major C-style 2D
                array of `double`. The array has shape=(number of points, geometrical dimension)
                and represents array of points in physical space at which the Expression
                is being evaluated,
                `cells_p` is a pointer to a 1D C-style array of `int`. It is an array
                of indices of cells where points are evaluated. Value -1 represents
                cell-independent eval function,
                `num_points`,
                `value_size`,
                `gdim` geometrical dimension of point where expression is evaluated,
                `num_cells`
        shape: tuple
            Value shape
        """
        self.shape = shape
        # Without this, undefined behaviour might happen due pybind docs
        super().__init__(shape)
        self.set_eval(eval_func)
