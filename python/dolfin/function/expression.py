# -*- coding: utf-8 -*-
# Copyright (C) 2018 Michal Habera and Jack S. Hale
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import typing

import numba
import numba.ccallback
from petsc4py import PETSc

from dolfin import cpp


def numba_eval(*args,
               numba_jit_options: dict = {
                   "nopython": True,
                   "cache": True
               },
               numba_cfunc_options: dict = {
                   "nopython": True,
                   "cache": True
               }):
    """Decorator to create Numba JIT-compiled evaluate function.

    A decorator that takes an evaluation function ``f`` and returns the
    C address of the Numba JIT-ed method. The call signature of ``f``
    should be:

    f: Callable(None, (numpy.array, numpy.array, numpy.array))
        Python function accepting parameters: values, x, cell_index.

    For more information on ``numba_jit_options`` and ``numba_cfunc_options``
    read the Numba documentation.

    Parameters
    ----------
    numba_jit_options: dict, optional
        Options passed to ``numba.jit``. ``nopython`` must be ``True``.
    numba_cfunc_options: dict, optional
        Options passed to ``numba.cfunc``. ``nopython`` must be ``True``.

    Example
    -------
    >>> @function.expression.numba_eval
    >>> def expr(values, x, cell_idx):
    >>>    values[:, 0] = x[:, 0] + x[:, 1] + x[:, 2]
    >>>    values[:, 1] = x[:, 0] - x[:, 1] - x[:, 2]
    >>>    values[:, 2] = x[:, 0] + x[:, 1] + x[:, 2]
    >>> mesh = UnitCubeMesh(MPI.comm_world, 3, 3, 3)
    >>> W = VectorFunctionSpace(mesh, ('CG', 1))
    >>> e = Expression(expr, shape=(3,))
    >>> u = Function(W)
    >>> u.interpolate(e)
    """

    # Decomaker pattern see PEP 318
    def decorator(f: typing.Callable):
        scalar_type = numba.typeof(PETSc.ScalarType())
        c_signature = numba.types.void(
            numba.types.CPointer(scalar_type),
            numba.types.CPointer(numba.types.double),
            numba.types.CPointer(numba.types.int32), numba.types.intc,
            numba.types.intc, numba.types.intc, numba.types.intc)

        # Compile the user function
        f_jit = numba.jit(**numba_jit_options)(f)

        # Wrap the user function in a function with a C interface
        @numba.cfunc(c_signature, **numba_cfunc_options)
        def eval(values, x, cell_idx, num_points, value_size, gdim, num_cells):
            np_values = numba.carray(
                values, (num_points, value_size), dtype=scalar_type)
            np_x = numba.carray(
                x, (num_points, gdim), dtype=numba.types.double)
            np_cell_idx = numba.carray(
                cell_idx, (num_cells, ), dtype=numba.types.int32)

            f_jit(np_values, np_x, np_cell_idx)

        return eval

    if len(args) == 1 and callable(args[0]):
        # Allows decoration without arguments
        return decorator(args[0])
    else:
        return decorator


class Expression:
    def __init__(self,
                 f: typing.Union[numba.ccallback.CFunc, int],
                 shape: tuple = ()):
        """Initialise Expression

        Initialises Expression from Numba callback of compiled C function or
        integer address of C function and value shape.

        The majority of users should use this class in conjunction with the
        ``function.expression.numba_eval`` decorator that creates the Numba
        JIT-compiled evaluation functions.

        Parameters
        ---------
        f: numba.ccallback.CFunc, int
            The C function must accept the following arguments:
            ``(values_p, x_p, cells_p, num_points, value_size, gdim, num_cells)``
            1. ``values_p`` is a pointer to a row-major array of
               ``PetscScalar`` of shape ``(num_points, value_size)``.
               The function itself is responsible for filling ``values_p``
               with the desired Expression evaluations. ``values_p`` is not
               zeroed before being passed to the function.
            2. ``x_p`` is a pointer to a row-major array of ``double`` of shape
               ``(num_points, gdim)``. The array contains the coordinates
               of the points at which the expression function should be evaluated.
            3. ``cells_p`` is a pointer to an array of ``int`` of shape
               (num_cells).  It is an array of indices of cells where the points
               are evaluated. Value -1 represents a cell-independent evaluation.
               function,
            4. ``num_points``, ``int``, Number of points,
            5. ``value_size``, ``int``, Number of values,
            6. ``gdim``, ``int``, Geometric dimension of coordinates,
            7. ``num_cells``, ``int``, Number of cells.
        shape: tuple
            Value shape.
        """

        try:
            self._cpp_object = cpp.function.Expression(f.address, shape)
            # Hold reference to eval function to avoid premature garbage
            # collection
            self._f = f
        except AttributeError:
            self._cpp_object = cpp.function.Expression(f, shape)
