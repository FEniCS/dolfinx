# -*- coding: utf-8 -*-
# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Support for representing Dirichlet boundary conditions that are enforced
via modification of linear systems.

"""

import collections.abc
import types
import typing

import numpy as np

import ufl
from dolfinx import cpp as _cpp
from dolfinx.fem.function import Function, FunctionSpace

from petsc4py import PETSc


def locate_dofs_geometrical(V: typing.Iterable[typing.Union[_cpp.fem.FunctionSpace, FunctionSpace]],
                            marker: types.FunctionType):
    """Locate degrees-of-freedom geometrically using a marker function.

    Parameters
    ----------
    V
        Function space(s) in which to search for degree-of-freedom indices.

    marker
        A function that takes an array of points ``x`` with shape
        ``(gdim, num_points)`` and returns an array of booleans of length
        ``num_points``, evaluating to ``True`` for entities whose
        degree-of-freedom should be returned.

    Returns
    -------
    numpy.ndarray
        An array of degree-of-freedom indices (local to the process)
        for degrees-of-freedom whose coordinate evaluates to True for the
        marker function.

        If ``V`` is a list of two function spaces, then a 2-D array of
        shape (number of dofs, 2) is returned.

        Returned degree-of-freedom indices are unique and ordered by the
        first column.

    """

    if isinstance(V, collections.abc.Sequence):
        _V = []
        for space in V:
            try:
                _V.append(space._cpp_object)
            except AttributeError:
                _V.append(space)
        return _cpp.fem.locate_dofs_geometrical(_V, marker)
    else:
        try:
            return _cpp.fem.locate_dofs_geometrical(V, marker)
        except TypeError:
            return _cpp.fem.locate_dofs_geometrical(V._cpp_object, marker)


def locate_dofs_topological(V: typing.Iterable[typing.Union[_cpp.fem.FunctionSpace, FunctionSpace]],
                            entity_dim: int,
                            entities: typing.List[int],
                            remote: bool = True):
    """Locate degrees-of-freedom belonging to mesh entities topologically.

    Parameters
    ----------
    V
        Function space(s) in which to search for degree-of-freedom indices.
    entity_dim
        Topological dimension of entities where degrees-of-freedom are located.
    entities
        Indices of mesh entities of dimension ``entity_dim`` where
        degrees-of-freedom are located.
    remote : True
        True to return also "remotely located" degree-of-freedom indices.

    Returns
    -------
    numpy.ndarray
        An array of degree-of-freedom indices (local to the process) for
        degrees-of-freedom topologically belonging to mesh entities.

        If ``V`` is a list of two function spaces, then a 2-D array of
        shape (number of dofs, 2) is returned.

        Returned degree-of-freedom indices are unique and ordered by the
        first column.
    """

    _entities = np.asarray(entities, dtype=np.int32)
    if isinstance(V, collections.abc.Sequence):
        _V = []
        for space in V:
            try:
                _V.append(space._cpp_object)
            except AttributeError:
                _V.append(space)
        return _cpp.fem.locate_dofs_topological(_V, entity_dim, _entities, remote)
    else:
        try:
            return _cpp.fem.locate_dofs_topological(V, entity_dim, _entities, remote)
        except TypeError:
            return _cpp.fem.locate_dofs_topological(V._cpp_object, entity_dim, _entities, remote)


class DirichletBC:
    def __init__(
            self,
            value: typing.Union[ufl.Coefficient, Function],
            dofs: typing.List[int],
            V: typing.Union[FunctionSpace] = None,
            dtype=PETSc.ScalarType):
        """Representation of Dirichlet boundary condition which is imposed on
        a linear system.

        Parameters
        ----------
        value
            Lifted boundary values function.
        dofs
            Local indices of degrees of freedom in function space to which
            boundary condition applies.
            Expects array of size (number of dofs, 2) if function space of the
            problem, ``V``, is passed. Otherwise assumes function space of the
            problem is the same of function space of boundary values function.
        V : optional
            Function space of a problem to which boundary conditions are applied.
        dtype : optional
            The function scalar type, e.g. ``numpy.float64``.
        """

        # Construct bc value
        if isinstance(value, ufl.Coefficient):
            _value = value._cpp_object
        elif isinstance(value, _cpp.fem.Function):
            _value = value
        elif isinstance(value, Function):
            _value = value._cpp_object
        else:
            raise NotImplementedError

        # Construct bc value
        if isinstance(value, ufl.Coefficient):
            _value = value._cpp_object
        elif isinstance(value, _cpp.fem.Function):
            _value = value
        elif isinstance(value, Function):
            _value = value._cpp_object
        else:
            raise NotImplementedError

        def dirichletbc_obj(dtype):
            if dtype is np.float64:
                return _cpp.fem.DirichletBC_float64
            elif dtype is np.complex128:
                return _cpp.fem.DirichletBC_complex128
            else:
                raise NotImplementedError(f"Type {dtype} not supported.")

        if V is not None:
            # Extract cpp function space
            try:
                self._cpp_object = dirichletbc_obj(dtype)(_value, dofs, V)
            except TypeError:
                self._cpp_object = dirichletbc_obj(dtype)(_value, dofs, V._cpp_object)
        else:
            self._cpp_object = dirichletbc_obj(dtype)(_value, dofs)

    @property
    def function_space(self):
        """The function space to which boundary condition constrains
        will be applied"""
        return self._cpp_object.function_space


def bcs_by_block(spaces: typing.Iterable[FunctionSpace],
                 bcs: typing.Iterable[DirichletBC]) -> typing.Iterable[typing.Iterable[DirichletBC]]:
    """This function arranges Dirichlet boundary conditions by the
    function space that they constrain.

    Given a sequence of function spaces `spaces` and a sequence of
    DirichletBC objects `bcs`, return a list where the ith entry is the
    list of DirichletBC objects whose space is contained in
    `space[i]`."""
    def _bc_space(V, bcs):
        "Return list of bcs that have the same space as V"
        return [bc for bc in bcs if V.contains(bc.function_space)]

    return [_bc_space(V, bcs) if V is not None else [] for V in spaces]
