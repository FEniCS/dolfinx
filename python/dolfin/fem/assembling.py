# -*- coding: utf-8 -*-
# Copyright (C) 2007-2015 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Form assembly

The C++ :py:class:`assemble <dolfin.cpp.assemble>` function
(renamed to cpp_assemble) is wrapped with an additional
preprocessing step where code is generated using the
FFC JIT compiler.

The C++ PDE classes are reimplemented in Python since the C++ classes
rely on the dolfin::Form class which is not used on the Python side.

"""
import functools
import typing

import ufl

from dolfin import cpp
from dolfin.fem.form import Form
from dolfin.fem.dirichletbc import DirichletBC


def _create_cpp_form(form, form_compiler_parameters=None):
    # First check if we got a cpp.Form
    if isinstance(form, cpp.fem.Form):

        # Check that jit compilation has already happened
        if not hasattr(form, "_compiled_form"):
            raise TypeError(
                "Expected a dolfin form to have a _compiled_form attribute.")

        # Warn that we don't use the parameters if we get any
        if form_compiler_parameters is not None:
            cpp.warning(
                "Ignoring form_compiler_parameters when passed a dolfin Form!")
        return form
    elif isinstance(form, ufl.Form):
        form = Form(
            form, form_compiler_parameters=form_compiler_parameters)
        return form._cpp_object
    else:
        raise TypeError("Invalid form type %s" % (type(form), ))


def assemble_local(form, cell, form_compiler_parameters=None):
    """JIT assemble_local"""
    # Create dolfin Form object
    if isinstance(form, cpp.fem.Form):
        dolfin_form = form
    else:
        dolfin_form = _create_cpp_form(form, form_compiler_parameters)
    result = cpp.fem.assemble_local(dolfin_form, cell)
    if result.shape[1] == 1:
        if result.shape[0] == 1:
            result = result[0][0]
        else:
            result = result.reshape((result.shape[0]))
    return result


def assemble_system(A_form,
                    b_form,
                    bcs=None,
                    x0=None,
                    form_compiler_parameters=None,
                    add_values=False,
                    finalize_tensor=True,
                    keep_diagonal=False,
                    A_tensor=None,
                    b_tensor=None,
                    backend=None):
    """Assemble form(s) and apply any given boundary conditions in a
    symmetric fashion and return tensor(s).

    The standard application of boundary conditions does not
    necessarily preserve the symmetry of the assembled matrix. In
    order to perserve symmetry in a system of equations with boundary
    conditions, one may use the alternative assemble_system instead of
    multiple calls to :py:func:`assemble
    <dolfin.fem.assembling.assemble>`.

    *Examples of usage*

       For instance, the statements

       .. code-block:: python

           A = assemble(a)
           b = assemble(L)
           bc.apply(A, b)

       can alternatively be carried out by

       .. code-block:: python

           A, b = assemble_system(a, L, bc)

       The statement above is valid even if ``bc`` is a list of
       :py:class:`DirichletBC <dolfin.fem.bcs.DirichletBC>`
       instances. For more info and options, see :py:func:`assemble
       <dolfin.fem.assembling.assemble>`.

    """
    # Create dolfin Form objects referencing all data needed by
    # assembler
    A_dolfin_form = _create_cpp_form(A_form, form_compiler_parameters)
    b_dolfin_form = _create_cpp_form(b_form, form_compiler_parameters)

    # Create tensors
    if A_tensor is None:
        A_tensor = cpp.la.PETScMatrix()
    if b_tensor is None:
        b_tensor = cpp.la.PETScVector()

    # Check bcs
    bcs = _wrap_in_list(bcs, 'bcs', cpp.fem.DirichletBC)

    # Call C++ assemble function
    assembler = cpp.fem.SystemAssembler(A_dolfin_form, b_dolfin_form, bcs)
    assembler.add_values = add_values
    assembler.finalize_tensor = finalize_tensor
    assembler.keep_diagonal = keep_diagonal
    if x0 is not None:
        assembler.assemble(A_tensor, b_tensor, x0)
    else:
        assembler.assemble(A_tensor, b_tensor)

    return A_tensor, b_tensor


def _wrap_in_list(obj, name, types=type):
    if obj is None:
        lst = []
    elif hasattr(obj, '__iter__'):
        lst = list(obj)
    else:
        lst = [obj]
    for obj in lst:
        if not isinstance(obj, types):
            raise TypeError("expected a (list of) %s as '%s' argument" %
                            (str(types), name))
    return lst


def _create_tensor(mpi_comm, form, rank, backend, tensor):
    """Create tensor for form"""

    # Check if tensor is supplied by user
    if tensor is not None:
        return tensor

    # Check backend argument
    if (backend is not None) and (not isinstance(
            backend, cpp.la.GenericLinearAlgebraFactory)):
        raise TypeError("Provide a GenericLinearAlgebraFactory as 'backend'")

    # Create tensor
    if rank == 0:
        tensor = cpp.la.Scalar(mpi_comm)
    elif rank == 1:
        if backend:
            tensor = backend.create_vector(mpi_comm)
        else:
            tensor = cpp.la.Vector(mpi_comm)
    elif rank == 2:
        if backend:
            tensor = backend.create_matrix(mpi_comm)
        else:
            tensor = cpp.la.Matrix(mpi_comm)
    else:
        raise RuntimeError("Unable to create tensors of rank %d." % rank)

    return tensor


class SystemAssembler(cpp.fem.SystemAssembler):
    def __init__(self, A_form, b_form, bcs=None,
                 form_compiler_parameters=None):
        """
        Create a SystemAssembler

        * Arguments *
           a (ufl.Form, _Form_)
              Bilinear form
           L (ufl.Form, _Form_)
              Linear form
           bcs (_DirichletBC_)
              A list or a single DirichletBC (optional)
        """
        # Create dolfin Form objects referencing all data needed by
        # assembler
        A_dolfin_form = _create_cpp_form(A_form, form_compiler_parameters)
        b_dolfin_form = _create_cpp_form(b_form, form_compiler_parameters)

        # Check bcs
        bcs = _wrap_in_list(bcs, 'bcs', cpp.fem.DirichletBC)

        # Call C++ assemble function
        cpp.fem.SystemAssembler.__init__(self, A_dolfin_form, b_dolfin_form,
                                         bcs)

        # Keep Python counterpart of bcs (and Python object it owns)
        # alive
        self._bcs = bcs


@functools.singledispatch
def assemble(M: typing.Union[Form, cpp.fem.Form]
             ) -> typing.Union[float, cpp.la.PETScMatrix, cpp.la.PETScVector]:
    """Assemble a form over mesh"""
    M_cpp = _create_cpp_form(M)
    return cpp.fem.assemble(M_cpp)


@assemble.register(cpp.la.PETScVector)
def _assemble_vector(b: cpp.la.PETScVector,
                     L,
                     a=[],
                     bcs: typing.List[DirichletBC] = [],
                     scale: float = 1.0) -> cpp.la.PETScVector:
    """Re-assemble linear form into a vector, with modification for Dirichlet
    boundary conditions

    """
    try:
        L_cpp = [_create_cpp_form(form) for form in L]
        a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    except TypeError:
        L_cpp = [_create_cpp_form(L)]
        a_cpp = [[_create_cpp_form(form) for form in a]]

    cpp.fem.reassemble_blocked_vector(b, L_cpp, a_cpp, bcs, scale)
    return b


@assemble.register(cpp.la.PETScMatrix)
def _assemble_matrix(A: cpp.la.PETScMatrix, a, bcs=[]) -> cpp.la.PETScMatrix:
    """Re-assemble bilinear form into a vector, with rows and columns with Dirichlet
    boundary conditions zeroed.

    """
    try:
        a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    except TypeError:
        a_cpp = [[_create_cpp_form(a)]]
    cpp.fem.reassemble_blocked_matrix(A, a_cpp, bcs)
    return A


def assemble_vector(L,
                    a,
                    bcs: typing.List[DirichletBC],
                    block_type: cpp.fem.BlockType,
                    scale: float = 1.0) -> cpp.la.PETScVector:
    """Assemble linear form into vector"""
    L_cpp = [_create_cpp_form(form) for form in L]
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    return cpp.fem.assemble_blocked_vector(L_cpp, a_cpp, bcs, block_type,
                                           scale)


def assemble_matrix(a, bcs: typing.List[DirichletBC],
                    block_type) -> cpp.la.PETScMatrix:
    """Assemble bilinear forms into matrix"""
    a_cpp = [[_create_cpp_form(form) for form in row] for row in a]
    return cpp.fem.assemble_blocked_matrix(a_cpp, bcs, block_type)


def set_bc(b: cpp.la.PETScVector, L, bcs: typing.List[DirichletBC]) -> None:
    """Insert boundary condition values into vector"""
    cpp.fem.set_bc(b, L, bcs)
