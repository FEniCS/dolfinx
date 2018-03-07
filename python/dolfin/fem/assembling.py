# -*- coding: utf-8 -*-
# Copyright (C) 2007-2015 Anders Logg
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""This module provides functionality for form assembly in Python,
corresponding to the C++ assembly and PDE classes.

The C++ :py:class:`assemble <dolfin.cpp.assemble>` function
(renamed to cpp_assemble) is wrapped with an additional
preprocessing step where code is generated using the
FFC JIT compiler.

The C++ PDE classes are reimplemented in Python since the C++ classes
rely on the dolfin::Form class which is not used on the Python side.

"""

import ufl
import dolfin.cpp as cpp
from dolfin.fem.form import Form

__all__ = ["assemble_local", "SystemAssembler"]


class Assembler:
    def __init__(self, a, L, bcs=None, form_compiler_parameters=None):

        self.a = a
        self.L = L
        if bcs is None:
            self.bcs = []
        else:
            self.bcs = bcs
        self.assembler = None

    def assemble(self, A=None, b=None, mat_type=cpp.fem.Assembler.BlockType.monolithic):
        if self.assembler is None:
            # Compile forms
            try:
                a_forms = [[_create_dolfin_form(a)
                            for a in row] for row in self.a]
            except TypeError:
                a_forms = [[_create_dolfin_form(self.a)]]
            try:
                L_forms = [_create_dolfin_form(L) for L in self.L]
            except TypeError:
                L_forms = [_create_dolfin_form(self.L)]

            # Create assembler
            self.assembler = cpp.fem.Assembler(a_forms, L_forms, self.bcs)

        # Create matrix/vector (if required)
        if A is None:
            # comm = A_dolfin_form.mesh().mpi_comm()
            comm = cpp.MPI.comm_world
            A = cpp.la.PETScMatrix(comm)
        if b is None:
            # comm = b_dolfin_form.mesh().mpi_comm()
            comm = cpp.MPI.comm_world
            b = cpp.la.PETScVector(comm)

        #self.assembler.assemble(A, b)
        self.assembler.assemble(A, mat_type)
        self.assembler.assemble(b)
        return A, b


def _create_dolfin_form(form,
                        form_compiler_parameters=None,
                        function_spaces=None):
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
        return Form(
            form,
            form_compiler_parameters=form_compiler_parameters,
            function_spaces=function_spaces)
    else:
        raise TypeError("Invalid form type %s" % (type(form), ))


def assemble_local(form, cell, form_compiler_parameters=None):
    """JIT assemble_local"""
    # Create dolfin Form object
    if isinstance(form, cpp.fem.Form):
        dolfin_form = form
    else:
        dolfin_form = _create_dolfin_form(form, form_compiler_parameters)
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
    A_dolfin_form = _create_dolfin_form(A_form, form_compiler_parameters)
    b_dolfin_form = _create_dolfin_form(b_form, form_compiler_parameters)

    # Create tensors
    comm_A = A_dolfin_form.mesh().mpi_comm()
    # comm_b = A_dolfin_form.mesh().mpi_comm()
    if A_tensor is None:
        A_tensor = cpp.la.PETScMatrix(comm_A)
    if b_tensor is None:
        b_tensor = cpp.la.PETScVector(comm_A)

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
        A_dolfin_form = _create_dolfin_form(A_form, form_compiler_parameters)
        b_dolfin_form = _create_dolfin_form(b_form, form_compiler_parameters)

        # Check bcs
        bcs = _wrap_in_list(bcs, 'bcs', cpp.fem.DirichletBC)

        # Call C++ assemble function
        cpp.fem.SystemAssembler.__init__(self, A_dolfin_form, b_dolfin_form,
                                         bcs)

        # Keep Python counterpart of bcs (and Python object it owns)
        # alive
        self._bcs = bcs
