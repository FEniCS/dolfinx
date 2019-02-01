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

from petsc4py import PETSc

import ufl
from dolfin import cpp, fem
from dolfin.fem.form import Form


def _create_cpp_form(form, form_compiler_parameters=None):
    """Create a C++ Form from a UFL form"""
    if isinstance(form, cpp.fem.Form):
        if form_compiler_parameters is not None:
            cpp.warning(
                "Ignoring form_compiler_parameters when passed a dolfin Form!")
        return form
    elif isinstance(form, ufl.Form):
        form = Form(form, form_compiler_parameters=form_compiler_parameters)
        return form._cpp_object
    elif form is None:
        return None
    else:
        raise TypeError("Invalid form type: {}".format(type(form)))


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

    # A_tensor = fem.assemble(A_form)
    # b_tensor = fem.assemble(b_form)

    # Create tensors
    if A_tensor is None:
        A_tensor = cpp.fem.create_matrix(A_dolfin_form)
    if b_tensor is None:
        b_tensor = cpp.la.create_vector(b_dolfin_form.function_space(0).dofmap().index_map())

    # Check bcs
    bcs = _wrap_in_list(bcs, 'bcs', cpp.fem.DirichletBC)

    fem.assemble(b_tensor, b_form)
    fem.apply_lifting(b_tensor, [A_form], [bcs])
    b_tensor.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.set_bc(b_tensor, bcs)

    fem.assemble(A_tensor, A_form, bcs)

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
