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


