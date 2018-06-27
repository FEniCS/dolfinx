# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from dolfin import pkgconfig
import numpy
import hashlib
import dijitso
import ffc

import dolfin.cpp as cpp
from dolfin.cpp import MPI
from functools import wraps
from dolfin.parameter import parameters


# Get DOLFIN pkg-config data
if pkgconfig.exists("dolfin"):
    dolfin_pc = pkgconfig.parse("dolfin")
else:
    raise RuntimeError(
        "Could not find DOLFIN pkg-config file. Please make sure appropriate paths are set.")


# Copied over from site-packages
def mpi_jit_decorator(local_jit, *args, **kwargs):
    """A decorator for jit compilation

    Use this function as a decorator to any jit compiler function.  In
    a parallel run, this function will first call the jit compilation
    function on the first process. When this is done, and the module
    is in the cache, it will call the jit compiler on the remaining
    processes, which will then use the cached module.

    *Example*
        .. code-block:: python

            def jit_something(something):
                ....

    """
    @wraps(local_jit)
    def mpi_jit(*args, **kwargs):

        # FIXME: should require mpi_comm to be explicit
        # and not default to comm_world?
        mpi_comm = kwargs.pop("mpi_comm", MPI.comm_world)

        # Just call JIT compiler when running in serial
        if MPI.size(mpi_comm) == 1:
            return local_jit(*args, **kwargs)

        # Default status (0 == ok, 1 == fail)
        status = 0

        # Compile first on process 0
        root = MPI.rank(mpi_comm) == 0
        if root:
            try:
                output = local_jit(*args, **kwargs)
            except Exception as e:
                status = 1
                error_msg = str(e)

        # TODO: This would have lower overhead if using the dijitso.jit
        # features to inject a waiting callback instead of waiting out here.
        # That approach allows all processes to first look in the cache,
        # introducing a barrier only on cache miss.
        # There's also a sketch in dijitso of how to make only one
        # process per physical cache directory do the compilation.

        # Wait for the compiling process to finish and get status
        # TODO: Would be better to broadcast the status from root but this works.
        global_status = MPI.max(mpi_comm, status)

        if global_status == 0:
            # Success, call jit on all other processes
            # (this should just read the cache)
            if not root:
                output = local_jit(*args, **kwargs)
        else:
            # Fail simultaneously on all processes,
            # to allow catching the error without deadlock
            if not root:
                error_msg = "Compilation failed on root node."
            cpp.dolfin_error("jit.py",
                             "perform just-in-time compilation of form",
                             error_msg)
        return output

    # Return the decorated jit function
    return mpi_jit


# Wrap FFC JIT compilation with decorator
@mpi_jit_decorator
def ffc_jit(ufl_form, form_compiler_parameters=None):

    # Prepare form compiler parameters with overrides from dolfin and kwargs
    p = ffc.default_jit_parameters()
    p.update(dict(parameters["form_compiler"]))
    p.update(form_compiler_parameters or {})
    return ffc.jit(ufl_form, parameters=p)


# Wrap dijitso JIT compilation with decorator
@mpi_jit_decorator
def dijitso_jit(*args, **kwargs):
    return dijitso.jit(*args, **kwargs)


_cpp_math_builtins = [
    # <cmath> functions: from http://www.cplusplus.com/reference/cmath/
    "cos", "sin", "tan", "acos", "asin", "atan", "atan2",
    "cosh", "sinh", "tanh", "exp", "frexp", "ldexp", "log", "log10", "modf",
    "pow", "sqrt", "ceil", "fabs", "floor", "fmod",
    "max", "min"]

_math_header = """
// cmath functions
%s

const double pi = DOLFIN_PI;
const PetscComplex j = PETSC_i;
""" % "\n".join("using std::%s;" % mf for mf in _cpp_math_builtins)


def compile_class(cpp_data):
    """Compile a user C(++) string or set of statements to a Python object

    cpp_data is a dict containing:
      "name": must be "expression"
      "statements": must be a string, or list/tuple of strings
      "properties": a dict of float properties
      "jit_generate": callable (generates cpp code with this dict as input)

    """

    # Set compiler/build options
    params = dijitso.params.default_params()
    params['build']['include_dirs'] = dolfin_pc["include_dirs"]
    params['build']['libs'] = dolfin_pc["libraries"]
    params['build']['lib_dirs'] = dolfin_pc["library_dirs"]

    name = cpp_data['name']
    if name not in ('expression'):
        raise ValueError("DOLFIN JIT only for Expression")
    statements = cpp_data['statements']
    properties = cpp_data['properties']

    if not isinstance(statements, (str, tuple, list)):
        raise RuntimeError(
            "Expression must be a string, or a list or tuple of strings")

    # Flatten tuple of tuples (2D array) and get value_shape
    statement_array = numpy.array(statements)
    cpp_data['statements'] = tuple(statement_array.flatten())
    cpp_data['value_shape'] = statement_array.shape

    # Make a string representing the properties (and distinguish float/GenericFunction)
    # by adding '*' for GenericFunction
    property_str = ''
    for k, v in properties.items():
        property_str += str(k)
        if hasattr(v, '_cpp_object') and isinstance(v._cpp_object, cpp.function.GenericFunction):
            property_str += '*'

    hash_str = str(statements) + str(property_str)
    module_hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()
    module_name = "dolfin_" + name + "_" + module_hash

    try:
        module, signature = dijitso_jit(cpp_data, module_name, params,
                                        generate=cpp_data['jit_generate'])
        submodule = dijitso.extract_factory_function(
            module, "create_" + module_name)()
    except Exception:
        raise RuntimeError("Unable to compile C++ code with dijitso")

    python_object = cpp.function.make_dolfin_expression(submodule)

    # Set properties to initial values
    # FIXME: maybe remove from here (do it in Expression instead)
    for k, v in properties.items():
        python_object.set_property(k, v)

    return python_object
