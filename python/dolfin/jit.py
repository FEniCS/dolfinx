# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import functools
import os
from pathlib import Path

import dolfinx.pkgconfig
import ffc
import ffc.codegeneration.jit
import ufl
from dolfinx import common, cpp

if dolfinx.pkgconfig.exists("dolfin"):
    dolfin_pc = dolfinx.pkgconfig.parse("dolfin")
else:
    raise RuntimeError(
        "Could not find DOLFIN pkg-config file. Make sure appropriate paths are set."
    )


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

    @functools.wraps(local_jit)
    def mpi_jit(*args, **kwargs):

        # FIXME: should require mpi_comm to be explicit
        # and not default to comm_world?
        mpi_comm = kwargs.pop("mpi_comm", cpp.MPI.comm_world)

        # Just call JIT compiler when running in serial
        if cpp.MPI.size(mpi_comm) == 1:
            return local_jit(*args, **kwargs)

        # Default status (0 == ok, 1 == fail)
        status = 0

        # Compile first on process 0
        root = cpp.MPI.rank(mpi_comm) == 0
        if root:
            try:
                output = local_jit(*args, **kwargs)
            except Exception as e:
                status = 1
                error_msg = str(e)

        # TODO: This would have lower overhead if using the dijitso.jit
        # features to inject a waiting callback instead of waiting out
        # here. That approach allows all processes to first look in the
        # cache, introducing a barrier only on cache miss. There's also
        # a sketch in dijitso of how to make only one process per
        # physical cache directory do the compilation.

        # Wait for the compiling process to finish and get status
        # TODO: Would be better to broadcast the status from root but
        # this works.
        global_status = cpp.MPI.max(mpi_comm, status)

        if global_status == 0:
            # Success, call jit on all other processes (this should just
            # read the cache)
            if not root:
                output = local_jit(*args, **kwargs)
        else:
            # Fail simultaneously on all processes, to allow catching
            # the error without deadlock
            if not root:
                error_msg = "Compilation failed on root node."
            raise RuntimeError("Failed just-in-time compilation of form: {}".format(error_msg))
        return output

    # Return the decorated jit function
    return mpi_jit


@mpi_jit_decorator
def ffc_jit(ufl_object, form_compiler_parameters=None):
    # Prepare form compiler parameters with overrides from dolfin
    p = ffc.default_parameters()
    p["scalar_type"] = "double complex" if common.has_petsc_complex else "double"
    p.update(form_compiler_parameters or {})

    # CFFI compiler options/flags
    extra_compile_args = ['-g0', '-O3', '-march=native']
    user_cflags = os.getenv('DOLFIN_JIT_CFLAGS')
    if user_cflags is not None:
        extra_compile_args = user_cflags.split(" ")
    cffi_options = dict(cffi_extra_compile_args=extra_compile_args, cffi_verbose=False,
                        cffi_debug=False)

    # Set FFC cache location
    cache_dir = "~/.cache/fenics"
    cache_dir = os.getenv('FENICS_CACHE_DIR', cache_dir)
    cache_dir = Path(cache_dir).expanduser()

    # Switch on type and compile, returning cffi object
    if isinstance(ufl_object, ufl.Form):
        r = ffc.codegeneration.jit.compile_forms([ufl_object], parameters=p, cache_dir=cache_dir, **cffi_options)
    elif isinstance(ufl_object, ufl.FiniteElementBase):
        r = ffc.codegeneration.jit.compile_elements([ufl_object], parameters=p, cache_dir=cache_dir, **cffi_options)
    elif isinstance(ufl_object, ufl.Mesh):
        r = ffc.codegeneration.jit.compile_coordinate_maps(
            [ufl_object], parameters=p, cache_dir=cache_dir, **cffi_options)
    elif isinstance(ufl_object, tuple) and isinstance(ufl_object[0], ufl.core.expr.Expr):
        r = ffc.codegeneration.jit.compile_expressions([ufl_object], parameters=p, cache_dir=cache_dir, **cffi_options)
    else:
        raise TypeError(type(ufl_object))

    return r[0][0]
