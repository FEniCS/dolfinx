# Copyright (C) 2017-2018 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Just-in-time (JIT) compilation using FFCx"""

import functools
import json
import os
from pathlib import Path
from typing import Optional

import ffcx
import ffcx.codegeneration.jit
import ufl

from mpi4py import MPI

__all__ = ["ffcx_jit", "get_parameters"]

DOLFINX_DEFAULT_JIT_PARAMETERS = {
    "cache_dir":
        (os.getenv("XDG_CACHE_HOME", default=Path.home().joinpath(".cache")) / Path("fenics"),
         "Path for storing DOLFINx JIT cache. "
         "Default prefix ~/.cache/ can be changed using XDG_CACHE_HOME environment variable."),
    "cffi_debug":
        (False, "CFFI debug mode"),
    "cffi_extra_compile_args":
        (["-O2", "-g0"], "Extra C compiler arguments to pass to CFFI"),
    "cffi_verbose":
        (False, "CFFI verbose mode"),
    "cffi_libraries":
        (None, "Extra libraries to link"),
    "timeout":
        (10, "Timeout for JIT compilation")
}


def mpi_jit_decorator(local_jit, *args, **kwargs):
    """A decorator for jit compilation.

    Use this function as a decorator to any jit compiler function. In a
    parallel run, this function will first call the jit compilation
    function on the first process. When this is done, and the module is
    in the cache, it will call the jit compiler on the remaining
    processes, which will then use the cached module.

    """

    @functools.wraps(local_jit)
    def mpi_jit(comm, *args, **kwargs):

        # Just call JIT compiler when running in serial
        if comm.size == 1:
            return local_jit(*args, **kwargs)

        # Default status (0 == ok, 1 == fail)
        status = 0

        # Compile first on process 0
        root = comm.rank == 0
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
        global_status = comm.allreduce(status, op=MPI.MAX)
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
            raise RuntimeError(f"Failed just-in-time compilation of form: {error_msg}")

        return output

    # Return the decorated jit function
    return mpi_jit


@functools.lru_cache(maxsize=None)
def _load_parameters():
    """Loads parameters from JSON files."""
    user_config_file = os.getenv("XDG_CONFIG_HOME", default=Path.home().joinpath(".config")) \
        / Path("dolfinx", "dolfinx_jit_parameters.json")
    try:
        with open(user_config_file) as f:
            user_parameters = json.load(f)
    except FileNotFoundError:
        user_parameters = {}

    pwd_config_file = Path.cwd().joinpath("dolfinx_jit_parameters.json")
    try:
        with open(pwd_config_file) as f:
            pwd_parameters = json.load(f)
    except FileNotFoundError:
        pwd_parameters = {}

    return (user_parameters, pwd_parameters)


def get_parameters(priority_parameters: Optional[dict] = None) -> dict:
    """Return a copy of the merged JIT parameter values for DOLFINx.

    Args:
        priority_parameters: take priority over all other parameter values (see notes)

    Returns:
        dict: merged parameter values

    Notes:
        See ffcx_jit for user facing documentation.

    """
    parameters = {}

    for param, (value, _) in DOLFINX_DEFAULT_JIT_PARAMETERS.items():
        parameters[param] = value

    # NOTE: _load_parameters uses functools.lru_cache
    user_parameters, pwd_parameters = _load_parameters()

    parameters.update(user_parameters)
    parameters.update(pwd_parameters)
    if priority_parameters is not None:
        parameters.update(priority_parameters)

    parameters["cache_dir"] = Path(str(parameters["cache_dir"])).expanduser()

    return parameters


@mpi_jit_decorator
def ffcx_jit(ufl_object, form_compiler_params={}, jit_params={}):
    """Compile UFL object with FFCx and CFFI.

    Args:
        ufl_object: Object to compile, e.g. ufl.Form
        form_compiler_params: Parameters used in FFCx compilation of
            this form. Run `ffcx --help` at the commandline to see all
            available options. Takes priority over all other parameter
            values.
        jit_params: Parameters used in CFFI JIT compilation of C code
            generated by FFCx. See `python/dolfinx/jit.py` for all
            available parameters. Takes priority over all other
            parameter values.

    Returns:
        (compiled object, module, (header code, implementation code))

    Notes:
        Priority ordering of parameters controlling DOLFINx JIT
        compilation from highest to lowest is:

        -  **jit_params** (API)
        -  **$PWD/dolfinx_jit_parameters.json** (local parameters)
        -  **$XDG_CONFIG_HOME/dolfinx/dolfinx_jit_parameters.json**
           (user parameters)
        -  **DOLFINX_DEFAULT_JIT_PARAMETERS** in `dolfinx.jit`

        Priority ordering of parameters controlling FFCx from highest to
        lowest is:

        -  **form_compiler_paramss** (API)
        -  **$PWD/ffcx_parameters.json** (local parameters)
        -  **$XDG_CONFIG_HOME/ffcx/ffcx_parameters.json** (user parameters)
        -  **FFCX_DEFAULT_PARAMETERS** in `ffcx.parameters`

        `$XDG_CONFIG_HOME` is `~/.config/` if the environment variable is not set.

        The contents of the `dolfinx_parameters.json` files are cached
        on the first call. Subsequent calls to this function use this
        cache.

        Example `dolfinx_jit_parameters.json` file:

            **{ "cffi_extra_compile_args": ["-O2", "-march=native" ],  "cffi_verbose": True }**

    """
    # Prepare form compiler parameters with priority parameters
    p_ffcx = ffcx.get_parameters(form_compiler_params)
    p_jit = get_parameters(jit_params)

    # Switch on type and compile, returning cffi object
    if isinstance(ufl_object, ufl.Form):
        r = ffcx.codegeneration.jit.compile_forms([ufl_object], parameters=p_ffcx, **p_jit)
    elif isinstance(ufl_object, ufl.FiniteElementBase):
        r = ffcx.codegeneration.jit.compile_elements([ufl_object], parameters=p_ffcx, **p_jit)
    elif isinstance(ufl_object, ufl.Mesh):
        r = ffcx.codegeneration.jit.compile_coordinate_maps(
            [ufl_object], parameters=p_ffcx, **p_jit)
    elif isinstance(ufl_object, tuple) and isinstance(ufl_object[0], ufl.core.expr.Expr):
        r = ffcx.codegeneration.jit.compile_expressions([ufl_object], parameters=p_ffcx, **p_jit)
    else:
        raise TypeError(type(ufl_object))

    return (r[0][0], r[1], r[2])
