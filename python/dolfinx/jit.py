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

from mpi4py import MPI

import ffcx
import ffcx.codegeneration.jit
import ufl

__all__ = ["ffcx_jit", "get_options", "mpi_jit_decorator"]

DOLFINX_DEFAULT_JIT_OPTIONS = {
    "cache_dir": (
        os.getenv("XDG_CACHE_HOME", default=Path.home().joinpath(".cache")) / Path("fenics"),
        "Path for storing DOLFINx JIT cache. "
        "Default prefix ~/.cache/ can be changed using XDG_CACHE_HOME environment variable.",
    ),
    "cffi_debug": (False, "CFFI debug mode"),
    "cffi_extra_compile_args": (["-O2", "-g0"], "Extra C compiler arguments to pass to CFFI"),
    "cffi_verbose": (False, "CFFI verbose mode"),
    "cffi_libraries": (None, "Extra libraries to link"),
    "timeout": (10, "Timeout for JIT compilation"),
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


@functools.cache
def _load_options():
    """Loads options from JSON files."""
    user_config_file = os.getenv("XDG_CONFIG_HOME", default=Path.home().joinpath(".config")) / Path(
        "dolfinx", "dolfinx_jit_options.json"
    )
    try:
        with open(user_config_file) as f:
            user_options = json.load(f)
    except FileNotFoundError:
        user_options = dict()

    pwd_config_file = Path.cwd().joinpath("dolfinx_jit_options.json")
    try:
        with open(pwd_config_file) as f:
            pwd_options = json.load(f)
    except FileNotFoundError:
        pwd_options = dict()

    return (user_options, pwd_options)


def get_options(priority_options: Optional[dict] = None) -> dict:
    """Return a copy of the merged JIT option values for DOLFINx.

    Args:
        priority_options: Take priority over all other option values
            (see notes).

    Returns:
        dict: Merged option values.

    Note:
        See :func:`ffcx_jit` for user facing documentation.

    """
    options = dict()
    for param, (value, _) in DOLFINX_DEFAULT_JIT_OPTIONS.items():
        options[param] = value

    # NOTE: _load_options uses functools.lru_cache
    user_options, pwd_options = _load_options()

    options.update(user_options)
    options.update(pwd_options)
    if priority_options is not None:
        options.update(priority_options)

    options["cache_dir"] = Path(str(options["cache_dir"])).expanduser()

    return options


@mpi_jit_decorator
def ffcx_jit(
    ufl_object, form_compiler_options: Optional[dict] = None, jit_options: Optional[dict] = None
):
    """Compile UFL object with FFCx and CFFI.

    Args:
        ufl_object: Object to compile, e.g. ``ufl.Form``.
        form_compiler_options: Options used in FFCx compilation of
            this form. Run ``ffcx --help`` at the command line to see
            all available options. Takes priority over all other option
            values.
        jit_options: Options used in CFFI JIT compilation of C code
            generated by FFCx. See ``python/dolfinx/jit.py`` for all
            available options. Takes priority over all other option
            values.

    Returns:
        (compiled object, module, (header code, implementation code))

    Note:
        Priority ordering of options controlling DOLFINx JIT
        compilation from highest to lowest is:

        -  **jit_options** (API)
        -  **$PWD/dolfinx_jit_options.json** (local options)
        -  **$XDG_CONFIG_HOME/dolfinx/dolfinx_jit_options.json**
           (user options)
        -  **DOLFINX_DEFAULT_JIT_OPTIONS** in `dolfinx.jit`

        Priority ordering of options controlling FFCx from highest to
        lowest is:

        -  **form_compiler_optionss** (API)
        -  **$PWD/ffcx_options.json** (local options)
        -  **$XDG_CONFIG_HOME/ffcx/ffcx_options.json** (user options)
        -  **FFCX_DEFAULT_OPTIONS** in `ffcx.options`

        `$XDG_CONFIG_HOME` is `~/.config/` if the environment variable is not set.

        The contents of the `dolfinx_options.json` files are cached
        on the first call. Subsequent calls to this function use this
        cache.

        Example `dolfinx_jit_options.json` file:

            **{ "cffi_extra_compile_args": ["-O2", "-march=native" ],  "cffi_verbose": True }**

    """
    p_ffcx = ffcx.get_options(form_compiler_options)
    p_jit = get_options(jit_options)

    # Switch on type and compile, returning cffi object
    if isinstance(ufl_object, ufl.Form):
        r = ffcx.codegeneration.jit.compile_forms([ufl_object], options=p_ffcx, **p_jit)
    elif isinstance(ufl_object, ufl.Mesh):
        r = ffcx.codegeneration.jit.compile_coordinate_maps([ufl_object], options=p_ffcx, **p_jit)
    elif isinstance(ufl_object, tuple) and isinstance(ufl_object[0], ufl.core.expr.Expr):
        r = ffcx.codegeneration.jit.compile_expressions([ufl_object], options=p_ffcx, **p_jit)
    else:
        raise TypeError(type(ufl_object))

    return (r[0][0], r[1], r[2])
