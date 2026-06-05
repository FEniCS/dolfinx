# Copyright (C) 2017-2026 Jack S. Hale, Chris N. Richardson
# and Garth N. Wells
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Just-in-time (JIT) compilation using FFCx."""

import functools
import json
import os
import sys
from pathlib import Path

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
    "cffi_verbose": (False, "CFFI verbose mode"),
    "cffi_libraries": (None, "Extra libraries to link"),
    "timeout": (10, "Timeout for JIT compilation"),
}

if sys.platform.startswith("win32"):
    DOLFINX_DEFAULT_JIT_OPTIONS["cffi_extra_compile_args"] = (
        ["-O2"],
        "Extra C compiler arguments to pass to CFFI",
    )
else:
    DOLFINX_DEFAULT_JIT_OPTIONS["cffi_extra_compile_args"] = (
        ["-O2", "-g0"],
        "Extra C compiler arguments to pass to CFFI",
    )


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
        # Just call JIT compiler when running in with one rank
        if comm.size == 1:
            return local_jit(*args, **kwargs)

        # Remove possibility of unbound variables
        output = None
        status = 1  # assume failure
        error_msg = ""

        # Compile on rank 0
        is_root = comm.rank == 0
        if is_root:
            try:
                output = local_jit(*args, **kwargs)
                status = 0
            except Exception as e:
                error_msg = str(e)
        else:
            status = None  # placeholder for bcast

        status = comm.bcast(status, root=0)
        if status != 0:
            # Only root includes the detailed message.
            if is_root:
                raise RuntimeError(f"Failed JIT compilation of form: {error_msg}")
            else:
                raise RuntimeError("JIT compilation failed on rank 0.")

        # Load cache on all other ranks
        if not is_root:
            try:
                status_local = 0
                output = local_jit(*args, **kwargs)
            except Exception as e:
                status_local = 1
                print(f"JIT cache load failed on rank {comm.rank}: {e}", flush=True)
        else:
            status_local = 0  # root already succeeded

        any_fail = comm.allreduce(status_local, op=MPI.MAX)
        if any_fail:
            raise RuntimeError("JIT cache load failed on at least one rank.")

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


def get_options(priority_options: dict | None = None) -> dict:
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
    ufl_object, form_compiler_options: dict | None = None, jit_options: dict | None = None
):
    """Compile UFL object with FFCx and CFFI.

    Args:
        ufl_object: Object to compile, e.g. ``ufl.Form``.
        form_compiler_options: Options used in FFCx compilation of
            this form. Execute ``print(ffcx.options.FFCX_DEFAULT_OPTIONS)``
            to see all available options. Takes priority over all other
            option values.
        jit_options: Options used in CFFI JIT compilation of C code
            generated by FFCx. Execute
            ``print(dolfinx.jit.DOLFINX_DEFAULT_JIT_OPTIONS)`` to see all
            available options. The values in the ``jit_options``
            argument take priority over all other option values.

    Returns:
        A three-tuple containing the compiled object, module and a tuple
        containing the header and implementation code.

    Priority ordering of options controlling DOLFINx JIT
    compilation from highest to lowest is:

    -  ``jit_options`` (API, recommended),
    -  ``$PWD/dolfinx_jit_options.json`` (local options),
    -  ``$XDG_CONFIG_HOME/dolfinx/dolfinx_jit_options.json``
       (user options),
    -  default ``DOLFINX_DEFAULT_JIT_OPTIONS`` dictionary in
       :mod:`dolfinx.jit`.

    Priority ordering of options controlling FFCx from highest to
    lowest is:

    -  ``form_compiler_options`` (API, recommended),
    -  ``$PWD/ffcx_options.json`` (local options),
    -  ``$XDG_CONFIG_HOME/ffcx/ffcx_options.json`` (user options),
    -  ``FFCX_DEFAULT_OPTIONS`` in ``ffcx.options``.

    ``$XDG_CONFIG_HOME`` is ``~/.config/`` if the environment variable is
    not set.

    For example, ``dolfinx_jit_options.json`` could contain:

    Example::

        { "cffi_extra_compile_args": ["-O2", "-march=native" ],
          "cffi_verbose": True }

    Note::
        The contents of the ``dolfinx_options.json`` files are cached
        on the first call. Subsequent calls to this function use this
        cache.
    """
    p_ffcx = ffcx.get_options(form_compiler_options)
    p_jit = get_options(jit_options)

    # Switch on type and compile, returning cffi object
    if isinstance(ufl_object, ufl.Form):
        r = ffcx.codegeneration.jit.compile_forms([ufl_object], options=p_ffcx, **p_jit)
    elif isinstance(ufl_object, tuple) and isinstance(ufl_object[0], ufl.core.expr.Expr):
        r = ffcx.codegeneration.jit.compile_expressions([ufl_object], options=p_ffcx, **p_jit)
    else:
        raise TypeError(type(ufl_object))

    return (r[0][0], r[1], r[2])
