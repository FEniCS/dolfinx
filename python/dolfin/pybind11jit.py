# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import hashlib
import os
import re
import sys

import dijitso
from dolfin import cpp, jit
from dolfin.cpp.log import LogLevel, log


def jit_generate(cpp_code, module_name, signature, parameters):

    log(LogLevel.TRACE,
        "Calling dijitso just-in-time (JIT) compiler for pybind11 code.")

    # Split code on reserved word "SIGNATURE" which will be replaced
    # by the module signature
    # This must occur only once in the code
    split_cpp_code = re.split('SIGNATURE', cpp_code)
    if len(split_cpp_code) < 2:
        raise RuntimeError(
            "Cannot find keyword: SIGNATURE in pybind11 C++ code.")
    elif len(split_cpp_code) > 2:
        raise RuntimeError(
            "Found multiple instances of keyword: SIGNATURE in pybind11 C++ code."
        )

    code_c = split_cpp_code[0] + signature + split_cpp_code[1]

    code_h = ""
    depends = []

    return code_h, code_c, depends


def compile_cpp_code(cpp_code,
                     include_dirs=[],
                     libs=[],
                     lib_dirs=[],
                     cxxflags=[]):
    """Compile a user C(++) string and expose as a Python object with
    pybind11.

    """

    # Set compiler/build options
    # FIXME: need to locate Python libs and pybind11
    from distutils import sysconfig
    params = dijitso.params.default_params()
    pyversion = "python" + sysconfig.get_config_var("LDVERSION")
    params['cache']['lib_prefix'] = ""
    params['cache']['lib_basename'] = ""
    params['cache']['lib_loader'] = "import"

    libpython = []
    if sysconfig.get_config_var("Py_ENABLE_SHARED"):
        # only link libpython if Python itself is dynamically linked.
        # libpython must not be linked if Python itself is statically linked,
        # and probably doesn't ever need to be linked here.
        libpython = [pyversion]

    # Include path and library info from DOLFIN (dolfin.pc)
    params['build']['include_dirs'] = jit.dolfin_pc["include_dirs"] + get_pybind_include() \
        + [sysconfig.get_config_var("INCLUDEDIR") + "/" + pyversion]
    params['build']['libs'] = jit.dolfin_pc["libraries"] + libpython
    params['build']['lib_dirs'] = jit.dolfin_pc["library_dirs"] + [
        sysconfig.get_config_var("LIBDIR")
    ]

    params['build']['cxxflags'] += ('-fno-lto', )
    if sys.platform == 'darwin':
        # TODO: this should be default in dijitso
        params['build']['cxxflags'] += ('-undefined', 'dynamic_lookup')

    # Enable all macros from dolfin.pc
    dmacros = ['-D' + dm for dm in jit.dolfin_pc['define_macros']]

    params['build']['cxxflags'] += tuple(dmacros)

    # Parse argument compilation options
    params['build']['include_dirs'] += include_dirs
    params['build']['libs'] += libs
    params['build']['lib_dirs'] += lib_dirs
    params['build']['cxxflags'] += tuple(cxxflags)

    hash_str = cpp_code + cpp.__version__
    module_hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()
    module_name = "dolfin_cpp_module_" + module_hash

    module, signature = jit.dijitso_jit(
        cpp_code, module_name, params, generate=jit_generate)

    return module


def get_pybind_include():
    """Find the pybind11 include path"""

    # Look in PYBIND11_DIR
    pybind_dir = os.getenv('PYBIND11_DIR', None)
    if pybind_dir:
        p = os.path.join(pybind_dir, "include")
        if (_check_pybind_path(p)):
            return [p]

    # Try extracting from pybind11 module
    try:
        # Get include paths from module
        import pybind11
        return [pybind11.get_include(True), pybind11.get_include()]
    except Exception:
        pass

    # Look in /usr/local/include and /usr/include
    root = os.path.abspath(os.sep)
    for p in (os.path.join(root, "usr", "local", "include"),
              os.path.join(root, "usr", "include")):
        if (_check_pybind_path(p)):
            return [p]

    raise RuntimeError("Unable to locate pybind11 header files")


def _check_pybind_path(root):
    p = os.path.join(root, "pybind11", "pybind11.h")
    if os.path.isfile(p):
        return True
    else:
        return False
