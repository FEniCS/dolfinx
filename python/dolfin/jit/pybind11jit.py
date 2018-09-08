# -*- coding: utf-8 -*-
# Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
#
# This file is part of DOLFIN (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import hashlib
import dijitso
import re

from dolfin.cpp.log import log, LogLevel
from . import get_pybind_include
import dolfin.cpp as cpp
from dolfin.jit.jit import dijitso_jit, dolfin_pc


def jit_generate(cpp_code, module_name, signature, parameters):

    log(LogLevel.TRACE, "Calling dijitso just-in-time (JIT) compiler for pybind11 code.")

    # Split code on reserved word "SIGNATURE" which will be replaced
    # by the module signature
    # This must occur only once in the code
    split_cpp_code = re.split('SIGNATURE', cpp_code)
    if len(split_cpp_code) < 2:
        raise RuntimeError("Cannot find keyword: SIGNATURE in pybind11 C++ code.")
    elif len(split_cpp_code) > 2:
        raise RuntimeError("Found multiple instances of keyword: SIGNATURE in pybind11 C++ code.")

    code_c = split_cpp_code[0] + signature + split_cpp_code[1]

    code_h = ""
    depends = []

    return code_h, code_c, depends


def compile_cpp_code(cpp_code, include_dirs=[], libs=[], lib_dirs=[],
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

    # Include path and library info from DOLFIN (dolfin.pc)
    params['build']['include_dirs'] = dolfin_pc["include_dirs"] + get_pybind_include() \
        + [sysconfig.get_config_var("INCLUDEDIR") + "/" + pyversion]
    params['build']['libs'] = dolfin_pc["libraries"] + [pyversion]
    params['build']['lib_dirs'] = dolfin_pc["library_dirs"] + [sysconfig.get_config_var("LIBDIR")]

    params['build']['cxxflags'] += ('-fno-lto',)

    # Enable all macros from dolfin.pc
    dmacros = ['-D' + dm for dm in dolfin_pc['define_macros']]

    params['build']['cxxflags'] += tuple(dmacros)

    # Parse argument compilation options
    params['build']['include_dirs'] += include_dirs
    params['build']['libs'] += libs
    params['build']['lib_dirs'] += lib_dirs
    params['build']['cxxflags'] += cxxflags

    hash_str = cpp_code + cpp.__version__
    module_hash = hashlib.md5(hash_str.encode('utf-8')).hexdigest()
    module_name = "dolfin_cpp_module_" + module_hash

    module, signature = dijitso_jit(cpp_code, module_name, params,
                                    generate=jit_generate)

    return module
