# -*- coding: utf-8 -*-

import hashlib
import dijitso
import re

from dolfin.cpp.log import log, LogLevel
from . import get_pybind_include

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


def compile_cpp_code(cpp_code):
    """Compile a user C(++) string and expose as a Python object with
    pybind11.

    Note: this is experimental

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
    params['build']['include_dirs'] = dolfin_pc["include_dirs"] + get_pybind_include() + [sysconfig.get_config_var("INCLUDEDIR") + "/" + pyversion]
    params['build']['libs'] = dolfin_pc["libraries"] + [pyversion]
    params['build']['lib_dirs'] = dolfin_pc["library_dirs"] + [sysconfig.get_config_var("LIBDIR")]

    params['build']['cxxflags'] += ('-fno-lto',)

    # Enable all macros from dolfin.pc
    dmacros = ()
    for dm in dolfin_pc['define_macros']:
        if len(dm[1]) == 0:
            dmacros += ('-D' + dm[0],)
        else:
            dmacros += ('-D' + dm[0] + '=' + dm[1],)

    params['build']['cxxflags'] += dmacros

    # This seems to be needed by OSX but not in Linux
    # FIXME: probably needed for other libraries too
    # if cpp.common.has_petsc():
    #     import os
    #     params['build']['libs'] += ['petsc']
    #     params['build']['lib_dirs'] += [os.environ["PETSC_DIR"] + "/lib"]

    module_hash = hashlib.md5(cpp_code.encode('utf-8')).hexdigest()
    module_name = "dolfin_cpp_module_" + module_hash

    module, signature = dijitso_jit(cpp_code, module_name, params,
                                    generate=jit_generate)

    return module
