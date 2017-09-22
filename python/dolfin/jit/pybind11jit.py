# -*- coding: utf-8 -*-

import hashlib
import dijitso
import pkgconfig
import re

import dolfin.cpp as cpp
from . import get_pybind_include


def jit_generate(cpp_code, module_name, signature, parameters):

    # Split code on reserved word "SIGNATURE" which will be replaced by the module signature
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
    """Compile a user C(++) string to a Python object with pybind11.  Note
       this is still experimental.

    """

    if not pkgconfig.exists('dolfin'):
        raise RuntimeError("Could not find DOLFIN pkg-config file. Please make sure appropriate paths are set.")

    # Get pkg-config data for DOLFIN
    d = pkgconfig.parse('dolfin')

    # Set compiler/build options
    # FIXME: need to locate Python libs and pybind11
    from distutils import sysconfig
    params = dijitso.params.default_params()
    pyversion = "python" + sysconfig.get_config_var("LDVERSION")
    params['cache']['lib_prefix'] = ""
    params['cache']['lib_basename'] = ""
    params['cache']['lib_loader'] = "import"
    params['build']['include_dirs'] = d["include_dirs"] + get_pybind_include() + [sysconfig.get_config_var("INCLUDEDIR") + "/" + pyversion]
    params['build']['libs'] = d["libraries"] + [pyversion]
    params['build']['lib_dirs'] = d["library_dirs"] + [sysconfig.get_config_var("LIBDIR")]
    params['build']['cxxflags'] += ('-fno-lto',)

    # enable all define macros from DOLFIN
    dmacros = ()
    for dm in d['define_macros']:
        if len(dm[1]) == 0:
            dmacros += ('-D' + dm[0],)
        else:
            dmacros += ('-D' + dm[0] + '=' + dm[1],)

    params['build']['cxxflags'] += dmacros

    # This seems to be needed by OSX but not in Linux
    # FIXME: probably needed for other libraries too
    if cpp.common.has_petsc():
        import os
        params['build']['libs'] += ['petsc']
        params['build']['lib_dirs'] += [os.environ["PETSC_DIR"] + "/lib"]

    module_hash = hashlib.md5(cpp_code.encode('utf-8')).hexdigest()
    module_name = "dolfin_cpp_module_" + module_hash

    module, signature = dijitso.jit(cpp_code, module_name, params,
                                    generate=jit_generate)

    return module
