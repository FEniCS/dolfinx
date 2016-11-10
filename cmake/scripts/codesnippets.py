# Code snippets for autogeneration of SWIG code
#
# Copyright (C) 2012-2016 Johan Hake
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.

import sys

copyright_statement = r"""%(comment)s Auto generated SWIG file for Python interface of DOLFIN
%(comment)s
%(comment)s Copyright (C) 2012-2016 %(holder)s
%(comment)s
%(comment)s This file is part of DOLFIN.
%(comment)s
%(comment)s DOLFIN is free software: you can redistribute it and/or modify
%(comment)s it under the terms of the GNU Lesser General Public License as published by
%(comment)s the Free Software Foundation, either version 3 of the License, or
%(comment)s (at your option) any later version.
%(comment)s
%(comment)s DOLFIN is distributed in the hope that it will be useful,
%(comment)s but WITHOUT ANY WARRANTY; without even the implied warranty of
%(comment)s MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%(comment)s GNU Lesser General Public License for more details.
%(comment)s
%(comment)s You should have received a copy of the GNU Lesser General Public License
%(comment)s along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
%(comment)s

"""

# Template code for all combined SWIG modules
module_template = r"""
// The PyDOLFIN extension module for the %(module)s module
%%module(package="dolfin.cpp", directors="1") %(module)s
// Define module name for conditional includes
#define %(MODULE)sMODULE

%%{
%(headers)s

// NumPy includes
#define PY_ARRAY_UNIQUE_SYMBOL PyDOLFIN_%(MODULE)s
#include <numpy/arrayobject.h>
%%}

%%init%%{
import_array();
%%}

// Include global SWIG interface files:
// Typemaps, shared_ptr declarations, exceptions, version
%%include "dolfin/swig/globalincludes.i"
%(imports)s

// Turn on SWIG generated signature documentation and include doxygen
// generated docstrings
//%%feature("autodoc", "1");
%(docstrings)s
%(includes)s

"""


if sys.version_info[0] < 3:
    swig_py_args = ""
else:
    swig_py_args = "\n  -py3 \n  -relativeimport"


swig_cmakelists_str = \
"""# Automatic get the module name
get_filename_component(SWIG_MODULE_NAME ${CMAKE_CURRENT_BINARY_DIR} NAME)

# Set project name
project(${SWIG_MODULE_NAME})

# What does this do?
get_directory_property(cmake_defs COMPILE_DEFINITIONS)

# Set SWIG flags
set(CMAKE_SWIG_FLAGS
  -module ${SWIG_MODULE_NAME}
  -shadow
  -modern
  -modernargs
  -fastdispatch
  -fvirtual
  -nosafecstrings
  -noproxydel
  -fastproxy
  -fastinit
  -fastunpack
  -fastquery
  -nobuildnone%s
  -Iinclude/swig
  ${DOLFIN_CXX_DEFINITIONS}
  ${DOLFIN_PYTHON_DEFINITIONS}
  )

# Get all SWIG interface files
file(READ ${CMAKE_CURRENT_BINARY_DIR}/dependencies.txt DOLFIN_SWIG_DEPENDENCIES )

# This prevents swig being run unnecessarily
set_source_files_properties(module.i PROPERTIES SWIG_MODULE_NAME ${SWIG_MODULE_NAME})

# Tell CMake SWIG has generated a C++ file
set_source_files_properties(module.i PROPERTIES CPLUSPLUS ON)

# Generate SWIG files in
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR})

# Tell CMake which SWIG interface files should be checked for changes
# when recompile
set(SWIG_MODULE_${SWIG_MODULE_NAME}_EXTRA_DEPS copy_swig_files ${DOLFIN_SWIG_DEPENDENCIES})

# Tell CMake to run SWIG on module.i and to link against libdolfin
swig_add_module(${SWIG_MODULE_NAME} python module.i)
swig_link_libraries(${SWIG_MODULE_NAME} dolfin ${PYTHON_LIBRARIES})

# Install Python targets and .py files
install(TARGETS
  ${SWIG_MODULE_${SWIG_MODULE_NAME}_REAL_NAME}
  DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin/cpp
  COMPONENT RuntimeLibraries
  )
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${SWIG_MODULE_NAME}.py
  DESTINATION ${DOLFIN_INSTALL_PYTHON_MODULE_DIR}/dolfin/cpp
  COMPONENT RuntimeLibraries
  )
""" % (swig_py_args,)
