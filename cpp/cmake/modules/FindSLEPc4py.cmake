# - Try to find slepc4py
# Once done this will define
#
#  SLEPC4PY_FOUND        - system has slepc4py
#  SLEPC4PY_INCLUDE_DIRS  - include directories for slepc4py
#  SLEPC4PY_VERSION      - version of slepc4py
#  SLEPC4PY_VERSION_MAJOR - first number in SLEPC4PY_VERSION
#  SLEPC4PY_VERSION_MINOR - second number in SLEPC4PY_VERSION

# Based on FindPETSc4py.cmake

#=============================================================================
# Copyright (C) 2013 Lawrence Mitchell
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

message(STATUS "Checking for package 'SLEPc4Py'")

if(SLEPC4PY_INCLUDE_DIRS)
  # In cache already
  set(SLEPC4PY_FIND_QUIETLY TRUE)
endif(SLEPC4PY_INCLUDE_DIRS)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import slepc4py; print(slepc4py.get_include())"
  OUTPUT_VARIABLE SLEPC4PY_INCLUDE_DIRS
  RESULT_VARIABLE SLEPC4PY_NOT_FOUND
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

if(SLEPC4PY_INCLUDE_DIRS)
  set(SLEPC4PY_FOUND TRUE)
  set(SLEPC4PY_INCLUDE_DIRS ${SLEPC4PY_INCLUDE_DIRS} CACHE STRING "slepc4py include path")
else(SLEPC4PY_INCLUDE_DIRS)
  set(SLEPC4PY_FOUND FALSE)
endif(SLEPC4PY_INCLUDE_DIRS)

if(SLEPC4PY_FOUND)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import slepc4py; print(slepc4py.__version__)"
    OUTPUT_VARIABLE SLEPC4PY_VERSION
    RESULT_VARIABLE SLEPC4PY_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  string(REPLACE "." ";" SLEPC4PY_VERSION_LIST ${SLEPC4PY_VERSION})
  list(GET SLEPC4PY_VERSION_LIST 0 SLEPC4PY_VERSION_MAJOR)
  list(GET SLEPC4PY_VERSION_LIST 1 SLEPC4PY_VERSION_MINOR)
  if(NOT SLEPC4PY_FIND_QUIETLY)
    message(STATUS "slepc4py version ${SLEPC4PY_VERSION} found")
  endif(NOT SLEPC4PY_FIND_QUIETLY)
else(SLEPC4PY_FOUND)
  if(SLEPC4PY_FIND_REQUIRED)
    message(FATAL_ERROR "slepc4py missing")
  endif(SLEPC4PY_FIND_REQUIRED)
endif(SLEPC4PY_FOUND)

mark_as_advanced(SLEPC4PY_INCLUDE_DIRS, SLEPC4PY_VERSION, SLEPC4PY_VERSION_MAJOR, SLEPC4PY_VERSION_MINOR)

if (SLEPc4py_FIND_VERSION)
  # Check if version found is >= required version
  if (NOT "${SLEPC4PY_VERSION}" VERSION_LESS "${SLEPc4py_FIND_VERSION}")
    set(SLEPC4PY_VERSION_OK TRUE)
  endif()
else()
  # No specific version requested
  set(SLEPC4PY_VERSION_OK TRUE)
endif()
mark_as_advanced(SLEPC4PY_VERSION_OK)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SLEPc4py
  "SLEPc4py could not be found. Be sure to set PYTHONPATH appropriately."
  SLEPC4PY_INCLUDE_DIRS SLEPC4PY_VERSION SLEPC4PY_VERSION_OK)
