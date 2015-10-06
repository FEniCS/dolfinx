# - Try to find petsc4py
# Once done this will define
#
#  PETSC4PY_FOUND        - system has petsc4py
#  PETSC4PY_INCLUDE_DIRS  - include directories for petsc4py
#  PETSC4PY_VERSION      - version of petsc4py
#  PETSC4PY_VERSION_MAJOR - first number in PETSC4PY_VERSION
#  PETSC4PY_VERSION_MINOR - second number in PETSC4PY_VERSION

# Based on FindNumPy.cmake

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

message(STATUS "Checking for package 'PETSc4Py'")

if(PETSC4PY_INCLUDE_DIRS)
  # In cache already
  set(PETSC4PY_FIND_QUIETLY TRUE)
endif(PETSC4PY_INCLUDE_DIRS)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIRS
  RESULT_VARIABLE PETSC4PY_NOT_FOUND
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

if(PETSC4PY_INCLUDE_DIRS)
  set(PETSC4PY_FOUND TRUE)
  set(PETSC4PY_INCLUDE_DIRS ${PETSC4PY_INCLUDE_DIRS} CACHE STRING "petsc4py include path")
else(PETSC4PY_INCLUDE_DIRS)
  set(PETSC4PY_FOUND FALSE)
endif(PETSC4PY_INCLUDE_DIRS)

if(PETSC4PY_FOUND)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import petsc4py; print(petsc4py.__version__)"
    OUTPUT_VARIABLE PETSC4PY_VERSION
    RESULT_VARIABLE PETSC4PY_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  string(REPLACE "." ";" PETSC4PY_VERSION_LIST ${PETSC4PY_VERSION})
  list(GET PETSC4PY_VERSION_LIST 0 PETSC4PY_VERSION_MAJOR)
  list(GET PETSC4PY_VERSION_LIST 1 PETSC4PY_VERSION_MINOR)
  if(NOT PETSC4PY_FIND_QUIETLY)
    message(STATUS "petsc4py version ${PETSC4PY_VERSION} found")
  endif(NOT PETSC4PY_FIND_QUIETLY)
else(PETSC4PY_FOUND)
  if(PETSC4PY_FIND_REQUIRED)
    message(FATAL_ERROR "petsc4py missing")
  endif(PETSC4PY_FIND_REQUIRED)
endif(PETSC4PY_FOUND)

mark_as_advanced(PETSC4PY_INCLUDE_DIRS, PETSC4PY_VERSION, PETSC4PY_VERSION_MAJOR, PETSC4PY_VERSION_MINOR)

if (PETSc4py_FIND_VERSION)
  # Check if version found is >= required version
  if (NOT "${PETSC4PY_VERSION}" VERSION_LESS "${PETSc4py_FIND_VERSION}")
    set(PETSC4PY_VERSION_OK TRUE)
  endif()
else()
  # No specific version requested
  set(PETSC4PY_VERSION_OK TRUE)
endif()
mark_as_advanced(PETSC4PY_VERSION_OK)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc4py
  "PETSc4py could not be found. Be sure to set PYTHONPATH appropriately."
  PETSC4PY_INCLUDE_DIRS PETSC4PY_VERSION PETSC4PY_VERSION_OK)

# Check petsc4py.i for PETSC_INT
if(PETSC4PY_INCLUDE_DIRS)
  file(STRINGS "${PETSC4PY_INCLUDE_DIRS}/petsc4py/petsc4py.i" PETSC4PY_INT)
  string(REGEX MATCH "SWIG_TYPECHECK_INT[0-9]+" PETSC4PY_INT "${PETSC4PY_INT}")
  string(REPLACE "SWIG_TYPECHECK_INT" "" PETSC4PY_INT "${PETSC4PY_INT}")
  math(EXPR PETSC_INT "${PETSC_INT_SIZE}*8")
  if(NOT PETSC_INT STREQUAL PETSC4PY_INT)
    message(STATUS "PETSC_INT = ${PETSC4PY_INT} ${PETSC_INT}")
    message(STATUS " - does not match")
    set(PETSC4PY_FOUND FALSE)
  endif()
endif()