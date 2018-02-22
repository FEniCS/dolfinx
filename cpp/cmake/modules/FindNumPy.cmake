# - Find NumPy
# Find the native NumPy includes
# This module defines
#  NUMPY_INCLUDE_DIR, where to find numpy/arrayobject.h, etc.
#  NUMPY_VERSION, The string version of the numpy version
#  NUMPY_VERSION_MAJOR, The first number of the numpy version
#  NUMPY_VERSION_MINOR, The second number of the numpy version
#  NUMPY_VERSION_MICRO, The third number of the numpy version
#  NUMPY_FOUND, If false, do not try to use NumPy headers.

#=============================================================================
# Copyright (C) 2010 Johannes Ring
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

if(NUMPY_INCLUDE_DIR)
  # in cache already
  set(NUMPY_FIND_QUIETLY TRUE)
endif(NUMPY_INCLUDE_DIR)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  RESULT_VARIABLE NUMPY_NOT_FOUND
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

if(NUMPY_INCLUDE_DIR)
  set(NUMPY_FOUND TRUE)
  set(NUMPY_INCLUDE_DIR ${NUMPY_INCLUDE_DIR} CACHE STRING "NumPy include path")
else(NUMPY_INCLUDE_DIR)
  set(NUMPY_FOUND FALSE)
endif(NUMPY_INCLUDE_DIR)

if(NUMPY_FOUND)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.version.version)"
    OUTPUT_VARIABLE NUMPY_VERSION
    RESULT_VARIABLE NUMPY_NOT_FOUND
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  string(REPLACE "." ";" NUMPY_VERSION_LIST ${NUMPY_VERSION})
  list(GET NUMPY_VERSION_LIST 0 NUMPY_VERSION_MAJOR)
  list(GET NUMPY_VERSION_LIST 1 NUMPY_VERSION_MINOR)
  list(GET NUMPY_VERSION_LIST 2 NUMPY_VERSION_MICRO)
  if(NOT NUMPY_FIND_QUIETLY)
    message(STATUS "NumPy header version ${NUMPY_VERSION} found")
  endif(NOT NUMPY_FIND_QUIETLY)
else(NUMPY_FOUND)
  if(NUMPY_FIND_REQUIRED)
    message(FATAL_ERROR "NumPy headers missing")
  endif(NUMPY_FIND_REQUIRED)
endif(NUMPY_FOUND)

mark_as_advanced(NUMPY_INCLUDE_DIR, NUMPY_VERSION, NUMPY_VERSION_MAJOR, 
  NUMPY_VERSION_MINOR, NUMPY_VERSION_MICRO)
