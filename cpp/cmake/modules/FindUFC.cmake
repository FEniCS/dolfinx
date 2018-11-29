# - Try to find UFC
#
# Once done this will define
#
# This module defines
#
#  UFC_FOUND        - system has UFC with correct version
#  UFC_INCLUDE_DIRS - where to find ufc.h
#  UFC_VERSION      - UFC version
#  UFC_SIGNATURE    - UFC signature

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

# Two paths: Set UFC_INCLUDE_DIR manually, or ask Python/FFC for location
# of UFC headers.

if (DEFINED UFC_INCLUDE_DIR)
  MESSAGE(STATUS "Looking for UFC in ${UFC_INCLUDE_DIR}...")

  if (EXISTS "${UFC_INCLUDE_DIR}/ufc.h" AND EXISTS "${UFC_INCLUDE_DIR}/ufc_geometry.h")
    set(UFC_INCLUDE_DIRS ${UFC_INCLUDE_DIR} CACHE STRING "Where to find ufc.h and ufc_geometry.h")
    execute_process(
      COMMAND /bin/bash -c "cat ${UFC_INCLUDE_DIR}/ufc.h ${UFC_INCLUDE_DIR}/ufc_geometry.h | sha1sum | cut -c 1-40"
      OUTPUT_VARIABLE UFC_SIGNATURE OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Assume user knows what they are doing.
    set(UFC_VERSION ${UFC_FIND_VERSION})
    set(UFC_VERSION_OK TRUE)
   else()
       MESSAGE(STATUS "Could not find UFC header.")
   endif()
else()
  MESSAGE(STATUS "Asking Python module FFC for location of UFC...")
  find_package(PythonInterp 3 REQUIRED)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import ffc, sys; sys.stdout.write(ffc.codegeneration.get_include_path())"
    OUTPUT_VARIABLE UFC_INCLUDE_DIR
    )

  if (UFC_INCLUDE_DIR)
    set(UFC_INCLUDE_DIRS ${UFC_INCLUDE_DIR} CACHE STRING "Where to find ufc.h and ufc_geometry.h")

    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import ffc, sys; sys.stdout.write(ffc.__version__)"
      OUTPUT_VARIABLE UFC_VERSION
      )

    if (UFC_FIND_VERSION)
      # Check if version found is >= required version
      if (NOT "${UFC_VERSION}" VERSION_LESS "${UFC_FIND_VERSION}")
        set(UFC_VERSION_OK TRUE)
      endif()
    else()
      # No specific version requested
      set(UFC_VERSION_OK TRUE)
    endif()
  endif()

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import ffc.codegeneration, sys; sys.stdout.write(ffc.codegeneration.get_signature())"
    OUTPUT_VARIABLE UFC_SIGNATURE
  )
endif()

mark_as_advanced(UFC_VERSION UFC_INCLUDE_DIRS UFC_SIGNATURE UFC_VERSION_OK)
# Standard package handling
find_package_handle_standard_args(UFC
                                  "UFC could not be found."
                                  UFC_INCLUDE_DIRS
                                  UFC_VERSION
                                  UFC_VERSION_OK
                                  UFC_SIGNATURE)
