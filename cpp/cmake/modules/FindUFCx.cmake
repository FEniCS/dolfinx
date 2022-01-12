<<<<<<< HEAD
# - Find UFCx by interrogating the Python module FFCx
=======
# - Try to find UFCx
>>>>>>> origin/main
#
# Once done this will define
#
# This module defines
#
<<<<<<< HEAD
#  UFCX_FOUND        - system has UFCx
=======
#  UFCX_FOUND        - system has UFC with correct version
>>>>>>> origin/main
#  UFCX_INCLUDE_DIRS - where to find ufcx.h
#  UFCX_VERSION      - UFC version
#  UFCX_SIGNATURE    - UFC signature

#=============================================================================
<<<<<<< HEAD
# Copyright (C) 2010-2021 Johannes Ring and Garth N. Wells
=======
# Copyright (C) 2010 Johannes Ring
>>>>>>> origin/main
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

<<<<<<< HEAD
MESSAGE(STATUS "Asking Python module FFCx for location of UFC... (Python executable: ${Python3_EXECUTABLE})")

# Get include path
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import ffcx.codegeneration, sys; sys.stdout.write(ffcx.codegeneration.get_include_path())"
  OUTPUT_VARIABLE UFCX_INCLUDE_DIR)

# Get ufcx.h version
if (UFCX_INCLUDE_DIR)
  set(UFCX_INCLUDE_DIRS ${UFCX_INCLUDE_DIR} CACHE STRING "Where to find ufcx.h")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import ffcx, sys; sys.stdout.write(ffcx.__version__)"
    OUTPUT_VARIABLE UFCX_VERSION)
endif()

# Compute hash of ufcx.h
find_file(_UFCX_HEADER "ufcx.h" ${UFCX_INCLUDE_DIR})
if (_UFCX_HEADER)
  file(SHA1 ${_UFCX_HEADER} UFCX_SIGNATURE)
endif()

mark_as_advanced(UFCX_VERSION UFCX_INCLUDE_DIRS UFCX_SIGNATURE)
find_package_handle_standard_args(UFCx
                                  REQUIRED_VARS UFCX_INCLUDE_DIRS UFCX_SIGNATURE UFCX_VERSION
                                  VERSION_VAR UFCX_VERSION
                                  REASON_FAILURE_MESSAGE "UFCx could not be found.")
=======
# Two paths: Set UFCX_INCLUDE_DIR manually, or ask Python/FFCx for location
# of UFC headers.

if (DEFINED ENV{UFCX_INCLUDE_DIR})
  MESSAGE(STATUS "Looking for UFC in $ENV{UFCX_INCLUDE_DIR}...")

  if (EXISTS "$ENV{UFCX_INCLUDE_DIR}/ufcx.h")
    set(UFCX_INCLUDE_DIRS $ENV{UFCX_INCLUDE_DIR} CACHE STRING "Where to find ufcx.h")
    execute_process(
      COMMAND /bin/bash -c "cat $ENV{UFCX_INCLUDE_DIR}/ufcx.h | sha1sum | cut -c 1-40"
      OUTPUT_VARIABLE UFCX_SIGNATURE OUTPUT_STRIP_TRAILING_WHITESPACE)
    # Assume user knows what they are doing.
    set(UFCX_VERSION ${UFCX_FIND_VERSION})
    set(UFCX_VERSION_OK TRUE)
   else()
       MESSAGE(STATUS "Could not find UFC header.")
   endif()
else()
  MESSAGE(STATUS "Asking Python module FFCx for location of UFC... (Python executable: ${Python3_EXECUTABLE})")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import ffcx.codegeneration, sys; sys.stdout.write(ffcx.codegeneration.get_include_path())"
    OUTPUT_VARIABLE UFCX_INCLUDE_DIR
    )

  if (UFCX_INCLUDE_DIR)
    set(UFCX_INCLUDE_DIRS ${UFCX_INCLUDE_DIR} CACHE STRING "Where to find ufcx.h")

    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import ffcx, sys; sys.stdout.write(ffcx.__version__)"
      OUTPUT_VARIABLE UFCX_VERSION
      )

    if (UFCX_FIND_VERSION)
      # Check if version found is >= required version
      if (NOT "${UFCX_VERSION}" VERSION_LESS "${UFCX_FIND_VERSION}")
        set(UFCX_VERSION_OK TRUE)
      endif()
    else()
      # No specific version requested
      set(UFCX_VERSION_OK TRUE)
    endif()
  endif()

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import ffcx.codegeneration, sys; sys.stdout.write(ffcx.codegeneration.get_signature())"
    OUTPUT_VARIABLE UFCX_SIGNATURE
  )
endif()

mark_as_advanced(UFCX_VERSION UFCX_INCLUDE_DIRS UFCX_SIGNATURE UFCX_VERSION_OK)
# Standard package handling
find_package_handle_standard_args(UFCx
                                  "UFCx could not be found."
                                  UFCX_INCLUDE_DIRS
                                  UFCX_VERSION
                                  UFCX_VERSION_OK
                                  UFCX_SIGNATURE)
>>>>>>> origin/main
