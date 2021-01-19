# - Try to find basix
#
# Once done this will define
#
# This module defines
#
#  BASIX_FOUND        - system has BASIX with correct version
#  BASIX_INCLUDE_DIRS - where to find basix headers
#  BASIX_LIBRARY_DIR  - where to find basix library
#  BASIX_VERSION      - BASIX version

#=============================================================================
# Copyright (C) 2010-2020 Johannes Ring and Chris Richardson
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

# Two paths: Set BASIX_PREFIX_DIR manually, or ask Python for location
# of BASIX headers.

if (DEFINED ENV{BASIX_PREFIX_DIR})
  MESSAGE(STATUS "Looking for BASIX in $ENV{BASIX_PREFIX_DIR}...")

  if (EXISTS "$ENV{BASIX_PREFIX_DIR}/include/basix.h")
    set(BASIX_INCLUDE_DIRS $ENV{BASIX_PREFIX_DIR}/include CACHE STRING "Where to find basix.h")
    set(BASIX_LIBRARY $ENV{BASIX_PREFIX_DIR}/lib/basix.so)
    # Assume user knows what they are doing.
    set(BASIX_VERSION ${BASIX_FIND_VERSION})
    set(BASIX_VERSION_OK TRUE)
   else()
       MESSAGE(STATUS "Could not find basix headers.")
   endif()
else()
  MESSAGE(STATUS "Asking Python module for location of basix headers...")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import basix, sys; sys.stdout.write(basix.get_prefix_dir())"
    OUTPUT_VARIABLE BASIX_PREFIX_DIR
    )

  if (BASIX_PREFIX_DIR)
    set(BASIX_INCLUDE_DIRS ${BASIX_PREFIX_DIR}/include CACHE STRING "Where to find basix.h")

    find_library(BASIX_LIBRARY
      NAMES basix
      HINTS ${BASIX_PREFIX_DIR}/lib
      NO_DEFAULT_PATH
      DOC "The basix library"
      )

    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import basix, sys; sys.stdout.write(basix.__version__)"
      OUTPUT_VARIABLE BASIX_VERSION
      )

    if (BASIX_FIND_VERSION)
      # Check if version found is >= required version
      if (NOT "${BASIX_VERSION}" VERSION_LESS "${BASIX_FIND_VERSION}")
        set(BASIX_VERSION_OK TRUE)
      endif()
    else()
      # No specific version requested
      set(BASIX_VERSION_OK TRUE)
    endif()
  endif()

endif()

mark_as_advanced(BASIX_VERSION BASIX_INCLUDE_DIRS BASIX_VERSION_OK BASIX_LIBRARY)
# Standard package handling
find_package_handle_standard_args(basix
                                  "basix could not be found."
                                  BASIX_INCLUDE_DIRS
                                  BASIX_VERSION
                                  BASIX_VERSION_OK
                                  BASIX_LIBRARY)
