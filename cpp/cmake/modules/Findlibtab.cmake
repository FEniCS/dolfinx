# - Try to find libtab
#
# Once done this will define
#
# This module defines
#
#  LIBTAB_FOUND        - system has LIBTAB with correct version
#  LIBTAB_INCLUDE_DIRS - where to find libtab headers
#  LIBTAB_LIBRARY_DIR  - where to find libtab library
#  LIBTAB_VERSION      - LIBTAB version

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

# Two paths: Set LIBTAB_PREFIX_DIR manually, or ask Python for location
# of LIBTAB headers.

if (DEFINED ENV{LIBTAB_PREFIX_DIR})
  MESSAGE(STATUS "Looking for LIBTAB in $ENV{LIBTAB_PREFIX_DIR}...")

  if (EXISTS "$ENV{LIBTAB_PREFIX_DIR}/include/libtab.h")
    set(LIBTAB_INCLUDE_DIRS $ENV{LIBTAB_PREFIX_DIR}/include CACHE STRING "Where to find libtab.h")
    set(LIBTAB_LIBRARY $ENV{LIBTAB_PREFIX_DIR}/lib/libtab.so)
    # Assume user knows what they are doing.
    set(LIBTAB_VERSION ${LIBTAB_FIND_VERSION})
    set(LIBTAB_VERSION_OK TRUE)
   else()
       MESSAGE(STATUS "Could not find libtab headers.")
   endif()
else()
  MESSAGE(STATUS "Asking Python module for location of libtab headers...")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import libtab, sys; sys.stdout.write(libtab.get_prefix_dir())"
    OUTPUT_VARIABLE LIBTAB_PREFIX_DIR
    )

  if (LIBTAB_PREFIX_DIR)
    set(LIBTAB_INCLUDE_DIRS ${LIBTAB_PREFIX_DIR}/include CACHE STRING "Where to find libtab.h")
 
    find_library(LIBTAB_LIBRARY
      NAMES tab
      HINTS ${LIBTAB_PREFIX_DIR}/lib  
      NO_DEFAULT_PATH
      DOC "The libtab library"
      )

    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import libtab, sys; sys.stdout.write(libtab.__version__)"
      OUTPUT_VARIABLE LIBTAB_VERSION
      )

    if (LIBTAB_FIND_VERSION)
      # Check if version found is >= required version
      if (NOT "${LIBTAB_VERSION}" VERSION_LESS "${LIBTAB_FIND_VERSION}")
        set(LIBTAB_VERSION_OK TRUE)
      endif()
    else()
      # No specific version requested
      set(LIBTAB_VERSION_OK TRUE)
    endif()
  endif()

endif()

mark_as_advanced(LIBTAB_VERSION LIBTAB_INCLUDE_DIRS LIBTAB_VERSION_OK LIBTAB_LIBRARY)
# Standard package handling
find_package_handle_standard_args(libtab
                                  "libtab could not be found."
                                  LIBTAB_INCLUDE_DIRS
                                  LIBTAB_VERSION
                                  LIBTAB_VERSION_OK
                                  LIBTAB_LIBRARY)
