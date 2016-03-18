# - Try to find UFC
#
# Once done this will define
#
# This module defines
#
#  UFC_FOUND        - system has UFC with correct version
#  UFC_INCLUDE_DIRS - where to find ufc.h
#  UFC_VERSION      - UFC version

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

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} -c "import ffc, sys; sys.stdout.write(ffc.get_ufc_include())"
  OUTPUT_VARIABLE UFC_INCLUDE_DIR
  RESULT_VARIABLE UFC_NOT_FOUND
  ERROR_VARIABLE UFC_ERROR
  )

if (NOT UFC_NOT_FOUND)
  set(UFC_INCLUDE_DIRS ${UFC_INCLUDE_DIR} CACHE STRING "Where to find ufc.h")
  mark_as_advanced(UFC_INCLUDE_DIRS)

  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import ffc, sys; sys.stdout.write(ffc.__version__)"
    OUTPUT_VARIABLE UFC_VERSION
    RESULT_VARIABLE UFC_NOT_FOUND
    )
  mark_as_advanced(UFC_VERSION)

  if (UFC_FIND_VERSION)
    # Check if version found is >= required version
    if (NOT "${UFC_VERSION}" VERSION_LESS "${UFC_FIND_VERSION}")
      set(UFC_VERSION_OK TRUE)
    endif()
  else()
    # No specific version requested
    set(UFC_VERSION_OK TRUE)
  endif()
  mark_as_advanced(UFC_VERSION_OK)

endif()

# Standard package handling
find_package_handle_standard_args(UFC
                                  "UFC could not be found."
                                  UFC_INCLUDE_DIRS
                                  UFC_VERSION
                                  UFC_VERSION_OK)
