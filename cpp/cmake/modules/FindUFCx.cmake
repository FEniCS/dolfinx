#=============================================================================
# - Try to find UFCx by interrogating the Python module FFCx
# Once done this will define
#
#  UFCX_FOUND        - system has UFCx
#  UFCX_INCLUDE_DIRS - include directories for UFCx
#  UFCX_SIGNATURE    - signature for UFCx
#  UFCX_VERSION      - version for UFCx
#
#=============================================================================
# Copyright (C) 2010-2021 Johannes Ring and Garth N. Wells
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

find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED
)

message(
  STATUS
    "Asking Python module FFCx for location of ufcx.h..."
)

# Get include path
execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "import ffcx.codegeneration, sys; sys.stdout.write(ffcx.codegeneration.get_include_path())"
  OUTPUT_VARIABLE UFCX_INCLUDE_DIR
)
# TODO: CMake 3.20 has more modern cmake_path.
file(TO_CMAKE_PATH "${UFCX_INCLUDE_DIR}" UFCX_INCLUDE_DIR)

# Get ufcx.h version
if(UFCX_INCLUDE_DIR)
  set(UFCX_INCLUDE_DIRS
      ${UFCX_INCLUDE_DIR}
      CACHE STRING "Where to find ufcx.h"
  )
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import ffcx, sys; sys.stdout.write(ffcx.__version__)"
    OUTPUT_VARIABLE UFCX_VERSION
  )
endif()

# Compute hash of ufcx.h
find_file(_UFCX_HEADER "ufcx.h" ${UFCX_INCLUDE_DIR})
if(_UFCX_HEADER)
  file(SHA1 ${_UFCX_HEADER} UFCX_SIGNATURE)
endif()

mark_as_advanced(UFCX_VERSION UFCX_INCLUDE_DIRS UFCX_SIGNATURE)
find_package_handle_standard_args(
  UFCx
  REQUIRED_VARS UFCX_INCLUDE_DIRS UFCX_SIGNATURE UFCX_VERSION
  VERSION_VAR UFCX_VERSION HANDLE_VERSION_RANGE REASON_FAILURE_MESSAGE
                           "UFCx could not be found."
)
