#=============================================================================
# - Try to find UFCx by interrogating the Python module FFCx
# Once done this will define
#
#  UFCx_FOUND     - system has UFCx
#  UFCx::UFCx     - imported interface target
#  UFCX_SIGNATURE - SHA1 hash of ufcx.h
#  UFCX_VERSION   - version for UFCx
#
#=============================================================================
# Copyright (C) 2010-2026 Johannes Ring, Garth N. Wells, Jack S. Hale
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

find_package(Python3 COMPONENTS Interpreter REQUIRED)

message(STATUS "Asking Python module FFCx for location of ufcx.h...")

# Get include path
execute_process(
  COMMAND
    ${Python3_EXECUTABLE} -c
    "import ffcx.codegeneration, sys; sys.stdout.write(ffcx.codegeneration.get_include_path())"
  OUTPUT_VARIABLE _UFCX_INCLUDE_DIR
)
# Converts os native to cmake native path type
cmake_path(SET _UFCX_INCLUDE_DIR "${_UFCX_INCLUDE_DIR}")

# Get ufcx.h version
if(_UFCX_INCLUDE_DIR)
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} -c
      "import ffcx, sys; sys.stdout.write(ffcx.__version__)"
    OUTPUT_VARIABLE UFCX_VERSION
  )
endif()

# Compute hash of ufcx.h
find_file(_UFCX_HEADER "ufcx.h" PATHS ${_UFCX_INCLUDE_DIR} NO_DEFAULT_PATH)
if(_UFCX_HEADER)
  file(SHA1 ${_UFCX_HEADER} UFCX_SIGNATURE)
endif()

mark_as_advanced(UFCX_VERSION UFCX_SIGNATURE)
find_package_handle_standard_args(
  UFCx
  REQUIRED_VARS _UFCX_INCLUDE_DIR UFCX_SIGNATURE UFCX_VERSION
  VERSION_VAR UFCX_VERSION
  HANDLE_VERSION_RANGE
  REASON_FAILURE_MESSAGE "UFCx could not be found."
)

if(UFCx_FOUND AND NOT TARGET UFCx::UFCx)
  add_library(UFCx::UFCx INTERFACE IMPORTED)
  set_target_properties(
    UFCx::UFCx
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_UFCX_INCLUDE_DIR}"
  )
endif()
