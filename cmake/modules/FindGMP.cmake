# - Try to find the GMP librairies
# Once done this will define
#
#  GMP_FOUND        - system has GMP lib
#  GMP_INCLUDE_DIRS - include directories for GMP
#  GMP_LIBRARIES    - libraries for GMP

#=============================================================================
# Copyright (c) 2006, Laurent Montel, <montel@kde.org>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 
# 1. Redistributions of source code must retain the copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. The name of the author may not be used to endorse or promote products 
#    derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

if (GMP_INCLUDE_DIRS AND GMP_LIBRARIES)
  # Already in cache, be silent
  set(GMP_FIND_QUIETLY TRUE)
endif (GMP_INCLUDE_DIRS AND GMP_LIBRARIES)

find_path(GMP_INCLUDE_DIRS
  NAMES gmp.h
  HINTS ${GMP_DIR}/include $ENV{GMP_DIR}/include
  DOC "Directory where the GMP header file is located"
  )

find_library(GMP_LIBRARIES
  NAMES gmp libgmp
  HINTS ${GMP_DIR}/lib $ENV{GMP_DIR}/lib
  DOC "The GMP libraries"
  )

find_library(GMPXX_LIBRARIES
  NAMES gmpxx
  HINTS ${GMP_DIR}/lib $ENV{GMP_DIR}/lib
  DOC "The GMPXX libraries"
  )

find_library(MPFR_LIBRARIES
  NAMES mpfr
  HINTS ${GMP_DIR}/lib $ENV{GMP_DIR}/lib
  DOC "The NPFR libraries"
  )

set(GMP_LIBRARIES ${GMP_LIBRARIES} ${GMPXX_LIBRARIES} ${MPFR_LIBRARIES})
message(STATUS "GMP_LIBRARIES = ${GMP_LIBRARIES}")

mark_as_advanced(GMP_INCLUDE_DIRS GMP_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG GMP_LIBRARIES GMP_INCLUDE_DIRS)


