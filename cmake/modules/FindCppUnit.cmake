# - Try to find CPPUNIT
# Once done this will define
#
#  CPPUNIT_FOUND        - system has CPPUNIT
#  CPPUNIT_INCLUDE_DIRS - include directories for CPPUNIT
#  CPPUNIT_LIBRARIES    - libraries for CPPUNIT
#  CPPUNIT_VERSION      - CPPUNIT version string (MAJOR.MINOR.MICRO)

#=============================================================================
# Copyright (C) 2010-2012 Garth N. Wells, Anders Logg and Johannes Ring
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

include(FindPkgConfig)
pkg_check_modules(PC_CPPUNIT cppunit)

set(CPPUNIT_VERSION ${PC_CPPUNIT_VERSION})

# Check for header file
find_path(CPPUNIT_INCLUDE_DIRS cppunit/Test.h
 HINTS ${PC_CPPUNIT_INCLUDEDIR} ${CPPUNIT_DIR}/include $ENV{CPPUNIT_DIR}/include
 DOC "Directory where the CPPUNIT header is located"
 )

# Check for library
find_library(CPPUNIT_LIBRARIES cppunit
  HINTS ${PC_CPPUNIT_LIBDIR} ${CPPUNIT_DIR}/lib $ENV{CPPUNIT_DIR}/lib
  DOC "The CPPUNIT library"
  )

mark_as_advanced(
  CPPUNIT_LIBRARIES
  CPPUNIT_INCLUDE_DIRS
  CPPUNIT_VERSION
  )

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CPPUNIT
  "CPPUNIT could not be found. Be sure to set CPPUNIT_DIR."
  CPPUNIT_LIBRARIES CPPUNIT_INCLUDE_DIRS)
