# - Try to find LAPACK
# Once done this will define
#
#  LAPACK_FOUND        - system has LAPACK
#  LAPACK_INCLUDE_DIRS - include directories for LAPACK
#  LAPACK_LIBRARIES    - libraries for LAPACK
#
# This test is necessary since the CMake supplied FindLAPACK script
# requires Fortran (Fortran which may not be installed).
#

message(STATUS "Checking for package 'LAPACK'")

# Check for header file (this is not always available)
#find_path(LAPACK_INCLUDE_DIRS clapack.h
# PATHS ${LAPACK_DIR}/include $ENV{LAPACK_DIR}/include
#  PATH_SUFFIXES atlas
# DOC "Directory where the LAPACK header is located"
# )
#mark_as_advanced(LAPACK_INCLUDE_DIRS)

# Check for library
find_library(LAPACK_LIBRARIES
  NAMES lapack
  HINTS ${LAPACK_DIR}/lib $ENV{LAPACK_DIR}/lib
  DOC "The LAPACK library"
  )
mark_as_advanced(LAPACK_LIBRARIES)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACK
  "LAPACK could not be found. Be sure to set LAPACK_DIR."
  LAPACK_LIBRARIES)
