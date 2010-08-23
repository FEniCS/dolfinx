# - Try to find CHOLMOD
# Once done this will define
#
#  CHOLMOD_FOUND        - system has CHOLMOD
#  CHOLMOD_INCLUDE_DIRS - include directories for CHOLMOD
#  CHOLMOD_LIBRARIES    - libraries for CHOLMOD

message(STATUS "Checking for package 'CHOLMOD'")

# Check for header file
find_path(CHOLMOD_INCLUDE_DIRS cholmod.h
 PATHS $ENV{CHOLMOD_DIR}/include
 PATH_SUFFIXES suitesparse ufsparse
 DOC "Directory where the CHOLMOD header is located"
 )
mark_as_advanced(CHOLMOD_INCLUDE_DIRS)

# Check for CHOLMOD library
find_library(CHOLMOD_LIBRARY cholmod
  PATHS $ENV{CHOLMOD_DIR}/lib
  DOC "The CHOLMOD library"
  )
mark_as_advanced(CHOLMOD_LIBRARY)

# Collect libraries
set(CHOLMOD_LIBRARIES "${CHOLMOD_LIBRARY};${AMD_LIBRARIES}")

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CHOLMOD
  "CHOLMOD could not be found. Be sure to set CHOLMOD_DIR."
  CHOLMOD_INCLUDE_DIRS CHOLMOD_LIBRARIES AMD_LIBRARIES)
