# - Try to find AMD
# Once done this will define
#
#  AMD_FOUND        - system has AMD
#  AMD_INCLUDE_DIRS - include directories for AMD
#  AMD_LIBRARIES    - libraries for AMD

message(STATUS "Checking for package 'AMD'")

# Check for header file
find_path(AMD_INCLUDE_DIRS amd.h
 PATHS $ENV{AMD_DIR}/include
 PATH_SUFFIXES suitesparse ufsparse
 DOC "Directory where the AMD header is located"
 )
mark_as_advanced(AMD_INCLUDE_DIRS)

# Check for library
find_library(AMD_LIBRARIES amd
  PATHS $ENV{AMD_DIR}/lib
  DOC "The AMD library"
  )
mark_as_advanced(AMD_LIBRARIES)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMD
  "AMD could not be found. Be sure to set AMD_DIR."
  AMD_INCLUDE_DIRS AMD_LIBRARIES)
