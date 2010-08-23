# - Try to find CPPUNIT
# Once done this will define
#
#  CPPUNIT_FOUND        - system has CPPUNIT
#  CPPUNIT_INCLUDE_DIRS - include directories for CPPUNIT
#  CPPUNIT_LIBRARIES    - libraries for CPPUNIT

message(STATUS "Checking for package 'CPPUNIT'")

# Check for header file
find_path(CPPUNIT_INCLUDE_DIRS Test.h
 PATHS $ENV{CPPUNIT_DIR}/include
 PATH_SUFFIXES cppunit
 DOC "Directory where the CPPUNIT header is located"
 )
mark_as_advanced(CPPUNIT_INCLUDE_DIRS)

# Check for library
find_library(CPPUNIT_LIBRARIES cppunit
  PATHS $ENV{CPPUNIT_DIR}/lib
  DOC "The CPPUNIT library"
  )
mark_as_advanced(CPPUNIT_LIBRARIES)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CPPUNIT
  "CPPUNIT could not be found. Be sure to set CPPUNIT_DIR."
  CPPUNIT_INCLUDE_DIRS CPPUNIT_LIBRARIES)
