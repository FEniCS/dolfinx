# - Try to find CPPUNIT
# Once done this will define
#
#  CPPUNIT_FOUND        - system has CPPUNIT
#  CPPUNIT_INCLUDE_DIRS - include directories for CPPUNIT
#  CPPUNIT_LIBRARIES    - libraries for CPPUNIT
#  CPPUNIT_VERSION      - CPPUNIT version string (MAJOR.MINOR.MICRO)

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
