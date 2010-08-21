# - Try to find CGAL
# Once done this will define
#
#  CGAL_FOUND    - system has CGAL
#  CGAL_USE_FILE - CMake file to use CGAL
#
# This test calls the standard CGAL test but in addition checks for
# MPFR which is needed by CGAL.

message("CHECKING FOR CGAL")

# Check for header file
find_path(MPFR_INCLUDE_DIR mpfr.h
  $ENV{MPFR_DIR}
  /usr/local/include
  /usr/include
  DOC "Directory where the MPFR header is located"
  )

# Call standard CGAL test
find_package(CGAL PATHS $ENV{CGAL_DIR}/lib QUIET)

# Standard package handling
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CGAL "CGAL could not be found. Be sure to set CGAL_DIR and MPFR_DIR."
                                  CGAL_FOUND MPFR_INCLUDE_DIR)
