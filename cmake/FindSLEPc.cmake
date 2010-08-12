# - Try to find SLEPC
# Once done this will define
#
#  SLEPC_FOUND        - system has found SLEPc
#  SLEPC_INCLUDE_DIR  - the SLEPc include directory
#  SLEPC_LIBARIES     - the SLEPc libararies


# Check for header file
find_path (SLEPC_INCLUDE_DIR slepc.h
  HINTS ENV SLEPC_DIR
  PATHS
  /usr/local
  /usr/
  PATH_SUFFIXES include
  DOC "SLEPc include path"
  )
mark_as_advanced(SLEPC_INCLUDE_DIR)

# FIXME: This is a major hack. Need to create a Makefile (like for PETSc)
#        and extract build data

# Check for library
find_library(SLEPC_LIBRARIES
  NAMES slepc
  HINTS $ENV{SLEPC_DIR}
  PATHS /usr/local /opt/local /sw
  PATH_SUFFIXES lib lib64 linux-gnu-cxx-opt/lib linux-gnu-cxx-debug/lib
  DOC "The SLEPC library"
  )
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SLEPC DEFAULT_MSG
                                  SLEPC_INCLUDE_DIR SLEPC_LIBRARIES)

