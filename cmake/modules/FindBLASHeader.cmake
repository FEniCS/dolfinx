# - Try to find BLAS header cblas.h
# Once done this will define
#
#  BLASHEADER_FOUND  - system has BLAS
#  BLAS_INCLUDE_DIRS - include directories for BLAS
#

message(STATUS "Checking for package 'BLAS'")

# Check for header file
find_path(BLASHEADER_INCLUDE_DIRS cblas.h
 PATHS ${BLASHEADER_DIR}/include $ENV{BLASHEADER_DIR}/include
 DOC "Directory where the BLAS header is located"
 )
mark_as_advanced(BLASBLASHEADER_INCLUDE_DIRS)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BLASHEADER
  "BLAS C header could not be found. Be sure to set BLASHEADER_DIR."
  BLASHEADER_INCLUDE_DIRS)
