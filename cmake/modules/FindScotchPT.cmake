# - Try to find SCOTCH
# Once done this will define
#
#  SCOTCH_FOUND        - system has found SCOTCH
#  SCOTCH_INCLUDE_DIRS - include directories for SCOTCH
#  SCOTCH_LIBARIES     - libraries for SCOTCH

set(ScotchPT_FOUND 0)

message(STATUS "Checking for package 'SCOTCH-PT'")

# Check for header file
find_path(SCOTCH_INCLUDE_DIRS ptscotch.h
  PATHS ${SCOTCH_DIR}/include $ENV{SCOTCH_DIR}/include
  PATH_SUFFIXES scotch
  DOC "Directory where the SCOTCH-PT header is located"
  )

# Check for library
#find_library(SCOTCH_LIBRARY
#  NAMES scotch
#  PATHS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
#  DOC "The SCOTCH library"
#  )

#find_library(SCOTCHERR_LIBRARY
#  NAMES scotcherr
#  PATHS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
#  DOC "The SCOTCH-ERROR library"
#  )

# Check for ptscotch
find_library(PTSCOTCH_LIBRARY
  NAMES ptscotch
  PATHS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
  DOC "The PTSCOTCH library"
  )

# Check for ptscotcherr
find_library(PTSCOTCHERR_LIBRARY
  NAMES ptscotcherr
  PATHS ${SCOTCH_DIR}/lib $ENV{SCOTCH_DIR}/lib
  DOC "The PTSCOTCH-ERROR library"
  )

set(SCOTCH_LIBRARIES ${PTSCOTCH_LIBRARY} ${PTSCOTCHERR_LIBRARY})

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCOTCH DEFAULT_MSG
                                  SCOTCH_INCLUDE_DIRS SCOTCH_LIBRARIES)
