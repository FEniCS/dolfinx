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
  PATHS $ENV{SCOTCH_DIR}/include
  PATH_SUFFIXES scotch
  DOC "Directory where the SCOTCH-PT header is located"
  )

# Check for library
find_library(SCOTCH_LIBRARIES
  NAMES scotch
  PATHS $ENV{SCOTCH_DIR}/lib
  DOC "The SCOTCH library"
  )

# Check for scotcherr
if(SCOTCH_LIBRARIES)
  find_library(SCOTCERR_LIBRARY
    NAMES scotcherr
    PATHS $ENV{SCOTCH_DIR}/lib
    DOC "The SCOTCH-ERROR library"
    )

  if (SCOTCERR_LIBRARY)
    set(SCOTCH_LIBRARIES ${SCOTCH_LIBRARIES} ${SCOTCERR_LIBRARY})
  else (SCOTCERR_LIBRARY)
    set(SCOTCH_LIBRARIES FALSE)
  endif (SCOTCERR_LIBRARY)

endif(SCOTCH_LIBRARIES)

# Check for ptscotch
if(SCOTCH_LIBRARIES)
  find_library(PTSCOTCH_LIBRARY
    NAMES ptscotch
    PATHS $ENV{SCOTCH_DIR}/lib
    DOC "The PTSCOTCH library"
    )

  if (PTSCOTCH_LIBRARY)
    set(SCOTCH_LIBRARIES ${SCOTCH_LIBRARIES} ${PTSCOTCH_LIBRARY})
  else (PTSCOTCH_LIBRARY)
    set(SCOTCH_LIBRARIES FALSE)
  endif (PTSCOTCH_LIBRARY)

endif(SCOTCH_LIBRARIES)

# Check for ptscotcherr
if(SCOTCH_LIBRARIES)
  find_library(PTSCOTCERR_LIBRARY
    NAMES ptscotcherr
    PATHS $ENV{SCOTCH_DIR}/lib
    DOC "The PTSCOTCH-ERROR library"
    )

  if (PTSCOTCERR_LIBRARY)
    set(SCOTCH_LIBRARIES ${SCOTCH_LIBRARIES} ${PTSCOTCERR_LIBRARY})
  else (PTSCOTCERR_LIBRARY)
    set(SCOTCH_LIBRARIES FALSE)
  endif (PTSCOTCERR_LIBRARY)

endif(SCOTCH_LIBRARIES)

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCOTCH DEFAULT_MSG
                                  SCOTCH_INCLUDE_DIRS SCOTCH_LIBRARIES)
