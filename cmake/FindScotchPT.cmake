# - Try to find SCOTCH
# Once done this will define
#
#  SCOTCH_FOUND        - system has found MTL4
#  SCOTCH_INCLUDE_DIR  - the SCOTCH include directory
#  SCOTCH_LIBARIES     - the SCOTCH libararies

set(ScotchPT_FOUND 0)

message(STATUS "Checking for package 'SCOTCH-PT'")

# Check for header file
find_path(SCOTCH_INCLUDE_DIR ptscotch.h
  $ENV{SCOTCH_DIR}/include
  /usr/local/include
  /usr/local/include/scotch
  /usr/include
  /usr/include/scotch
  DOC "Directory where the SCOTCH-PT header is located"
  )

# Check for library
find_library(SCOTCH_LIBRARIES
  NAMES scotch
  HINTS $ENV{SCOTCH_DIR}/lib
  PATHS /usr/local /opt/local /sw
  PATH_SUFFIXES lib lib64
  DOC "The SCOTCH library"
  )

# Check for scotcherr
if(SCOTCH_LIBRARIES)
  find_library(SCOTCERR_LIBRARY
    NAMES scotcherr
    HINTS $ENV{SCOTCH_DIR}/lib
    PATHS /usr/local /opt/local /sw
    PATH_SUFFIXES lib lib64
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
    HINTS $ENV{SCOTCH_DIR}/lib
    PATHS /usr/local /opt/local /sw
    PATH_SUFFIXES lib lib64
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
    HINTS $ENV{SCOTCH_DIR}/lib
    PATHS /usr/local /opt/local /sw
    PATH_SUFFIXES lib lib64
    DOC "The PTSCOTCH-ERROR library"
    )

  if (PTSCOTCERR_LIBRARY)
    set(SCOTCH_LIBRARIES ${SCOTCH_LIBRARIES} ${PTSCOTCERR_LIBRARY})
  else (PTSCOTCERR_LIBRARY)
    set(SCOTCH_LIBRARIES FALSE)
  endif (PTSCOTCERR_LIBRARY)

endif(SCOTCH_LIBRARIES)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SCOTCH DEFAULT_MSG
                                  SCOTCH_INCLUDE_DIR SCOTCH_LIBRARIES)

