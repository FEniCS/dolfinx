# - Try to find SuperLU-dist
# Once done this will define
# SuperLUDist_FOUND - System has SuperLU-dist
# SuperLUDist_INCLUDE_DIRS - The SuperLU-dist include directories
# SuperLUDist_LIBRARY_DIRS - The library directories needed to use SuperLU-dist
# SuperLUDist_LIBRARIES    - The libraries needed to use SuperLU-dist

find_package(PkgConfig REQUIRED)
pkg_check_modules(SuperLUDist superlu_dist)

if(NOT SuperLUDist_FOUND)
  find_path(SuperLUDist_INCLUDE_DIR NAMES superlu_dist_config.h
    PATHS "${PETSC_INCLUDE_DIRS}/"
    "/usr/include/"
  )

  find_library(SuperLUDist_LIBRARY
    superlu_dist
    PATHS "${PETSC_LIBRARY_DIRS}/lib/"
    "/usr/lib")
endif()

include(FindPackageHandleStandardArgs)

# handle the QUIETLY and REQUIRED arguments and set FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(SuperLUDist DEFAULT_MSG
  SuperLUDist_LIBRARY SuperLUDist_INCLUDE_DIR)

mark_as_advanced(SuperLUDist_INCLUDE_DIR SuperLUDist_LIBRARY)
