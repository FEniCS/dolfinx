# - Try to find Trilinos
# Once done this will define
#
#  TRILINOS_FOUND        - system has CGAL
#  TRILINOS_INCLUDE_DIRS - include directories for CGAL
#  TRILINOS_LIBRARIES    - libraries for CGAL
#  TRILINOS_DEFINITIONS  - compiler flags for CGAL

message(STATUS "Checking for package 'Trilinos'")

# Find Trilinos CMake config
find_package(Trilinos PATHS ${UMFPACK_DIR}/include $ENV{TRILINOS_DIR}/include QUIET)

if (Trilinos_FOUND)

  # Trilinos found
  set(TRILINOS_FOUND true)

  # Get Trilinos include directories
  set(TRILINOS_INCLUDE_DIRS ${Trilinos_INCLUDE_DIRS})

  # Trilinos definitons
  set(TRILINOS_DEFINITIONS)

  # Loop over Trilinos libs and get full path
  foreach (lib ${Trilinos_LIBRARIES})
    find_library(TRILINOS_LIB_${lib} ${lib} PATHS ${Trilinos_LIBRARY_DIRS})
    if (TRILINOS_LIB_${lib})
      list(APPEND TRILINOS_LIBRARIES ${TRILINOS_LIB_${lib}})
    endif()
  endforeach()

endif()
