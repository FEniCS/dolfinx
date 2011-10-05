# - Try to find Trilinos
# Once done this will define
#
#  TRILINOS_FOUND        - system has Trilinos
#  TRILINOS_INCLUDE_DIRS - include directories for Trilinos
#  TRILINOS_LIBRARIES    - libraries for Trilinos
#  TRILINOS_DEFINITIONS  - compiler flags for Trilinos
#  TRILINOS_VERSION      - Trilinos version

message(STATUS "Checking for package 'Trilinos'")

# Find Trilinos CMake config
find_package(Trilinos
  HINTS ${TRILINOS_DIR} ${Trilinos_DIR} $ENV{TRILINOS_DIR} ${Trilinos_DIR}/include ${TRILINOS_DIR}/include $ENV{TRILINOS_DIR}/include
  PATHS /usr/include/trilinos
  QUIET)

if (Trilinos_FOUND)

  # Trilinos found
  set(TRILINOS_FOUND true)

  # Get Trilinos include directories
  set(TRILINOS_INCLUDE_DIRS ${Trilinos_INCLUDE_DIRS})

  # Trilinos definitons
  set(TRILINOS_DEFINITIONS)

  # Trilinos version
  set(TRILINOS_VERSION ${Trilinos_VERSION})

  # Loop over Trilinos libs and get full path
  foreach (lib ${Trilinos_LIBRARIES})
    find_library(TRILINOS_LIB_${lib} ${lib} HINTS ${Trilinos_LIBRARY_DIRS})
    if (TRILINOS_LIB_${lib} AND NOT ${lib} MATCHES ".*pytrilinos")
      set(TRILINOS_LIBRARIES ${TRILINOS_LIBRARIES} ${TRILINOS_LIB_${lib}})
    endif()
  endforeach()

  message(STATUS "Found Trilinos (found version ${TRILINOS_VERSION})")

endif()
