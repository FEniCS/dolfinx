# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# Set verbose output while testing CMake
set(CMAKE_VERBOSE_MAKEFILE 1)

# Find DOLFIN config file (not used here, but check that the demo will be able
# to find it
find_package(dolfin REQUIRED)

# If config file is found, add all demo sub-directories, else print helper
# message
if (dolfin_FOUND)

  # Build list of all cpp demo directories
  file( GLOB_RECURSE list "main.cpp")
  list( SORT list )
  string(REGEX REPLACE "/main.cpp" "" list "${list}")

  # Add each C++ demo directory
  foreach( demo_dir ${list} )
    message(STATUS "Add C++ demo: ${demo_dir}")
    add_subdirectory( ${demo_dir} )
  endforeach()

else()

  message(STATUS "Could not locate dolfin-config.cmake file. Did you do 'make install' for the DOLFIN library and set the appropriate paths (source <build_dir>/dolfin.conf)?")

endif()


