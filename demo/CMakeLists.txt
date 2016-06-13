project(dolfin-demos)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# FIXME: Temporary fix for whitespace error
cmake_policy(SET CMP0004 OLD)

# Find DOLFIN config file (not used here, but check that the demo will
# be able to find it
find_package(DOLFIN PATHS ${CMAKE_BINARY_DIR} REQUIRED)

# If config file is found, add all demo sub-directories, else print
# helper message
if (DOLFIN_FOUND)

  # Build list of all cpp demo directories
  file(GLOB_RECURSE list "main.cpp")
  list(SORT list)
  string(REGEX REPLACE "/main.cpp" "" list "${list}")

  # Add each C++ demo directory
  foreach (demo_dir ${list})
    add_subdirectory(${demo_dir})
  endforeach()

else()

  message("Could not locate DOLFINConfig.cmake file. Did you do 'make install' for the DOLFIN library and set the appropriate paths (source <build_dir>/dolfin.conf)?")

endif()
