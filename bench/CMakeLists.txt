# Require CMake 2.8
cmake_minimum_required(VERSION 3.5)

# FIXME: Temporary fix for whitespace error
cmake_policy(SET CMP0004 OLD)

# Find DOLFIN config file (not used here, but check that benchmarks will be
# able to find it
find_package(DOLFIN REQUIRED)

# If config file is found, add all demo sub-directories, else print helper
# message
if (DOLFIN_FOUND)

  # Build list of all cpp directories
  file( GLOB_RECURSE list "main.cpp")
  list( SORT list )
  string(REGEX REPLACE "/main.cpp" "" list "${list}")

  # Add each C++ code directory
  foreach (cpp_dir ${list})
    #message(STATUS "Add C++ benchmark: ${cpp_dir}")
    add_subdirectory(${cpp_dir})
  endforeach()

  # Add fem/speedup/cpp manually (no main.cpp so not picked up automatically)
  add_subdirectory(fem/speedup/cpp)

else()

  message(STATUS "Could not locate DOLFINConfig.cmake file. Did you do 'make install' for the DOLFIN library and set the appropriate paths (source <build_dir>/dolfin.conf)?")

endif()
