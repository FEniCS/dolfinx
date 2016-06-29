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

  # Documented demos
  add_subdirectory(documented/auto-adaptive-poisson/cpp)
  add_subdirectory(documented/bcs/cpp)
  add_subdirectory(documented/biharmonic/cpp)
  add_subdirectory(documented/built-in-meshes/cpp)
  add_subdirectory(documented/cahn-hilliard/cpp)
  add_subdirectory(documented/eigenvalue/cpp)
  add_subdirectory(documented/hyperelasticity/cpp)
  add_subdirectory(documented/mixed-poisson/cpp)
  add_subdirectory(documented/mixed-poisson-dual/cpp)
  add_subdirectory(documented/navier-stokes/cpp)
  add_subdirectory(documented/neumann-poisson/cpp)
  add_subdirectory(documented/nonlinear-poisson/cpp)
  add_subdirectory(documented/periodic/cpp)
  add_subdirectory(documented/poisson/cpp)
  add_subdirectory(documented/poisson-pylit/cpp)
  add_subdirectory(documented/singular-poisson/cpp)
  #add_subdirectory(documented/singular-poisson-rst/cpp)
  add_subdirectory(documented/stokes-iterative/cpp)
  #add_subdirectory(documented/stokes-mini/cpp)
  add_subdirectory(documented/stokes-taylor-hood/cpp)
  add_subdirectory(documented/subdomains/cpp)
  #add_subdirectory(documented/subdomains-poisson/cpp)
  #add_subdirectory(documented/tensor-weighted-poisson/cpp)

  # Undocumented demos

  # Build list of all cpp demo directories
  #file(GLOB_RECURSE list "main.cpp")
  #list(SORT list)
  #string(REGEX REPLACE "/main.cpp" "" list "${list}")

  # Add each C++ demo directory
  #foreach (demo_dir ${list})
  #  add_subdirectory(${demo_dir})
  #endforeach()

else()

  message("Could not locate DOLFINConfig.cmake file. Did you do 'make install' for the DOLFIN library and set the appropriate paths (source <build_dir>/dolfin.conf)?")

endif()
