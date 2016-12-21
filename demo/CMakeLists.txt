project(dolfin-demos)

# Require CMake 2.8
cmake_minimum_required(VERSION 3.5)

# FIXME: Temporary fix for whitespace error
cmake_policy(SET CMP0004 OLD)

# Find DOLFIN config file (not used here, but check that the demo will
# be able to find it
find_package(DOLFIN PATHS ${CMAKE_BINARY_DIR} REQUIRED)

# If config file is found, add all demo sub-directories, else print
# helper message
if (DOLFIN_FOUND)

  # Directories that are commented out correspond to demos that have
  # not been implemented in C++

  # Documented demos
  add_subdirectory(documented/auto-adaptive-poisson/cpp)
  add_subdirectory(documented/bcs/cpp)
  add_subdirectory(documented/biharmonic/cpp)
  add_subdirectory(documented/built-in-meshes/cpp)
  add_subdirectory(documented/cahn-hilliard/cpp)
  add_subdirectory(documented/eigenvalue/cpp)
  add_subdirectory(documented/hyperelasticity/cpp)
  add_subdirectory(documented/mixed-poisson/cpp)
  add_subdirectory(documented/navier-stokes/cpp)
  add_subdirectory(documented/neumann-poisson/cpp)
  add_subdirectory(documented/nonlinear-poisson/cpp)
  add_subdirectory(documented/nonmatching-interpolation/cpp)
  add_subdirectory(documented/periodic/cpp)
  add_subdirectory(documented/poisson/cpp)
  add_subdirectory(documented/singular-poisson/cpp)
  #add_subdirectory(documented/singular-poisson-rst/cpp)
  add_subdirectory(documented/stokes-iterative/cpp)
  #add_subdirectory(documented/stokes-mini/cpp)
  add_subdirectory(documented/stokes-taylor-hood/cpp)
  add_subdirectory(documented/subdomains/cpp)
  #add_subdirectory(documented/subdomains-poisson/cpp)
  #add_subdirectory(documented/tensor-weighted-poisson/cpp)

  # Undocumented demos
  #add_subdirectory(undocumented/adaptive-poisson/cpp)
  add_subdirectory(undocumented/ale/cpp)
  add_subdirectory(undocumented/advection-diffusion/cpp)
  add_subdirectory(undocumented/auto-adaptive-navier-stokes/cpp)
  add_subdirectory(undocumented/block-matrix/cpp)
  #add_subdirectory(undocumented/buckling-tao/cpp)
  #add_subdirectory(undocumented/collision-detection/cpp)
  #add_subdirectory(undocumented/compiled-extension-module/cpp)
  add_subdirectory(undocumented/conditional/cpp)
  add_subdirectory(undocumented/contact-vi-snes/cpp)
  add_subdirectory(undocumented/contact-vi-tao/cpp)
  #add_subdirectory(undocumented/coordinates/cpp)
  add_subdirectory(undocumented/curl-curl/cpp)
  add_subdirectory(undocumented/dg-advection-diffusion/cpp)
  add_subdirectory(undocumented/dg-poisson/cpp)
  add_subdirectory(undocumented/elasticity/cpp)
  add_subdirectory(undocumented/elastodynamics/cpp)
  #add_subdirectory(undocumented/elementplot/cpp)
  add_subdirectory(undocumented/eval/cpp)
  add_subdirectory(undocumented/extrapolation/cpp)
  add_subdirectory(undocumented/functional/cpp)
  #add_subdirectory(undocumented/ghost-mesh/cpp)
   add_subdirectory(undocumented/gmg-poisson/cpp)
  #add_subdirectory(undocumented/interpolation/cpp)
  add_subdirectory(undocumented/lift-drag/cpp)
  add_subdirectory(undocumented/mesh-quality/cpp)
  #add_subdirectory(undocumented/mixed-poisson-sphere/cpp)
  #add_subdirectory(undocumented/mplot/cpp)
  #add_subdirectory(undocumented/multimesh/cpp)
  add_subdirectory(undocumented/multimesh-poisson/cpp)
  add_subdirectory(undocumented/multimesh-stokes/cpp)
  add_subdirectory(undocumented/meshfunction-refinement/cpp)
  #add_subdirectory(undocumented/multistage-solver/cpp)
  add_subdirectory(undocumented/nonmatching-projection/cpp)
  #add_subdirectory(undocumented/optimization/cpp)
  #add_subdirectory(undocumented/overlapping-regions/cpp)
  add_subdirectory(undocumented/parallel-refinement/cpp)
  add_subdirectory(undocumented/parameters/cpp)
  add_subdirectory(undocumented/plot/cpp)
  add_subdirectory(undocumented/poisson1D/cpp)
  add_subdirectory(undocumented/poisson1D-in-2D/cpp)
  add_subdirectory(undocumented/poisson-disc/cpp)
  #add_subdirectory(undocumented/point-integral/cpp)
  #add_subdirectory(undocumented/projection-interpolation/cpp)
  add_subdirectory(undocumented/refinement/cpp)
  #add_subdirectory(undocumented/smoothing/cpp)
  add_subdirectory(undocumented/spatial-coordinates/cpp)
  #add_subdirectory(undocumented/special-functions/cpp)
  #add_subdirectory(undocumented/sub-function-assignment/cpp)
  add_subdirectory(undocumented/submesh/cpp)
  add_subdirectory(undocumented/sym-dirichlet-bc/cpp)
  add_subdirectory(undocumented/time-series/cpp)
  #add_subdirectory(undocumented/timing/cpp)
  add_subdirectory(undocumented/waveguide/cpp)

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
