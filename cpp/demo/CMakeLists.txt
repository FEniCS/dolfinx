cmake_minimum_required(VERSION 3.16)
project(dolfinx-demos)

# Find DOLFINx config file
find_package(DOLFINX REQUIRED)

# Enable testing
enable_testing()

# Macro to add demos. Some subdirectories might be skipped because demos may
# not be running in both real and complex modes.
macro(add_demo_subdirectory subdir)
  if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    add_subdirectory(${subdir})
  endif()
endmacro(add_demo_subdirectory)

# Add demos
add_demo_subdirectory(poisson)
add_demo_subdirectory(poisson_matrix_free)
add_demo_subdirectory(hyperelasticity)
add_demo_subdirectory(interpolation-io)
add_demo_subdirectory(interpolation_different_meshes)

