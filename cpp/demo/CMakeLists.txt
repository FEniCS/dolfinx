project(dolfin-demos)

# Require CMake 2.8
cmake_minimum_required(VERSION 3.5)

# Find DOLFIN config file (not used here, but check that the demo will
# be able to find it
#find_package(DOLFIN PATHS ${CMAKE_BINARY_DIR} REQUIRED)

# Documented demos
add_subdirectory(documented/hyperelasticity)
add_subdirectory(documented/poisson)

# Undocumented demos
