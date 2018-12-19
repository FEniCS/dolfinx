cmake_minimum_required(VERSION 3.5)
project(dolfin-demos)

# Find DOLFIN config file (not used here, but check that the demo will
# be able to find it
find_package(DOLFIN REQUIRED)
# include(${DOLFIN_USE_FILE})

# Documented demos
add_subdirectory(documented/hyperelasticity)
add_subdirectory(documented/poisson)

# Undocumented demos
