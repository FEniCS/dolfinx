cmake_minimum_required(VERSION 3.9)
project(dolfin-demos)

# Find DOLFIN config file (not used here, but check that the demo will
# be able to find it
find_package(DOLFIN REQUIRED)
# include(${DOLFIN_USE_FILE})

# Demos that run in real and complex modes
add_subdirectory(poisson)

# Demos that run only in real mode
if (NOT PETSC_SCALAR_COMPLEX)
  add_subdirectory(hyperelasticity)
endif()

# Demos that run only in complex mode
if (PETSC_SCALAR_COMPLEX)
  # Add here
endif()

# FIXME: do this programatically. Can we get a list of all targets?
if (NOT PETSC_SCALAR_COMPLEX)
  add_custom_target(demos DEPENDS demo_poisson demo_hyperelasticity)
else()
  add_custom_target(demos DEPENDS demo_poisson)
endif()