project(dolfin-demos)

# Require CMake 2.8
cmake_minimum_required(VERSION 3.5)

# Find DOLFIN config file (not used here, but check that the demo will
# be able to find it
#find_package(DOLFIN PATHS ${CMAKE_BINARY_DIR} REQUIRED)

# Documented demos
add_subdirectory(documented/biharmonic)
add_subdirectory(documented/cahn-hilliard)
add_subdirectory(documented/eigenvalue)
add_subdirectory(documented/hyperelasticity)
add_subdirectory(documented/mixed-poisson)
add_subdirectory(documented/navier-stokes)
add_subdirectory(documented/neumann-poisson)
add_subdirectory(documented/nonlinear-poisson)
add_subdirectory(documented/nonmatching-interpolation)
add_subdirectory(documented/periodic)
add_subdirectory(documented/poisson)
add_subdirectory(documented/singular-poisson)
add_subdirectory(documented/stokes-iterative)
add_subdirectory(documented/stokes-taylor-hood)

# Undocumented demos
add_subdirectory(undocumented/advection-diffusion)
add_subdirectory(undocumented/conditional)
add_subdirectory(undocumented/contact-vi-snes)
add_subdirectory(undocumented/contact-vi-tao)
add_subdirectory(undocumented/curl-curl)
add_subdirectory(undocumented/dg-advection-diffusion)
add_subdirectory(undocumented/elasticity)
add_subdirectory(undocumented/elastodynamics)
add_subdirectory(undocumented/eval)
add_subdirectory(undocumented/gmg-poisson)
add_subdirectory(undocumented/nonmatching-projection)
add_subdirectory(undocumented/poisson1D)
add_subdirectory(undocumented/poisson1D-in-2D)
add_subdirectory(undocumented/poisson-disc)
