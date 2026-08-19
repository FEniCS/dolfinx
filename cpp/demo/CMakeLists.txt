cmake_minimum_required(VERSION 3.21)

project(dolfinx-demos LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  include(CTest) # enables testing
  find_package(DOLFINX REQUIRED)
else()
  # Required to find version.h, propagates to the demo subdirectories.
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/../)
endif()

# Macro to add demos. Some subdirectories might be skipped because demos may not
# be running in both real and complex modes.
macro(add_demo_subdirectory subdir)
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    add_subdirectory(${subdir})
    list(APPEND DEMO_TARGETS demo_${subdir})
  endif()
endmacro(add_demo_subdirectory)

# The JIT demo runs FFCx and a C compiler when it is executed, rather than
# when it is built. It is off by default because it makes the C++ demos
# depend on a Python environment at run time, and because it uses dlopen.
option(
  DOLFINX_ENABLE_JIT_DEMO
  "Build the demo that compiles UFL forms at run time (POSIX only)."
  OFF
)

# Add demos
add_demo_subdirectory(biharmonic)
add_demo_subdirectory(codim_0_assembly)
add_demo_subdirectory(custom_kernel)
add_demo_subdirectory(hyperelasticity)
add_demo_subdirectory(interpolation-io)
add_demo_subdirectory(interpolation_different_meshes)
add_demo_subdirectory(mixed_poisson)
add_demo_subdirectory(poisson)
add_demo_subdirectory(poisson_matrix_free)
if(DOLFINX_ENABLE_JIT_DEMO AND NOT WIN32)
  add_demo_subdirectory(poisson_jit)
endif()

# Target to build all demos, e.g. `ninja demos`
add_custom_target(demos DEPENDS ${DEMO_TARGETS})
