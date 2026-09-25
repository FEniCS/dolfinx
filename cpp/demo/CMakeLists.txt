cmake_minimum_required(VERSION 3.26)

project(dolfinx-demos LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  include(CTest) # enables testing
endif()

# Populate PETSC_FOUND (checked below), same as each demo subdirectory
# would do on its own first.
if(NOT TARGET dolfinx::dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

# Add a demo sub-directory, if it exists. Some are skipped because a demo
# may not build in both real and complex modes, in which case it creates
# no target.
macro(add_demo_subdirectory subdir)
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${subdir})
    add_subdirectory(${subdir})
    if(TARGET demo_${subdir})
      list(APPEND DEMO_TARGETS demo_${subdir})
    endif()
  endif()
endmacro()

# Add demos. biharmonic, codim_0_assembly, hyperelasticity, mixed_poisson,
# poisson and poisson_matrix_free use PETSc directly in their source, so
# are only buildable when PETSc was found.
if(PETSC_FOUND)
  add_demo_subdirectory(biharmonic)
  add_demo_subdirectory(codim_0_assembly)
  add_demo_subdirectory(hyperelasticity)
  add_demo_subdirectory(mixed_poisson)
  add_demo_subdirectory(poisson)
  add_demo_subdirectory(poisson_matrix_free)
endif()
add_demo_subdirectory(custom_kernel)
add_demo_subdirectory(interpolation-io)
add_demo_subdirectory(interpolation_different_meshes)

# Target to build all demos, e.g. `ninja demos`
add_custom_target(demos DEPENDS ${DEMO_TARGETS})
