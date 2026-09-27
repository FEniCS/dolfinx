cmake_minimum_required(VERSION 3.26)

project(demo_custom_kernel LANGUAGES C CXX)

if(NOT TARGET dolfinx::dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

if(PROJECT_IS_TOP_LEVEL)
  include(CTest) # enables testing
endif()

if(DOLFINX_DIR)
  list(APPEND CMAKE_MODULE_PATH ${DOLFINX_DIR})
endif()
include(DolfinxDemo)
if(DOLFINX_DIR)
  list(REMOVE_ITEM CMAKE_MODULE_PATH ${DOLFINX_DIR})
endif()

# Compile mass.py by hand rather than via dolfinx_add_demo's UFL option,
# which compiles the generated C as a separate source. Here main.cpp
# includes mass.c so that the kernels can be inlined, and the float64
# kernels are used only by the double instantiation, so the generated
# scalar type must not follow PETSc.
add_custom_command(
  OUTPUT mass.c mass.h
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/mass.py
  VERBATIM
  DEPENDS mass.py
  COMMENT "Compile mass.py using FFCx"
)
set_source_files_properties(
  main.cpp
  PROPERTIES OBJECT_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/mass.c"
)

dolfinx_add_demo(custom_kernel)
