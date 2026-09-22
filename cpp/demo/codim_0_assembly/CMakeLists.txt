cmake_minimum_required(VERSION 3.26)

project(demo_codim_0_assembly LANGUAGES C CXX)

if(NOT TARGET dolfinx::dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

if(PROJECT_IS_TOP_LEVEL)
  include(CTest) # enables testing
endif()

# Included by path, as putting DOLFINX_DIR on CMAKE_MODULE_PATH lets the
# installed FindUFCx.cmake shadow the ufcx CONFIG package on
# case-insensitive filesystems.
if(DOLFINX_DIR)
  include("${DOLFINX_DIR}/DolfinxDemo.cmake")
else()
  include(DolfinxDemo)
endif()

dolfinx_add_demo(codim_0_assembly UFL mixed_codim0.py)
