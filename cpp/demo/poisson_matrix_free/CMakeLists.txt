cmake_minimum_required(VERSION 3.26)

project(demo_poisson_matrix_free LANGUAGES C CXX)

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

dolfinx_add_demo(poisson_matrix_free UFL poisson.py)
