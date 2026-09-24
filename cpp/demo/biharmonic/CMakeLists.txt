cmake_minimum_required(VERSION 3.26)

project(demo_biharmonic LANGUAGES C CXX)

if(NOT TARGET dolfinx::dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

if(PROJECT_IS_TOP_LEVEL)
  include(CTest) # enables testing
endif()

list(APPEND CMAKE_MODULE_PATH ${DOLFINX_DIR})
include(DolfinxDemo)

dolfinx_add_demo(biharmonic UFL biharmonic.py)
