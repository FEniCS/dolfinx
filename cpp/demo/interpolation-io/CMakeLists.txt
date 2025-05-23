# This file was generated by running
#
# python cmake/scripts/generate-cmakefiles.py from dolfinx/cpp
#
cmake_minimum_required(VERSION 3.21)

set(PROJECT_NAME demo_interpolation-io)
project(${PROJECT_NAME} LANGUAGES C CXX)

if(NOT TARGET dolfinx)
  find_package(DOLFINX REQUIRED)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} dolfinx)

# Set C++20 standard
set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Do not throw error for 'multi-line comments' (these are typical in rst which
# includes LaTeX)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wno-comment" HAVE_NO_MULTLINE)
set_source_files_properties(
  main.cpp
  PROPERTIES
    COMPILE_FLAGS
    "$<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment -Wall -Wextra -pedantic -Werror>"
)

# Test targets (used by DOLFINx testing system)
set(TEST_PARAMETERS2 -np 2 ${MPIEXEC_PARAMS} "./${PROJECT_NAME}")
set(TEST_PARAMETERS3 -np 3 ${MPIEXEC_PARAMS} "./${PROJECT_NAME}")
add_test(NAME ${PROJECT_NAME}_mpi_2 COMMAND "mpirun" ${TEST_PARAMETERS2})
add_test(NAME ${PROJECT_NAME}_mpi_3 COMMAND "mpirun" ${TEST_PARAMETERS3})
add_test(NAME ${PROJECT_NAME}_serial COMMAND ${PROJECT_NAME})
