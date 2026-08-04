cmake_minimum_required(VERSION 3.21)

project(dolfinx-tests LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  include(CTest) # enables testing
  find_package(DOLFINX REQUIRED)
  include("${DOLFINX_DIR}/DolfinxDeveloperCompilerFlags.cmake")
endif()

find_package(Catch2 3 REQUIRED)

# FFCx compile: expr.py -> expr.c
add_custom_command(
  OUTPUT expr.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/fem/expr.py
  VERBATIM
  DEPENDS fem/expr.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Compile expr.py using FFCx"
)

# FFCx compile: poisson.py -> poisson.c
add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Compile poisson.py using FFCx"
)

add_executable(
  unittests
  main.cpp
  graph.cpp
  vector.cpp
  matrix.cpp
  io.cpp
  common/CIFailure.cpp
  common/sub_systems_manager.cpp
  common/index_map.cpp
  common/sort.cpp
  fem/form.cpp
  fem/functionspace.cpp
  mesh/branching_manifold.cpp
  mesh/distributed_mesh.cpp
  mesh/generation.cpp
  mesh/read_named_meshtags.cpp
  mesh/refinement/interval.cpp
  mesh/refinement/option.cpp
  mesh/refinement/rectangle.cpp
  mesh/refinement/mark.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/expr.c
  ${CMAKE_CURRENT_BINARY_DIR}/poisson.c
)
target_link_libraries(unittests PRIVATE Catch2::Catch2 dolfinx::dolfinx)
target_compile_features(unittests PRIVATE cxx_std_20 c_std_17)
set_target_properties(unittests PROPERTIES CXX_EXTENSIONS OFF)

# NOTE: Fixes when using Catch2 <= 3.13 with Clang 22 and derivatives.
# https://github.com/catchorg/Catch2/blob/devel/docs/release-notes.md#3140
if(
  CMAKE_CXX_COMPILER_ID MATCHES "Clang"
  OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"
)
  target_compile_options(unittests PRIVATE -Wno-c2y-extensions)
endif()

# UUID requires bcrypt to be linked on Windows, broken in vcpkg.
# https://github.com/microsoft/vcpkg/issues/4481
if(WIN32)
  target_link_libraries(unittests PRIVATE bcrypt)
endif()

# Required to find FFCx generated headers.
target_include_directories(unittests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

if(NOT PROJECT_IS_TOP_LEVEL)
  # Required to find version.h
  target_include_directories(unittests PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../)
endif()

if(DEFINED DOLFINX_CXX_DEVELOPER_FLAGS)
  target_compile_options(
    unittests
    PRIVATE
      $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
  )
endif()

if(DEFINED DOLFINX_CXX_DEVELOPER_DEFINITIONS)
  target_compile_definitions(
    unittests
    PRIVATE
      $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_DEFINITIONS}>
  )
endif()

option(ENABLE_CLANG_TIDY "Enable clang-tidy analysis" OFF)
if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(
    unittests
    PROPERTIES
      CXX_CLANG_TIDY
        "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../../.clang-tidy"
  )
endif()

# Uncommenting line allows for debugging individual tests, but slow.
#catch_discover_tests(unittests)

# To select one np use, e.g.:
# ctest -R unittests_np_3
foreach(N 1 2 3)
  add_test(
    NAME unittests_np_${N}
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${N} ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:unittests> ${MPIEXEC_POSTFLAGS}
  )
  # This tells ctest how many physical cores (i.e., ${N}) are needed for each
  # test, preventing oversubscription when using e.g. `ctest -j2`.
  set_tests_properties(unittests_np_${N} PROPERTIES PROCESSORS ${N})
endforeach()
