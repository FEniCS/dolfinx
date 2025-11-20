cmake_minimum_required(VERSION 3.21)

project(dolfinx-tests LANGUAGES C CXX)

find_package(MPI 3 REQUIRED)

if (PROJECT_IS_TOP_LEVEL)
  include(CTest) # enables testing
  find_package(DOLFINX REQUIRED)

  # NOTE: The following must precisely replicate the code in ../CMakeLists.txt.
  # Check for some compiler flags
  include(CheckCXXCompilerFlag)

  # Add some strict compiler checks
  check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)

  if(HAVE_PEDANTIC)
    list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic)
  endif()

  # Debug flags
  check_cxx_compiler_flag(-g HAVE_DEBUG)
  if(HAVE_DEBUG)
    list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS "-g")
  endif()

  # Optimisation
  check_cxx_compiler_flag(-O2 HAVE_O2_OPTIMISATION)
  if(HAVE_O2_OPTIMISATION)
    list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS "-O")
  endif()

  # Enable C++ standard library debugging
  include(CheckCXXSymbolExists)

  check_cxx_symbol_exists(_LIBCPP_VERSION "version" LIBCPP)
  check_cxx_symbol_exists(__GLIBCXX__ "version" GLIBCXX)

  if(LIBCPP)
    list(APPEND DOLFINX_CXX_DEVELOPER_DEFINITIONS "_LIBCPP_DEBUG=0")
  endif()

  if(GLIBCXX)
    list(APPEND DOLFINX_CXX_DEVELOPER_DEFINITIONS "_GLIBCXX_DEBUG=1")
  endif()

  # Turn off some checks in gcc12 and gcc13 due to false positives with the fmt
  # library
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11.4"
    AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0"
  )
    list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS
         -Wno-array-bounds;-Wno-stringop-overflow)
  endif()
endif()

find_package(Catch2 3 REQUIRED)

# FFCx compile: expr.py -> expr.c
add_custom_command(
  OUTPUT expr.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/fem/expr.py
  VERBATIM
  DEPENDS fem/expr.py
  COMMENT "Compile expr.py using FFCx"
)

# FFCx compile: poisson.py -> poisson.c
add_custom_command(
  OUTPUT poisson.c
  COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/poisson.py
  VERBATIM
  DEPENDS poisson.py
  COMMENT "Compile poisson.py using FFCx"
)

set(CMAKE_CXX_EXTENSIONS OFF)
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
  ${CMAKE_CURRENT_BINARY_DIR}/expr.c
  ${CMAKE_CURRENT_BINARY_DIR}/poisson.c
)
target_link_libraries(unittests PRIVATE Catch2::Catch2 dolfinx)
target_compile_features(unittests PRIVATE cxx_std_20 c_std_17)

# UUID requires bcrypt to be linked on Windows, broken in vcpkg.
# https://github.com/microsoft/vcpkg/issues/4481
if(WIN32)
  target_link_libraries(unittests PRIVATE bcrypt)
endif()

# Required to find FFCx generated headers.
target_include_directories(
  unittests PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

if (NOT PROJECT_IS_TOP_LEVEL)
  # Required to find version.h
  target_include_directories(
    unittests PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../>
  )
endif()

if (DEFINED DOLFINX_CXX_DEVELOPER_FLAGS)
  target_compile_options(
    unittests
    PRIVATE
      $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
  )
endif()

if (DEFINED DOLFINX_CXX_DEVELOPER_DEFINITIONS)
  target_compile_definitions(
    unittests
    PRIVATE
      $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_DEFINITIONS}>
  )
endif()

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
  add_test(NAME unittests_np_${N}
           COMMAND
           ${MPIEXEC_EXECUTABLE}
           ${MPIEXEC_NUMPROC_FLAG} ${N}
           ${MPIEXEC_PREFLAGS}
           $<TARGET_FILE:unittests>
           ${MPIEXEC_POSTFLAGS}
  )
  # This tells ctest how many physical cores (i.e., ${N}) are needed for each
  # test, preventing oversubscription when using e.g. `ctest -j2`.
  set_tests_properties(unittests_np_${N} PROPERTIES PROCESSORS ${N})
endforeach()
