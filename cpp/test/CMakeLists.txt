cmake_minimum_required(VERSION 3.26)

project(dolfinx-tests LANGUAGES C CXX)

if(PROJECT_IS_TOP_LEVEL)
  include(CTest) # enables testing
  find_package(DOLFINX REQUIRED)
  include("${DOLFINX_DIR}/DolfinxDeveloperCompilerFlags.cmake")
endif()

find_package(Catch2 3 REQUIRED)

# Select the compiler used by the GPU compatibility test.
set(
  DOLFINX_TEST_GPU_COMPILER
  AUTO
  CACHE STRING
  "GPU test compiler (AUTO, OFF, CUDA, HIP)"
)
set_property(
  CACHE DOLFINX_TEST_GPU_COMPILER
  PROPERTY STRINGS AUTO OFF CUDA HIP
)
if(NOT DOLFINX_TEST_GPU_COMPILER MATCHES "^(AUTO|OFF|CUDA|HIP)$")
  message(
    FATAL_ERROR
    "DOLFINX_TEST_GPU_COMPILER must be AUTO, OFF, CUDA or HIP"
  )
endif()

unset(_gpu_language)
if(DOLFINX_TEST_GPU_COMPILER STREQUAL "AUTO")
  # Honour an explicitly selected HIP compiler before probing CUDA;
  # check_language is a no-op when the compiler is already set.
  if(CMAKE_HIP_COMPILER)
    set(_gpu_language HIP)
  else()
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      set(_gpu_language CUDA)
    else()
      check_language(HIP)
      if(CMAKE_HIP_COMPILER)
        set(_gpu_language HIP)
      endif()
    endif()
  endif()
elseif(NOT DOLFINX_TEST_GPU_COMPILER STREQUAL "OFF")
  set(_gpu_language ${DOLFINX_TEST_GPU_COMPILER})
endif()
if(_gpu_language)
  # Enable before creating targets so they inherit GPU architectures.
  enable_language(${_gpu_language})
endif()

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
  petsc.cpp
  slepc.cpp
  io.cpp
  io_hdf5.cpp
  io_xdmf.cpp
  common/CIFailure.cpp
  common/sub_systems_manager.cpp
  common/index_map.cpp
  common/move_semantics.cpp
  common/mpi.cpp
  common/sort.cpp
  fem/assembly_kernels.cpp
  fem/form.cpp
  fem/functionspace.cpp
  fem/interpolation.cpp
  mesh/branching_manifold.cpp
  mesh/distributed_mesh.cpp
  mesh/generation.cpp
  mesh/geometric_partitioner.cpp
  mesh/permutations.cpp
  mesh/read_named_meshtags.cpp
  mesh/refinement/interval.cpp
  mesh/refinement/option.cpp
  mesh/refinement/rectangle.cpp
  mesh/refinement/mark.cpp
  nls/snessolver.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/expr.c
  ${CMAKE_CURRENT_BINARY_DIR}/poisson.c
)
target_link_libraries(unittests PRIVATE Catch2::Catch2 dolfinx::dolfinx)
target_compile_features(unittests PRIVATE cxx_std_20 c_std_17)
set_target_properties(unittests PROPERTIES CXX_EXTENSIONS OFF)

# If a GPU language is enabled, compile DOLFINx headers and code with
# its compiler.
if(_gpu_language)
  # dolfinx_headers.h, included by gpu_compiler.cu, includes every
  # installed DOLFINx header (the dolfinx target's public header sets,
  # each of which has a single base directory).
  get_target_property(_sets dolfinx::dolfinx INTERFACE_HEADER_SETS)
  set(_includes "")
  foreach(_set IN LISTS _sets)
    get_target_property(_files dolfinx::dolfinx HEADER_SET_${_set})
    get_target_property(_dir dolfinx::dolfinx HEADER_DIRS_${_set})
    foreach(_file IN LISTS _files)
      cmake_path(RELATIVE_PATH _file BASE_DIRECTORY "${_dir}")
      string(APPEND _includes "#include <${_file}>\n")
    endforeach()
  endforeach()
  if(NOT _includes)
    message(FATAL_ERROR "No DOLFINx headers found for the GPU test.")
  endif()
  file(
    GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dolfinx_headers.h
    CONTENT "#pragma once\n${_includes}"
  )

  target_sources(
    unittests
    PRIVATE common/gpu_compiler.cpp common/gpu_compiler.cu
  )
  set_source_files_properties(
    common/gpu_compiler.cu
    PROPERTIES LANGUAGE ${_gpu_language}
  )
  if(_gpu_language STREQUAL "HIP")
    # HIP code built with ROCm's clang against GCC-built DOLFINx and
    # Basix needs:
    # - FMT_CONSTEVAL=: the clang rejects fmt 10's consteval format-string
    #   check in spdlog (logger.h), used by DOLFINx's logging.
    # - -fclang-abi-compat=17: clang >= 17 mangles template
    #   type-constraints (e.g. basix::create_element<std::floating_point
    #   T>) and GCC does not, so symbols would not match at link time.
    set_source_files_properties(
      common/gpu_compiler.cu
      PROPERTIES
        COMPILE_DEFINITIONS "FMT_CONSTEVAL="
        COMPILE_OPTIONS "-fclang-abi-compat=17"
    )
  endif()
  string(TOLOWER "${_gpu_language}" _gpu_lang)
  target_compile_features(unittests PRIVATE ${_gpu_lang}_std_20)
  set_target_properties(unittests PROPERTIES ${_gpu_language}_EXTENSIONS OFF)
endif()

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

# Windows defaults new threads to a 1 MiB stack, versus 8 MiB on
# Linux/macOS. Mesh-partitioning tests that build a real mesh (e.g.
# via SCOTCH's recursive bisection, whose working set scales with the
# (sub)graph being partitioned) overflow the default Windows stack even
# though they run fine elsewhere; match the other platforms' size.
if(WIN32 AND MSVC)
  target_link_options(unittests PRIVATE /STACK:8388608)
endif()

# Required to find FFCx generated headers.
target_include_directories(unittests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

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
      $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX,CUDA,HIP>>:${DOLFINX_CXX_DEVELOPER_DEFINITIONS}>
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
