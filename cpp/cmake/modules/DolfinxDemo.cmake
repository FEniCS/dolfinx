# Support for building the DOLFINx C++ demos.
#
# The demos double as standalone example projects, so this module is
# installed alongside DOLFINXConfig.cmake and included by each demo.
#
# Provides:
#
#   dolfinx_add_demo(<name> [UFL <file.py>] [NO_COMPLEX])
#
# which builds main.cpp (plus, with UFL, the FFCx output of <file.py>)
# into a demo_<name> executable and registers it with CTest on 1, 2 and 3
# MPI ranks. NO_COMPLEX skips demos that do not support complex scalars.

include_guard(GLOBAL)

include(CMakePushCheckState)
include(CheckCXXCompilerFlag)
include(CheckSymbolExists)

# Set ${out_flag} to the FFCx --scalar_type option matching the PETSc
# scalar type, and ${out_is_complex} to whether that type is complex.
# Without PETSc there is nothing to match and FFCx's default is used.
function(_dolfinx_demo_scalar_type out_flag out_is_complex)
  if(NOT PETSC_INCLUDE_DIRS)
    set(${out_flag} "" PARENT_SCOPE)
    set(${out_is_complex} FALSE PARENT_SCOPE)
    return()
  endif()

  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES ${PETSC_INCLUDE_DIRS})
  check_symbol_exists(PETSC_USE_COMPLEX petscsystypes.h PETSC_SCALAR_COMPLEX)
  check_symbol_exists(PETSC_USE_REAL_DOUBLE petscsystypes.h PETSC_REAL_DOUBLE)
  cmake_pop_check_state()

  if(PETSC_SCALAR_COMPLEX)
    if(PETSC_REAL_DOUBLE)
      set(${out_flag} "--scalar_type=complex128" PARENT_SCOPE)
    else()
      set(${out_flag} "--scalar_type=complex64" PARENT_SCOPE)
    endif()
  else()
    if(PETSC_REAL_DOUBLE)
      set(${out_flag} "--scalar_type=float64" PARENT_SCOPE)
    else()
      set(${out_flag} "--scalar_type=float32" PARENT_SCOPE)
    endif()
  endif()
  set(${out_is_complex} ${PETSC_SCALAR_COMPLEX} PARENT_SCOPE)
endfunction()

function(dolfinx_add_demo name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "NO_COMPLEX" "UFL" "")
  if(ARG_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "dolfinx_add_demo: unrecognised arguments: ${ARG_UNPARSED_ARGUMENTS}"
    )
  endif()

  _dolfinx_demo_scalar_type(_scalar_type _is_complex)
  if(ARG_NO_COMPLEX AND _is_complex)
    message(STATUS "** Demo '${name}' does not support complex mode")
    return()
  endif()

  set(_target demo_${name})
  set(_sources main.cpp)

  # Compile the UFL file with FFCx
  if(ARG_UFL)
    cmake_path(REPLACE_EXTENSION ARG_UFL LAST_ONLY ".c" OUTPUT_VARIABLE _kernel)
    add_custom_command(
      OUTPUT ${_kernel}
      COMMAND ffcx ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_UFL} ${_scalar_type}
      VERBATIM
      DEPENDS ${ARG_UFL}
      COMMENT "Compile ${ARG_UFL} using FFCx"
    )
    list(APPEND _sources ${CMAKE_CURRENT_BINARY_DIR}/${_kernel})

    # FFCx-generated C kernels have unused parameters fixed by the UFL ABI
    set_source_files_properties(
      ${CMAKE_CURRENT_BINARY_DIR}/${_kernel}
      PROPERTIES COMPILE_OPTIONS "-Wno-unused-parameter"
    )
  endif()

  add_executable(${_target} ${_sources})
  target_link_libraries(${_target} PRIVATE dolfinx::dolfinx)
  target_include_directories(${_target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_compile_features(${_target} PRIVATE cxx_std_20)
  set_target_properties(${_target} PROPERTIES CXX_EXTENSIONS OFF)

  # Use the DOLFINx Developer compiler flags for Developer build types.
  # Included here rather than at module scope so that the flags land in
  # this function's scope, which is where they are used.
  # CMAKE_CURRENT_FUNCTION_LIST_DIR is the directory of the file defining
  # this function, which holds the other DOLFINx modules in both the
  # source and the install tree (unlike CMAKE_CURRENT_LIST_DIR, which
  # would be the calling CMakeLists.txt).
  include(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/DolfinxDeveloperCompilerFlags.cmake"
  )
  target_compile_options(
    ${_target}
    PRIVATE
      $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
  )
  target_compile_definitions(
    ${_target}
    PRIVATE
      $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_DEFINITIONS}>
  )

  # Do not throw an error for 'multi-line comments' (these are typical in
  # rst, which includes LaTeX). Appended last so that it overrides the
  # -Wcomment implied by -Wall above.
  check_cxx_compiler_flag("-Wno-comment" HAVE_NO_MULTLINE)
  target_compile_options(
    ${_target}
    PRIVATE $<$<BOOL:${HAVE_NO_MULTLINE}>:-Wno-comment>
  )

  # Test targets (used by the DOLFINx testing system). To select one
  # number of processes use, e.g.: ctest -R demo_poisson_np_3
  foreach(N 1 2 3)
    add_test(
      NAME ${_target}_np_${N}
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${N} ${MPIEXEC_PREFLAGS}
        $<TARGET_FILE:${_target}> ${MPIEXEC_POSTFLAGS}
    )
    # Tells ctest how many physical cores (i.e., ${N}) are needed for each
    # test, preventing oversubscription when using e.g. `ctest -j2`.
    set_tests_properties(${_target}_np_${N} PROPERTIES PROCESSORS ${N})
  endforeach()
endfunction()
