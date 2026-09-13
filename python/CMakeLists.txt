cmake_minimum_required(VERSION 3.26)

project(dolfinx_nanobind LANGUAGES CXX)

find_package(
  Python
  COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT}
  REQUIRED
)

# Options
include(FeatureSummary)

option(ENABLE_CLANG_TIDY "Run clang-tidy while building" OFF)
add_feature_info(
  ENABLE_CLANG_TIDY
  ENABLE_CLANG_TIDY
  "Run clang-tidy while building"
)

option(
  DOLFINX_INSTALL_RPATH_USE_LINK_PATH
  "Add linker search paths to rpath of dolfinx target. Packagers may want
to set FALSE."
  TRUE
)
add_feature_info(
  DOLFINX_INSTALL_RPATH_USE_LINK_PATH
  DOLFINX_INSTALL_RPATH_USE_LINK_PATH
  "Add linker search paths to rpath of dolfinx target. Packagers may want
	to set FALSE."
)

feature_summary(WHAT ALL)

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# Enabling C here is a workaround for an ADIOS2 bug, see
# https://github.com/ornladios/ADIOS2/pull/4255. Remove when the ADIOS2
# fix is released.
enable_language(C)

find_package(DOLFINX REQUIRED CONFIG)
message(STATUS "Found DOLFINx at ${DOLFINX_DIR}")

if(CMAKE_BUILD_TYPE STREQUAL "Developer")
  # This enables warnings from DOLFINx headers even though they are a SYSTEM
  # dependency.
  set_target_properties(dolfinx::dolfinx PROPERTIES SYSTEM OFF)
endif()

get_target_property(
  _dolfinx_defs
  dolfinx::dolfinx
  INTERFACE_COMPILE_DEFINITIONS
)
if("HAS_SUPERLU_DIST" IN_LIST _dolfinx_defs)
  # SuperLU_DIST Python interface requires the superlu_defs.h header for int_t.
  find_package(PkgConfig)
  pkg_search_module(SUPERLU_DIST OPTIONAL IMPORTED_TARGET superlu_dist)
endif()

# Create the binding library nanobind handles its own calls to
# target_link_libraries

if(NOT "${SKBUILD_SABI_VERSION}" STREQUAL "")
  set(NANOBIND_SABI "STABLE_ABI")
endif()

nanobind_add_module(
  cpp
  NB_SUPPRESS_WARNINGS
  NOMINSIZE
  ${NANOBIND_SABI}
  dolfinx/wrappers/dolfinx.cpp
  dolfinx/wrappers/assemble.cpp
  dolfinx/wrappers/common.cpp
  dolfinx/wrappers/fem.cpp
  dolfinx/wrappers/geometry.cpp
  dolfinx/wrappers/graph.cpp
  dolfinx/wrappers/io.cpp
  dolfinx/wrappers/la.cpp
  dolfinx/wrappers/log.cpp
  dolfinx/wrappers/mesh.cpp
  dolfinx/wrappers/petsc.cpp
  dolfinx/wrappers/refinement.cpp
)
set_target_properties(cpp PROPERTIES CXX_EXTENSIONS OFF)
if(TARGET PkgConfig::SUPERLU_DIST)
  target_include_directories(cpp PRIVATE ${SUPERLU_DIST_INCLUDE_DIRS})
endif()

if(NANOBIND_SABI)
  # mpi4py and petsc4py use generated Cython headers so tell
  # them that the limited API is in use.
  target_compile_definitions(cpp PRIVATE MPI4PY_LIMITED_API=1)
  # Needing to set CYTHON_COMPILING_IN_LIMITED_API is a bug in
  # Cython-generated API headers, should become unnecessary with
  # mpi4py/petsc4py releases published using Cython >= 3.1.4
  # https://github.com/cython/cython/issues/7108
  target_compile_definitions(cpp PRIVATE CYTHON_COMPILING_IN_LIMITED_API=1)
endif()

include("${DOLFINX_DIR}/DolfinxDeveloperCompilerFlags.cmake")

target_compile_options(
  cpp
  PRIVATE
    $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
)
target_compile_definitions(
  cpp
  PRIVATE
    $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_DEFINITIONS}>
)
target_compile_features(cpp PRIVATE cxx_std_20)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(
    cpp
    PROPERTIES
      CXX_CLANG_TIDY
        "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../.clang-tidy"
  )
endif()

# Add DOLFINx libraries
target_link_libraries(cpp PRIVATE dolfinx::dolfinx)
# UUID requires bcrypt to be linked on Windows, broken in vcpkg.
# https://github.com/microsoft/vcpkg/issues/4481
if(WIN32)
  target_link_libraries(cpp PRIVATE bcrypt)
endif()

# Check for petsc4py
execute_process(
  COMMAND
    "${Python_EXECUTABLE}" -c "import petsc4py; print(petsc4py.get_include())"
  OUTPUT_VARIABLE PETSC4PY_INCLUDE_DIR
  RESULT_VARIABLE PETSC4PY_COMMAND_RESULT
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if("HAS_PETSC" IN_LIST _dolfinx_defs)
  if(NOT PETSC4PY_COMMAND_RESULT)
    message(
      STATUS
      "Found petsc4py include directory at ${PETSC4PY_INCLUDE_DIR}"
    )
    target_include_directories(cpp PRIVATE ${PETSC4PY_INCLUDE_DIR})
    target_compile_definitions(cpp PRIVATE HAS_PETSC4PY)
  else()
    message(WARNING "DOLFINx was built with PETSc but petsc4py was not found.")
  endif()
else()
  if(NOT PETSC4PY_COMMAND_RESULT)
    message(WARNING "DOLFINx was built without PETSc but petsc4py was found.")
  endif()
endif()

# Check for mpi4py
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import mpi4py; print(mpi4py.get_include())"
  OUTPUT_VARIABLE MPI4PY_INCLUDE_DIR
  RESULT_VARIABLE MPI4PY_COMMAND_RESULT
  ERROR_QUIET
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT MPI4PY_COMMAND_RESULT)
  message(STATUS "Found mpi4py include directory at ${MPI4PY_INCLUDE_DIR}")
  target_include_directories(cpp PRIVATE ${MPI4PY_INCLUDE_DIR})
else()
  message(FATAL_ERROR "mpi4py could not be found.")
endif()

set_target_properties(
  cpp
  PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ${DOLFINX_INSTALL_RPATH_USE_LINK_PATH}
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

install(TARGETS cpp LIBRARY DESTINATION dolfinx)

# and the nanobind typing stubs.
set(
  NANOBIND_STUB_OUTPUTS
  dolfinx/cpp/__init__.pyi
  dolfinx/cpp/common.pyi
  dolfinx/cpp/fem.pyi
  dolfinx/cpp/fem/__init__.pyi
  dolfinx/cpp/fem/petsc.pyi
  dolfinx/cpp/geometry.pyi
  dolfinx/cpp/graph.pyi
  dolfinx/cpp/io.pyi
  dolfinx/cpp/la.pyi
  dolfinx/cpp/la/__init__.pyi
  dolfinx/cpp/la/petsc.pyi
  dolfinx/cpp/log.pyi
  dolfinx/cpp/mesh.pyi
  dolfinx/cpp/nls.pyi
  dolfinx/cpp/nls/__init__.pyi
  dolfinx/cpp/nls/petsc.pyi
  dolfinx/cpp/refinement.pyi
)

if(WIN32)
  # On Windows we cannot import dolfinx.cpp without installing the package
  # alongside its external dlls. cpp is installed unconditionally above,
  # which nanobind requires to have happened prior to nanobind_add_stub.
  # nanobind automatically installs into ${CMAKE_INSTALL_PREFIX} in
  # INSTALL_TIME mode.
  # DEPENDS is redundant in INSTALL_TIME mode - cpp must be installed.
  nanobind_add_stub(
    cpp_stub
    INSTALL_TIME
    MODULE dolfinx.cpp
    RECURSIVE
    VERBOSE
    PATTERN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx/wrappers/stub_patterns.txt
    MARKER_FILE dolfinx/cpp/py.typed
    OUTPUT ${NANOBIND_STUB_OUTPUTS}
  )
else()
  # On UNIX-like systems we can import the cpp compiled module before
  # installing and rely on dynamic linking at import time. Build it
  # directly into a throwaway "dolfinx" package directory so that the
  # module imports under its real name (dolfinx.cpp): otherwise
  # nanobind's stubgen resolves submodule cross-references (e.g.
  # dolfinx.cpp.la) against the bare top-level name "cpp", producing
  # "import cpp.la" in the generated stubs, which does not exist as an
  # importable module and silently resolves to Any under mypy. stubgen
  # infers its output location from the imported module's __file__, so
  # placing the build output under CMAKE_CURRENT_BINARY_DIR/dolfinx
  # keeps that location matching the OUTPUT paths below (relative to
  # CMAKE_CURRENT_BINARY_DIR). See nanobind's nanobind_add_stub docs
  # for the import mechanism:
  # https://nanobind.readthedocs.io/en/latest/api_cmake.html#command:nanobind_add_stub
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dolfinx)
  file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/dolfinx/__init__.py)
  set_target_properties(
    cpp
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dolfinx
  )

  nanobind_add_stub(
    cpp_stub
    MODULE dolfinx.cpp
    PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS cpp
    RECURSIVE
    VERBOSE
    PATTERN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/dolfinx/wrappers/stub_patterns.txt
    MARKER_FILE dolfinx/cpp/py.typed
    OUTPUT ${NANOBIND_STUB_OUTPUTS}
  )

  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/dolfinx/cpp
    DESTINATION dolfinx
    FILES_MATCHING
    PATTERN "*.pyi"
    PATTERN "*.typed"
  )
endif()
