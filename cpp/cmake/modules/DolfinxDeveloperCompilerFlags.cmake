# Detect and populate DOLFINX_CXX_DEVELOPER_FLAGS and
# DOLFINX_CXX_DEVELOPER_DEFINITIONS for use in Developer build type targets.
#
# This module is installed and included by consumers (the Python
# interface, and the demos when built standalone), so the probe results
# below are cached under DOLFINX_-prefixed names. An unprefixed name such
# as LIBCPP could collide with an entry a consuming project already has,
# and check_* silently reuses a cached value rather than re-probing.

include(CheckCXXCompilerFlag)
include(CheckCXXSymbolExists)

# Cleared so that the module can be included in more than one directory scope
# without accumulating duplicate flags.
unset(DOLFINX_CXX_DEVELOPER_FLAGS)
unset(DOLFINX_CXX_DEVELOPER_DEFINITIONS)

# Add some strict compiler checks
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" DOLFINX_HAVE_PEDANTIC)
if(DOLFINX_HAVE_PEDANTIC)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic)
endif()

# GCC's plain -Wshadow also flags constructor parameters and lambda
# parameters that intentionally reuse a member or enclosing-scope name,
# an idiom used throughout this library's public headers. Use the
# narrower -Wshadow=compatible-local on GCC, which is restricted to
# local variable/parameter shadowing of a compatible type; Clang's
# single -Wshadow is already scoped that way.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  check_cxx_compiler_flag(
    -Wshadow=compatible-local
    DOLFINX_HAVE_WSHADOW_COMPATIBLE_LOCAL
  )
  if(DOLFINX_HAVE_WSHADOW_COMPATIBLE_LOCAL)
    list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wshadow=compatible-local)
  endif()
else()
  check_cxx_compiler_flag(-Wshadow DOLFINX_HAVE_WSHADOW)
  if(DOLFINX_HAVE_WSHADOW)
    list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wshadow)
  endif()
endif()

# Debug flags
check_cxx_compiler_flag(-g DOLFINX_HAVE_G)
if(DOLFINX_HAVE_G)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation
check_cxx_compiler_flag(-O2 DOLFINX_HAVE_O2)
if(DOLFINX_HAVE_O2)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -O2)
endif()

# Enable C++ standard library debugging
check_cxx_symbol_exists(_LIBCPP_VERSION "version" DOLFINX_HAVE_LIBCPP)
check_cxx_symbol_exists(__GLIBCXX__ "version" DOLFINX_HAVE_GLIBCXX)

if(DOLFINX_HAVE_LIBCPP)
  list(
    APPEND DOLFINX_CXX_DEVELOPER_DEFINITIONS
    _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG
  )
endif()

if(DOLFINX_HAVE_GLIBCXX)
  list(APPEND DOLFINX_CXX_DEVELOPER_DEFINITIONS _GLIBCXX_ASSERTIONS)
endif()

# Turn off some checks in gcc12 and gcc13 due to false positives with the fmt
# library, and with std::optional (e.g. common::Timer::_start_time)
if(
  CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11.4"
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0"
)
  list(
    APPEND DOLFINX_CXX_DEVELOPER_FLAGS
    -Wno-array-bounds;-Wno-stringop-overflow;-Wno-maybe-uninitialized
  )
endif()
