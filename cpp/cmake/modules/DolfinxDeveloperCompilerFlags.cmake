# Detect and populate DOLFINX_CXX_DEVELOPER_FLAGS and
# DOLFINX_CXX_DEVELOPER_DEFINITIONS for use in Developer build type targets.

include(CheckCXXCompilerFlag)
include(CheckCXXSymbolExists)

# Add some strict compiler checks
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)
if(HAVE_PEDANTIC)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -Wall;-Werror;-Wextra;-pedantic)
endif()

# Debug flags
check_cxx_compiler_flag(-g HAVE_DEBUG)
if(HAVE_DEBUG)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation
check_cxx_compiler_flag(-O2 HAVE_O2_OPTIMISATION)
if(HAVE_O2_OPTIMISATION)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -O2)
endif()

# Enable C++ standard library debugging
check_cxx_symbol_exists(_LIBCPP_VERSION "version" LIBCPP)
check_cxx_symbol_exists(__GLIBCXX__ "version" GLIBCXX)

if(LIBCPP)
  list(
    APPEND DOLFINX_CXX_DEVELOPER_DEFINITIONS
    _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG
  )
endif()

if(GLIBCXX)
  list(APPEND DOLFINX_CXX_DEVELOPER_DEFINITIONS _GLIBCXX_ASSERTIONS)
endif()

# Turn off some checks in gcc12 and gcc13 due to false positives with the fmt
# library
if(
  CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11.4"
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0"
)
  list(
    APPEND DOLFINX_CXX_DEVELOPER_FLAGS
    -Wno-array-bounds;-Wno-stringop-overflow
  )
endif()
