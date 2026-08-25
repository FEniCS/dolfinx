# Detect and populate DOLFINX_CXX_DEVELOPER_FLAGS,
# DOLFINX_CXX_DEVELOPER_DEBUG_FLAGS and DOLFINX_CXX_DEVELOPER_DEFINITIONS for
# use in Developer and DeveloperDebug build type targets.
#
# Developer balances test speed against correctness checks (-O2 plus a strict
# warning set and standard library assertions). DeveloperDebug prioritises
# finding bugs (-Og, sanitizers, hardening) over performance, via the
# CMAKE_{C,CXX}_FLAGS_DEVELOPERDEBUG and CMAKE_*_LINKER_FLAGS_DEVELOPERDEBUG
# per-config variables, so it reaches every target, including FFCx-generated C
# kernels.

include(CheckCXXCompilerFlag)
include(CheckCXXSymbolExists)

# Cleared so that the module can be included in more than one directory scope
# without accumulating duplicate flags.
unset(DOLFINX_CXX_DEVELOPER_WARNING_FLAGS)
unset(DOLFINX_CXX_DEVELOPER_FLAGS)
unset(DOLFINX_CXX_DEVELOPER_DEBUG_FLAGS)
unset(DOLFINX_CXX_DEVELOPER_DEFINITIONS)

# Add some strict compiler checks
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)
if(HAVE_PEDANTIC)
  list(
    APPEND DOLFINX_CXX_DEVELOPER_WARNING_FLAGS
    -Wall;-Werror;-Wextra;-pedantic
  )
endif()

# Turn off some checks in gcc12 and gcc13 due to false positives with the fmt
# library, and with std::optional (e.g. common::Timer::_start_time)
if(
  CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11.4"
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0"
)
  list(
    APPEND DOLFINX_CXX_DEVELOPER_WARNING_FLAGS
    -Wno-array-bounds;-Wno-stringop-overflow;-Wno-maybe-uninitialized
  )
endif()

# The warning set is scoped to DOLFINx's own C++ sources in both Developer and
# DeveloperDebug, so it is never applied to FFCx-generated C.
set(DOLFINX_CXX_DEVELOPER_FLAGS ${DOLFINX_CXX_DEVELOPER_WARNING_FLAGS})
set(DOLFINX_CXX_DEVELOPER_DEBUG_FLAGS ${DOLFINX_CXX_DEVELOPER_WARNING_FLAGS})

# Debug flags (Developer only; DeveloperDebug uses -g3, set below)
check_cxx_compiler_flag(-g HAVE_DEBUG)
if(HAVE_DEBUG)
  list(APPEND DOLFINX_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation (Developer only; DeveloperDebug uses -Og, set below)
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
  # _GLIBCXX_ASSERTIONS, not _GLIBCXX_DEBUG, which is ABI-incompatible with
  # prebuilt dependencies (Catch2, spdlog, ADIOS2, PETSc).
  list(APPEND DOLFINX_CXX_DEVELOPER_DEFINITIONS _GLIBCXX_ASSERTIONS)
endif()

# ------------------------------------------------------------------------------
# DeveloperDebug: -Og plus sanitizers

set(
  DOLFINX_DEVELOPER_DEBUG_SANITIZERS
  "address;undefined"
  CACHE STRING
  "Sanitizers for the DeveloperDebug build type, e.g. 'address;undefined', 'thread', or '' for none."
)
mark_as_advanced(DOLFINX_DEVELOPER_DEBUG_SANITIZERS)

set(DOLFINX_CXX_DEVELOPER_DEBUG_SANITIZE_FLAGS)
if(DOLFINX_DEVELOPER_DEBUG_SANITIZERS)
  list(JOIN DOLFINX_DEVELOPER_DEBUG_SANITIZERS "," _dolfinx_dd_sanitize_arg)
  set(
    DOLFINX_CXX_DEVELOPER_DEBUG_SANITIZE_FLAGS
    -fsanitize=${_dolfinx_dd_sanitize_arg}
    -fno-sanitize-recover=all
    -fno-omit-frame-pointer
    # vptr reads RTTI across the non-instrumented libstdc++/libc++/PETSc/ADIOS2
    # boundary; function flags the reinterpret_cast'd complex UFCx kernels in
    # fem/kernel.h and fem/utils.h. Both are false positives here.
    -fno-sanitize=vptr,function
  )
endif()
set(
  DOLFINX_CXX_DEVELOPER_DEBUG_LINK_OPTIONS
  ${DOLFINX_CXX_DEVELOPER_DEBUG_SANITIZE_FLAGS}
)

# Absent here: the warning flags above (these variables apply to every target
# in scope, including FFCx-generated C, whose warnings are not DOLFINx's to
# fix -- instrumenting the kernels is the point).
list(JOIN DOLFINX_CXX_DEVELOPER_DEBUG_SANITIZE_FLAGS " " _dolfinx_dd_flags_str)
set(CMAKE_CXX_FLAGS_DEVELOPERDEBUG "-Og -g3 ${_dolfinx_dd_flags_str}")
set(CMAKE_C_FLAGS_DEVELOPERDEBUG "${CMAKE_CXX_FLAGS_DEVELOPERDEBUG}")
set(CMAKE_EXE_LINKER_FLAGS_DEVELOPERDEBUG "${_dolfinx_dd_flags_str}")
set(CMAKE_SHARED_LINKER_FLAGS_DEVELOPERDEBUG "${_dolfinx_dd_flags_str}")
set(CMAKE_MODULE_LINKER_FLAGS_DEVELOPERDEBUG "${_dolfinx_dd_flags_str}")

unset(_dolfinx_dd_sanitize_arg)
unset(_dolfinx_dd_flags_str)
