# Detect and populate DOLFINX_CXX_DEVELOPER_FLAGS,
# DOLFINX_CXX_DEVELOPER_DEBUG_FLAGS and DOLFINX_CXX_DEVELOPER_DEFINITIONS for
# use in Developer and DeveloperDebug build type targets.
#
# Developer balances test execution speed against correctness checks (-O2
# plus a strict warning set and standard library assertions). DeveloperDebug
# prioritises finding bugs (-Og, sanitizers, hardening) over performance, and
# additionally sets the CMAKE_{C,CXX}_FLAGS_DEVELOPERDEBUG and
# CMAKE_*_LINKER_FLAGS_DEVELOPERDEBUG per-config variables, so instrumentation
# reaches every target in this directory scope and every link line, including
# the FFCx-generated C kernels.

include(CheckCXXCompilerFlag)
include(CheckCXXSymbolExists)
include(CMakePushCheckState)

# Cleared so that the module can be included in more than one directory scope
# without accumulating duplicate flags.
unset(DOLFINX_CXX_DEVELOPER_WARNING_FLAGS)
unset(DOLFINX_CXX_DEVELOPER_FLAGS)
unset(DOLFINX_CXX_DEVELOPER_DEBUG_FLAGS)
unset(DOLFINX_CXX_DEVELOPER_DEFINITIONS)

# Build types DOLFINx knows about. A typo otherwise silently produces a build
# with no extra flags at all.
set(
  DOLFINX_BUILD_TYPES
  Debug
  Developer
  DeveloperDebug
  MinSizeRel
  Release
  RelWithDebInfo
)

# Reject an unrecognised CMAKE_BUILD_TYPE. Called explicitly by the DOLFINx
# C++ and Python builds; not called by the demos or by downstream projects,
# which may legitimately use their own build types.
function(dolfinx_validate_build_type)
  if(CMAKE_CONFIGURATION_TYPES OR NOT CMAKE_BUILD_TYPE)
    return()
  endif()
  if(DEFINED CACHE{CMAKE_BUILD_TYPE})
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${DOLFINX_BUILD_TYPES})
  endif()
  if(NOT CMAKE_BUILD_TYPE IN_LIST DOLFINX_BUILD_TYPES)
    message(
      FATAL_ERROR
      "Unknown CMAKE_BUILD_TYPE '${CMAKE_BUILD_TYPE}'. Options are: ${DOLFINX_BUILD_TYPES}."
    )
  endif()
endfunction()

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
  # _LIBCPP_HARDENING_MODE_DEBUG is libc++'s maximal (ABI-safe) mode, so
  # DeveloperDebug reuses it unchanged.
  list(
    APPEND DOLFINX_CXX_DEVELOPER_DEFINITIONS
    _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG
  )
endif()

if(GLIBCXX)
  # _GLIBCXX_ASSERTIONS, not _GLIBCXX_DEBUG: the latter changes the layout of
  # libstdc++ containers and would be an ABI mismatch against prebuilt
  # Catch2, spdlog, ADIOS2 and PETSc.
  list(APPEND DOLFINX_CXX_DEVELOPER_DEFINITIONS _GLIBCXX_ASSERTIONS)
endif()

# ------------------------------------------------------------------------------
# DeveloperDebug per-config flags

set(
  DOLFINX_DEVELOPER_DEBUG_SANITIZERS
  "address;undefined"
  CACHE STRING
  "Sanitizers enabled by the DeveloperDebug build type, as a ;-list of -fsanitize= names, e.g. 'address;undefined', 'thread', or '' for none."
)
mark_as_advanced(DOLFINX_DEVELOPER_DEBUG_SANITIZERS)

unset(_dolfinx_dd_flags)
unset(_dolfinx_dd_link_flags)
unset(_dolfinx_dd_sanitizers)

# -Og rather than -O0: with the standard library hardening checks above and
# sanitizer instrumentation below, -O0 leaves every std::span/mdspan accessor
# out of line and the test suite stops being runnable. Frame pointers and
# sibling calls must be preserved for readable sanitizer stack traces.
foreach(
  _flag
  IN
  ITEMS
    -Og
    -g3
    -fno-omit-frame-pointer
    -fno-optimize-sibling-calls
    -fstack-protector-strong
)
  string(MAKE_C_IDENTIFIER "HAVE${_flag}" _have)
  check_cxx_compiler_flag(${_flag} ${_have})
  if(${_have})
    list(APPEND _dolfinx_dd_flags ${_flag})
  endif()
endforeach()

# Each sanitizer is probed with a link step, as -fsanitize= pulls in a
# runtime library that may not be installed (e.g. TSan on some platforms).
foreach(_san IN LISTS DOLFINX_DEVELOPER_DEBUG_SANITIZERS)
  string(MAKE_C_IDENTIFIER "HAVE_SANITIZE_${_san}" _have)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=${_san})
  check_cxx_compiler_flag(-fsanitize=${_san} ${_have})
  cmake_pop_check_state()
  if(${_have})
    list(APPEND _dolfinx_dd_sanitizers ${_san})
  else()
    message(
      WARNING
      "Sanitizer '${_san}' is not usable with ${CMAKE_CXX_COMPILER_ID}; skipping."
    )
  endif()
endforeach()

if(_dolfinx_dd_sanitizers)
  list(JOIN _dolfinx_dd_sanitizers "," _sanitize_arg)
  list(APPEND _dolfinx_dd_flags -fsanitize=${_sanitize_arg})
  list(APPEND _dolfinx_dd_link_flags -fsanitize=${_sanitize_arg})
  # UBSan otherwise prints and continues, leaving a test suite green in the
  # presence of undefined behaviour. Relax at run time with
  # UBSAN_OPTIONS=halt_on_error=0.
  check_cxx_compiler_flag(-fno-sanitize-recover=all HAVE_NO_SANITIZE_RECOVER)
  if(HAVE_NO_SANITIZE_RECOVER)
    list(APPEND _dolfinx_dd_flags -fno-sanitize-recover=all)
  endif()
endif()

if("address" IN_LIST _dolfinx_dd_sanitizers)
  check_cxx_compiler_flag(
    -fsanitize-address-use-after-scope
    HAVE_ASAN_USE_AFTER_SCOPE
  )
  if(HAVE_ASAN_USE_AFTER_SCOPE)
    list(APPEND _dolfinx_dd_flags -fsanitize-address-use-after-scope)
  endif()
  # Clang links the ASan runtime statically by default. The Python extension
  # module is dlopened, so the runtime has to be a preloadable shared object.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=address -shared-libasan)
    check_cxx_compiler_flag(
      "-fsanitize=address -shared-libasan"
      HAVE_SHARED_LIBASAN
    )
    cmake_pop_check_state()
    if(HAVE_SHARED_LIBASAN)
      list(APPEND _dolfinx_dd_flags -shared-libasan)
      list(APPEND _dolfinx_dd_link_flags -shared-libasan)
    endif()
  endif()
endif()

if("undefined" IN_LIST _dolfinx_dd_sanitizers)
  # vptr requires an instrumented standard library and reads RTTI across the
  # (non-instrumented) libstdc++/libc++/PETSc/ADIOS2 boundary, producing
  # false positives. function checks that an indirect call's static type
  # matches the callee's recorded signature; fem/kernel.h and fem/utils.h
  # reinterpret_cast the C UFCx complex kernels to a different (but
  # ABI-compatible) function pointer type, which function would flag on
  # every complex-scalar assembly call.
  foreach(_flag IN ITEMS -fno-sanitize=vptr -fno-sanitize=function)
    string(MAKE_C_IDENTIFIER "HAVE${_flag}" _have)
    check_cxx_compiler_flag(${_flag} ${_have})
    if(${_have})
      list(APPEND _dolfinx_dd_flags ${_flag})
    endif()
  endforeach()
endif()

# Warning flags are deliberately absent here: these variables apply to every
# target in scope, including the FFCx-generated C kernels, whose warnings are
# not DOLFINx's to fix. Instrumenting the kernels is the point.
list(JOIN _dolfinx_dd_flags " " CMAKE_CXX_FLAGS_DEVELOPERDEBUG)
list(JOIN _dolfinx_dd_link_flags " " _dolfinx_dd_link_flags_str)
set(CMAKE_C_FLAGS_DEVELOPERDEBUG "${CMAKE_CXX_FLAGS_DEVELOPERDEBUG}")
set(CMAKE_EXE_LINKER_FLAGS_DEVELOPERDEBUG "${_dolfinx_dd_link_flags_str}")
set(CMAKE_SHARED_LINKER_FLAGS_DEVELOPERDEBUG "${_dolfinx_dd_link_flags_str}")
set(CMAKE_MODULE_LINKER_FLAGS_DEVELOPERDEBUG "${_dolfinx_dd_link_flags_str}")

unset(_dolfinx_dd_flags)
unset(_dolfinx_dd_link_flags)
unset(_dolfinx_dd_link_flags_str)
unset(_dolfinx_dd_sanitizers)
