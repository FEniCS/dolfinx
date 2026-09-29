# Generates and installs dolfinx.pc, the pkg-config file for a pure
# pkg-config (non-CMake) consumer of an installed DOLFINx.
#
# Build-time only: unlike DolfinxPkgConfigHelpers.cmake, this module is not
# installed. It is included once, from dolfinx/CMakeLists.txt.

include_guard(GLOBAL)

# Recursively accumulate the library files of a dolfinx PRIVATE dependency
# with no pkg-config package of its own (SCOTCH, ParMETIS, KaHIP, and the
# METIS/GKLib/kahip libraries they optionally bring in) into ${libs_var}.
# Only exposed by a static dolfinx: CMake wraps a STATIC library's PRIVATE
# deps in $<LINK_ONLY:target> in INTERFACE_LINK_LIBRARIES; FindParMETIS.cmake
# and FindKaHIP.cmake gate their own optional deps behind
# $<$<BOOL:...>:target>, unwrapped the same way. No dolfinx public header
# pulls in any of these libraries' headers, so include dirs aren't collected.
#
# ${visited_var}: targets already walked, so a shared dep (e.g. MPI::MPI_CXX)
# is only expanded once.
function(dolfinx_pkgconfig_collect_private target libs_var visited_var)
  # Separate if/elseif: CMake's if() OR doesn't short-circuit, so a second
  # failing MATCHES would clobber CMAKE_MATCH_1 from the first.
  if(target MATCHES "^\\$<LINK_ONLY:(.+)>$")
    set(target "${CMAKE_MATCH_1}")
  elseif(target MATCHES "^\\$<\\$<[A-Z_]+:[^<>]*>:(.+)>$")
    set(target "${CMAKE_MATCH_1}")
  endif()

  # Still wrapped in an unrecognised generator expression: give up.
  if(target MATCHES "[<>]")
    return()
  endif()

  if(NOT TARGET ${target} OR ${target} IN_LIST ${visited_var})
    return()
  endif()
  list(APPEND ${visited_var} ${target})

  # A find_library()-based IMPORTED target keeps its library file in
  # IMPORTED_LOCATION, not INTERFACE_LINK_LIBRARIES. A multi-config target
  # (SCOTCH, from its own upstream SCOTCHConfig.cmake) has only
  # per-configuration IMPORTED_LOCATION_<CONFIG>, unrelated to dolfinx's own
  # CMAKE_BUILD_TYPE; any one is equally valid to link against.
  set(_loc_props IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION)
  get_target_property(_imported_configs ${target} IMPORTED_CONFIGURATIONS)
  if(_imported_configs)
    foreach(_cfg IN LISTS _imported_configs)
      string(TOUPPER "${_cfg}" _cfg)
      list(APPEND _loc_props IMPORTED_LOCATION_${_cfg})
    endforeach()
  endif()
  foreach(_loc_prop IN LISTS _loc_props)
    get_target_property(_loc ${target} ${_loc_prop})
    # Unquoted: a pre-seeded cache var (e.g. Spack's METIS_LIBRARY) can make
    # this a real semicolon-joined list, not the single path it should be.
    if(_loc)
      list(APPEND ${libs_var} ${_loc})
      break()
    endif()
  endforeach()

  get_target_property(_deps ${target} INTERFACE_LINK_LIBRARIES)
  if(_deps)
    foreach(_dep IN LISTS _deps)
      dolfinx_pkgconfig_collect_private("${_dep}" ${libs_var} ${visited_var})
    endforeach()
  endif()

  set(${libs_var} ${${libs_var}} PARENT_SCOPE)
  set(${visited_var} ${${visited_var}} PARENT_SCOPE)
endfunction()

# Convert a list of library files/names in ${libs_var} to pkg-config link
# flags, de-duplicated, appended to any flags already in ${flags_var}.
function(dolfinx_pkgconfig_libs_to_flags libs_var flags_var)
  set(_flags "${${flags_var}}")
  foreach(_lib IN LISTS ${libs_var})
    # Add -Wl,option directives
    if("${_lib}" MATCHES "-Wl,[^ ]*")
      string(PREPEND _flags "${_lib} ")
    else()
      cmake_path(GET _lib PARENT_PATH _path)
      cmake_path(GET _lib STEM _name)
      string(REGEX REPLACE "^lib" "" _name "${_name}")

      # Add libraries that matches the form -L<libdir> -l<lib>
      if(NOT "${_path}" STREQUAL "")
        string(PREPEND _flags "-L${_path} -l${_name} ")
      endif()
    endif()
  endforeach()

  separate_arguments(_flags)
  list(REMOVE_DUPLICATES _flags)
  list(JOIN _flags " " _flags)
  set(${flags_var} "${_flags}" PARENT_SCOPE)
endfunction()

# Configure and install ${DOLFINX_SOURCE_DIR}/cmake/templates/dolfinx.pc.in
# for the ${target} library target.
function(dolfinx_pkgconfig_generate target)
  block(
    SCOPE_FOR VARIABLES
    PROPAGATE PKG_REQUIRES PKG_REQUIRES_PRIVATE PKG_CXXFLAGS PKG_LINKFLAGS
              PKG_LIBS_PRIVATE PKG_INCLUDES PKG_DEFINITIONS
  )
    set(PKG_REQUIRES "spdlog")
    list(APPEND PKG_REQUIRES "basix")
    list(APPEND PKG_REQUIRES "pugixml")

    if(TARGET PkgConfig::PETSC)
      list(APPEND PKG_REQUIRES ${PETSC_MODULE_NAME})
    endif()
    if(TARGET PkgConfig::SLEPC)
      list(APPEND PKG_REQUIRES ${SLEPC_MODULE_NAME})
    endif()

    list(JOIN PKG_REQUIRES " " PKG_REQUIRES)

    # SuperLU_DIST is linked PRIVATE (unlike PETSc/SLEPc, public via
    # la/petsc.h), so it goes in Requires.private, not Requires.
    set(PKG_REQUIRES_PRIVATE)
    if(TARGET PkgConfig::SUPERLU_DIST)
      list(APPEND PKG_REQUIRES_PRIVATE ${SUPERLU_DIST_MODULE_NAME})
    endif()
    list(JOIN PKG_REQUIRES_PRIVATE " " PKG_REQUIRES_PRIVATE)

    # Get link libraries and includes
    get_target_property(
      PKGCONFIG_DOLFINX_TARGET_LINK_LIBRARIES
      ${target}
      INTERFACE_LINK_LIBRARIES
    )
    get_target_property(
      PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES
      ${target}
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    )
    if(PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES)
      list(
        FILTER
        PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES
        EXCLUDE
        REGEX
        "^\\$<|>$"
      )
    else()
      set(PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES "")
    endif()

    # Add imported targets to lists for creating pkg-config file
    set(PKGCONFIG_DOLFINX_LIBS)
    set(PKGCONFIG_DOLFINX_LIBS_PRIVATE)
    set(_visited_private_targets)

    foreach(_target IN LISTS PKGCONFIG_DOLFINX_TARGET_LINK_LIBRARIES)
      # A static-only PRIVATE dependency with no pkg-config package of its
      # own (SCOTCH, ParMETIS, KaHIP), wrapped as "$<LINK_ONLY:target>".
      # Goes into Libs.private, not Libs: a shared dolfinx already links it.
      if("${_target}" MATCHES "^\\$<LINK_ONLY:.+>$")
        dolfinx_pkgconfig_collect_private(
          "${_target}"
          PKGCONFIG_DOLFINX_LIBS_PRIVATE
          _visited_private_targets
        )
        continue()
      endif()

      # Skip any other "$<foo...>", which we get with static libs
      if(NOT "${_target}" MATCHES "^[^<>]+$")
        continue()
      endif()

      if("${_target}" MATCHES "::")
        # Get include paths
        get_target_property(_inc_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(_inc_dirs)
          list(APPEND PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES ${_inc_dirs})
        endif()

        # Get libraries
        get_target_property(_libs ${_target} INTERFACE_LINK_LIBRARIES)
        if(_libs)
          list(APPEND PKGCONFIG_DOLFINX_LIBS ${_libs})
        endif()
      else()
        # 'regular' libs, i.e. not imported targets
        list(APPEND PKGCONFIG_DOLFINX_LIBS ${_target})
      endif()

      # Compiled Boost, ADIOS2 and HDF5 imported targets carry the library
      # in IMPORTED_LOCATION_RELEASE (or IMPORTED_LOCATION) rather than in
      # INTERFACE_LINK_LIBRARIES.
      if(
        "${_target}" MATCHES "^(adios2|hdf5)::"
        OR
          (
            "${_target}" MATCHES "^Boost::"
            AND NOT "${_target}" STREQUAL "Boost::headers"
          )
      )
        get_target_property(_libs ${_target} IMPORTED_LOCATION_RELEASE)
        if(NOT _libs)
          get_target_property(_libs ${_target} IMPORTED_LOCATION)
        endif()
        if(_libs)
          list(APPEND PKGCONFIG_DOLFINX_LIBS ${_libs})
        endif()
      endif()
    endforeach()

    # Join include lists and remove duplicates
    list(REMOVE_DUPLICATES PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES)
    list(REMOVE_DUPLICATES PKGCONFIG_DOLFINX_LIBS)

    # Convert include dirs to -I<incdir> form. Reversed only to keep the
    # emitted flag order the same as the accumulating loop this replaced.
    set(PKG_INCLUDES ${PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES})
    list(REVERSE PKG_INCLUDES)
    list(TRANSFORM PKG_INCLUDES PREPEND "-I")
    list(JOIN PKG_INCLUDES " " PKG_INCLUDES)

    # Get dolfinx definitions
    get_target_property(
      PKG_DOLFINX_DEFINITIONS
      ${target}
      INTERFACE_COMPILE_DEFINITIONS
    )
    set(PKG_DEFINITIONS ${PKG_DOLFINX_DEFINITIONS})
    list(TRANSFORM PKG_DEFINITIONS PREPEND "-D")
    list(JOIN PKG_DEFINITIONS " " PKG_DEFINITIONS)

    # Convert compiler flags and definitions into space separated strings
    string(REPLACE ";" " " PKG_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE ";" " " PKG_LINKFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

    dolfinx_pkgconfig_libs_to_flags(PKGCONFIG_DOLFINX_LIBS PKG_LINKFLAGS)
    set(PKG_LIBS_PRIVATE "")
    dolfinx_pkgconfig_libs_to_flags(
      PKGCONFIG_DOLFINX_LIBS_PRIVATE
      PKG_LIBS_PRIVATE
    )
  endblock()

  # Configure and install pkg-config file
  configure_file(
    ${DOLFINX_SOURCE_DIR}/cmake/templates/dolfinx.pc.in
    ${PROJECT_BINARY_DIR}/dolfinx.pc
    @ONLY
  )
  install(
    FILES ${PROJECT_BINARY_DIR}/dolfinx.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    COMPONENT Development
  )
endfunction()
