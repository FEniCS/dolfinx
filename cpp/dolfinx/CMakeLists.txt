# ------------------------------------------------------------------------------
include(GNUInstallDirs)

# ------------------------------------------------------------------------------
# Declare the library (target) and set C++ standard
add_library(dolfinx)
set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(dolfinx PUBLIC cxx_std_20)

# ------------------------------------------------------------------------------
# Add source files to the target
set(DOLFINX_DIRS
    common
    fem
    geometry
    graph
    io
    la
    mesh
    nls
    refinement
)

# Add source to dolfinx target, and get sets of header files
foreach(DIR ${DOLFINX_DIRS})
  add_subdirectory(${DIR})
endforeach()

# Set target include location (for build and installed)
target_include_directories(
  dolfinx
  PUBLIC
    $<INSTALL_INTERFACE:include>
    "$<BUILD_INTERFACE:${DOLFINX_SOURCE_DIR};${DOLFINX_SOURCE_DIR}/dolfinx>"
)

# ------------------------------------------------------------------------------
# Configure the common/version.h file
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/common/version.h.in common/version.h @ONLY
)

# ------------------------------------------------------------------------------
# Set target properties
set_target_properties(
  dolfinx
  PROPERTIES VERSION ${DOLFINX_VERSION}
             SOVERSION ${DOLFINX_VERSION_MAJOR}.${DOLFINX_VERSION_MINOR}
)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(dolfinx PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY};--config-file=${CMAKE_CURRENT_SOURCE_DIR}/../../.clang-tidy")
endif()

# Add git revision flag to the one affected file
set_source_files_properties(
  common/defines.cpp
  PROPERTIES
    COMPILE_DEFINITIONS
    "UFCX_SIGNATURE=\"${UFCX_SIGNATURE}\";DOLFINX_GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\""
)

# ------------------------------------------------------------------------------
# Set compiler options and definitions

# Set 'Developer' build type flags
target_compile_options(
  dolfinx PRIVATE $<$<CONFIG:Developer>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
)

# Add version to definitions (public)
target_compile_definitions(dolfinx PUBLIC DOLFINX_VERSION="${DOLFINX_VERSION}")

# MSVC does not support the optional C99 _Complex type. Consequently, ufcx.h
# does not contain tabulate_tensor_complex* functions when built with MSVC. On
# MSVC this DOLFINX macro is set and this removes all calls to
# tabulate_tensor_complex*.
if(MSVC)
  target_compile_definitions(dolfinx PUBLIC DOLFINX_NO_STDC_COMPLEX_KERNELS)
endif()

# ------------------------------------------------------------------------------
# Add include directories and libraries of required packages

# pugixml (see https://pugixml.org/docs/manual.html#v1.11)
if(TARGET pugixml::pugixml)
  target_link_libraries(dolfinx PUBLIC pugixml::pugixml)
else()
  target_link_libraries(dolfinx PUBLIC pugixml)
endif()

# UFCx
if(TARGET ufcx::ufcx)
  target_link_libraries(dolfinx PUBLIC ufcx::ufcx)
else()
  target_include_directories(dolfinx SYSTEM PUBLIC ${UFCX_INCLUDE_DIRS})
endif()

# Basix
target_link_libraries(dolfinx PUBLIC Basix::basix)

# Boost
target_link_libraries(dolfinx PUBLIC Boost::headers)

# MPI
target_link_libraries(dolfinx PUBLIC MPI::MPI_CXX)

target_link_libraries(dolfinx PUBLIC spdlog::spdlog)

# HDF5
target_link_libraries(dolfinx PUBLIC hdf5::hdf5)

# ------------------------------------------------------------------------------
# Optional packages

# ADIOS2
if(ADIOS2_FOUND)
  target_compile_definitions(dolfinx PUBLIC HAS_ADIOS2)
  target_link_libraries(dolfinx PUBLIC adios2::cxx11_mpi)
endif()

# PETSc
if(DOLFINX_ENABLE_PETSC AND PETSC_FOUND)
  target_link_libraries(dolfinx PUBLIC PkgConfig::PETSC)
  target_compile_definitions(dolfinx PUBLIC HAS_PETSC)
endif()

# SLEPC
if(DOLFINX_ENABLE_SLEPC AND SLEPC_FOUND)
  target_link_libraries(dolfinx PUBLIC PkgConfig::SLEPC)
  target_compile_definitions(dolfinx PUBLIC HAS_SLEPC)
endif()

# SCOTCH
if(DOLFINX_ENABLE_SCOTCH AND SCOTCH_FOUND)
  target_compile_definitions(dolfinx PUBLIC HAS_PTSCOTCH)
  target_link_libraries(dolfinx PRIVATE SCOTCH::ptscotch)
  if(TARGET SCOTCH::scotcherr)
    target_link_libraries(dolfinx PRIVATE SCOTCH::scotcherr)
  endif()
endif()

# ParMETIS
if(DOLFINX_ENABLE_PARMETIS AND PARMETIS_FOUND)
  target_compile_definitions(dolfinx PUBLIC HAS_PARMETIS)
  target_link_libraries(dolfinx PRIVATE ${PARMETIS_LIBRARIES})
  target_include_directories(dolfinx SYSTEM PRIVATE ${PARMETIS_INCLUDE_DIRS})
endif()

# KaHIP
if(DOLFINX_ENABLE_KAHIP AND KAHIP_FOUND)
  target_compile_definitions(dolfinx PUBLIC HAS_KAHIP)
  target_link_libraries(dolfinx PRIVATE ${KAHIP_LIBRARIES})
  target_include_directories(dolfinx SYSTEM PRIVATE ${KAHIP_INCLUDE_DIRS})
endif()

# ------------------------------------------------------------------------------
# Install dolfinx library and header files
if(WIN32)
  install(
    TARGETS dolfinx
    EXPORT DOLFINXTargets
    RUNTIME_DEPENDENCY_SET dependencies
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  )

  if(INSTALL_RUNTIME_DEPENDENCIES)
    list(APPEND PRE_EXCLUDE_REGEXES "api-ms-.*")
    list(APPEND PRE_EXCLUDE_REGEXES "ext-ms-.*")
    install(
      RUNTIME_DEPENDENCY_SET
      dependencies
      DESTINATION
      ${CMAKE_INSTALL_BINDIR}
      PRE_EXCLUDE_REGEXES
      ${PRE_EXCLUDE_REGEXES}
      POST_EXCLUDE_REGEXES
      ".*system32/.*\\.dll"
      DIRECTORIES
      $<TARGET_FILE_DIR:Basix::basix>
    )
  endif()
else()
  install(
    TARGETS dolfinx
    EXPORT DOLFINXTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  )
endif()

# Generate DOLFINTargets.cmake
install(EXPORT DOLFINXTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx)

# Install the header files
install(
  FILES dolfinx.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT Development
)

foreach(DIR ${DOLFINX_DIRS})
  install(
    FILES ${HEADERS_${DIR}}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx/${DIR}
    COMPONENT Development
  )
endforeach()

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/common/version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx/common
  COMPONENT Development
)

# ------------------------------------------------------------------------------
# Generate CMake config files (DOLFINXConfig{,Version}.cmake)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/dolfinx/DOLFINXConfigVersion.cmake
  VERSION ${DOLFINX_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
  ${DOLFINX_SOURCE_DIR}/cmake/templates/DOLFINXConfig.cmake.in
  ${CMAKE_BINARY_DIR}/dolfinx/DOLFINXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx
)

# Install CMake helper files
install(
  FILES ${CMAKE_BINARY_DIR}/dolfinx/DOLFINXConfig.cmake
        ${CMAKE_BINARY_DIR}/dolfinx/DOLFINXConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx
  COMPONENT Development
)

# ------------------------------------------------------------------------------
# Generate pkg-config file and install it

# Define packages that should be required by pkg-config file
set(PKG_REQUIRES "")

# Get link libraries and includes
get_target_property(
  PKGCONFIG_DOLFINX_TARGET_LINK_LIBRARIES dolfinx INTERFACE_LINK_LIBRARIES
)
get_target_property(
  PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES dolfinx
  INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
)

# Add imported targets to lists for creating pkg-config file
set(PKGCONFIG_DOLFINX_LIBS)

foreach(_target ${PKGCONFIG_DOLFINX_TARGET_LINK_LIBRARIES})
  if("${_target}" MATCHES "^[^<>]+$") # Skip "$<foo...>", which we get with
                                      # static libs
    if("${_target}" MATCHES "^.*::.*$")
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

    # Special handling to extract data for compiled Boost imported
    # targets to use in generate pkg-config file
    if(("${_target}" MATCHES "^.*Boost::.*$") AND NOT "${_target}" STREQUAL
                                                  "Boost::headers"
    )
      get_target_property(_libs ${_target} IMPORTED_LOCATION_RELEASE)

      if(_libs)
        list(APPEND PKGCONFIG_DOLFINX_LIBS ${_libs})
      endif()
    endif()
  endif()
endforeach()

# Join include lists and remove duplicates
list(REMOVE_DUPLICATES PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES)
list(REMOVE_DUPLICATES PKGCONFIG_DOLFINX_LIBS)

# Convert include dirs to -I<incdir> form
foreach(_inc_dir ${PKGCONFIG_DOLFINX_INCLUDE_DIRECTORIES})
  set(PKG_INCLUDES "-I${_inc_dir} ${PKG_INCLUDES}")
endforeach()

# Get dolfinx definitions
get_target_property(
  PKG_DOLFINX_DEFINITIONS dolfinx INTERFACE_COMPILE_DEFINITIONS
)
set(PKG_DEFINITIONS)

foreach(_def ${PKG_DOLFINX_DEFINITIONS})
  set(PKG_DEFINITIONS "${PKG_DEFINITIONS} -D${_def}")
endforeach()

# Get basix definitions (this is required to propagate Basix definition to the
# pkg-config file, in the future Basix should create its own basix.pc file, see
# https://github.com/FEniCS/basix/issues/204)
get_target_property(
  PKG_BASIX_DEFINITIONS Basix::basix INTERFACE_COMPILE_DEFINITIONS
)

foreach(_def ${PKG_BASIX_DEFINITIONS})
  set(PKG_DEFINITIONS "${PKG_DEFINITIONS} -D${_def}")
endforeach()

# Convert compiler flags and definitions into space separated strings
string(REPLACE ";" " " PKG_CXXFLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE ";" " " PKG_LINKFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Convert libraries to -L<libdir> -l<lib> form
foreach(_lib ${PKGCONFIG_DOLFINX_LIBS})
  # Add -Wl,option directives
  if("${_lib}" MATCHES "-Wl,[^ ]*")
    set(PKG_LINKFLAGS "${_lib} ${PKG_LINKFLAGS}")
  else()
    get_filename_component(_path ${_lib} DIRECTORY)
    get_filename_component(_name ${_lib} NAME_WE)
    string(REPLACE "lib" "" _name "${_name}")

    # Add libraries that matches the form -L<libdir> -l<lib>
    if(NOT "${_path}" STREQUAL "")
      set(PKG_LINKFLAGS "-L${_path} -l${_name} ${PKG_LINKFLAGS}")
    endif()
  endif()
endforeach()

# Remove duplicated link flags
separate_arguments(PKG_LINKFLAGS)
list(REMOVE_DUPLICATES PKG_LINKFLAGS)
string(REPLACE ";" " " PKG_LINKFLAGS "${PKG_LINKFLAGS}")

# Add additional link flags
foreach(_linkflag ${DOLFINX_LINK_FLAGS})
  set(PKG_LINKFLAGS "${PKG_LINKFLAGS} ${_linkflag}")
endforeach()

# Boost include dir (used as pkg-config variable)
get_target_property(
  BOOST_INCLUDE_DIR Boost::headers INTERFACE_INCLUDE_DIRECTORIES
)

# Configure and install pkg-config file
configure_file(
  ${DOLFINX_SOURCE_DIR}/cmake/templates/dolfinx.pc.in
  ${CMAKE_BINARY_DIR}/dolfinx.pc @ONLY
)
install(
  FILES ${CMAKE_BINARY_DIR}/dolfinx.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  COMPONENT Development
)

# ------------------------------------------------------------------------------
