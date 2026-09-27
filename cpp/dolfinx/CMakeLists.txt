# ------------------------------------------------------------------------------
# Declare the library (target)
add_library(dolfinx)
add_library(dolfinx::dolfinx ALIAS dolfinx)
target_compile_features(dolfinx PUBLIC cxx_std_20)
set_target_properties(dolfinx PROPERTIES CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Configure the common/version.h file
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/common/version.h.in
  common/version.h
  @ONLY
)

# ------------------------------------------------------------------------------
# Declare the header file sets
#
# The base directories of a file set are added to the build interface and
# mapped onto the install DESTINATION given in install(TARGETS) below, so
# include directories do not have to be maintained separately.
#
# Three sets are needed because the base directories of a single set may
# not be sub-directories of one another, and the build tree is allowed to
# sit inside the source tree. Base directories of *different* sets may
# nest, which is what keeps this legal for an in-source build directory.
#
# Sub-directories append to the main set with
# target_sources(dolfinx PUBLIC FILE_SET HEADERS FILES ...).
target_sources(dolfinx PUBLIC FILE_SET HEADERS BASE_DIRS ${DOLFINX_SOURCE_DIR})

# Generated headers, so that <dolfinx/common/version.h> resolves in the
# build tree for the library and for anything linking to it.
target_sources(
  dolfinx
  PUBLIC
    FILE_SET generated_headers
      TYPE HEADERS
      BASE_DIRS ${PROJECT_BINARY_DIR}
      FILES ${CMAKE_CURRENT_BINARY_DIR}/common/version.h
)

# The umbrella <dolfinx.h> sits at the root of the include directory rather
# than in dolfinx/, so it needs its own base directory too.
target_sources(
  dolfinx
  PUBLIC
    FILE_SET umbrella_header
      TYPE HEADERS
      BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
      FILES dolfinx.h
)

# ------------------------------------------------------------------------------
# Add sources and headers to the target
add_subdirectory(common)
add_subdirectory(fem)
add_subdirectory(geometry)
add_subdirectory(graph)
add_subdirectory(io)
add_subdirectory(la)
add_subdirectory(mesh)
add_subdirectory(nls)
add_subdirectory(refinement)

# ------------------------------------------------------------------------------
# Set target properties
set_target_properties(
  dolfinx
  PROPERTIES
    VERSION ${DOLFINX_VERSION}
    SOVERSION ${DOLFINX_VERSION_MAJOR}.${DOLFINX_VERSION_MINOR}
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# Add git revision and UFCx signature
target_compile_definitions(
  dolfinx
  PUBLIC
    UFCX_SIGNATURE="${UFCX_SIGNATURE}"
    DOLFINX_GIT_COMMIT_HASH="${GIT_COMMIT_HASH}"
)

# ------------------------------------------------------------------------------
# Set compiler options and definitions

# Set 'Developer' build type flags
target_compile_options(
  dolfinx
  PRIVATE
    $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_FLAGS}>
)

# Set 'Developer' compile definitions
target_compile_definitions(
  dolfinx
  PRIVATE
    $<$<AND:$<CONFIG:Developer>,$<COMPILE_LANGUAGE:CXX>>:${DOLFINX_CXX_DEVELOPER_DEFINITIONS}>
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
if(TARGET UFCxPython::UFCxPython)
  target_link_libraries(dolfinx PUBLIC UFCxPython::UFCxPython)
elseif(TARGET ufcx::ufcx)
  target_link_libraries(dolfinx PUBLIC ufcx::ufcx)
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
if(_DOLFINX_USE_ADIOS2 AND ADIOS2_FOUND)
  # ADIOS2 2.11 and later
  if(TARGET adios2::cxx_mpi)
    set(_adios2_mpi_target adios2::cxx_mpi)
    # And prior
  elseif(TARGET adios2::cxx11_mpi)
    set(_adios2_mpi_target adios2::cxx11_mpi)
  else()
    message(FATAL_ERROR "ADIOS2 found, but no adios2::cxx*_mpi target found.")
  endif()
  target_link_libraries(dolfinx PUBLIC ${_adios2_mpi_target})
  target_compile_definitions(dolfinx PUBLIC HAS_ADIOS2)
endif()

# PETSc
if(_DOLFINX_USE_PETSC AND TARGET PkgConfig::PETSC)
  target_link_libraries(dolfinx PUBLIC PkgConfig::PETSC)
  target_compile_definitions(dolfinx PUBLIC HAS_PETSC)
endif()

# SLEPC
if(_DOLFINX_USE_SLEPC AND TARGET PkgConfig::SLEPC)
  target_link_libraries(dolfinx PUBLIC PkgConfig::SLEPC)
  target_compile_definitions(dolfinx PUBLIC HAS_SLEPC)
endif()

# SCOTCH
if(_DOLFINX_USE_SCOTCH AND TARGET SCOTCH::ptscotch)
  target_compile_definitions(dolfinx PUBLIC HAS_PTSCOTCH)
  target_link_libraries(dolfinx PRIVATE SCOTCH::ptscotch)
  if(TARGET SCOTCH::scotcherr)
    target_link_libraries(dolfinx PRIVATE SCOTCH::scotcherr)
  endif()
endif()

# ParMETIS
if(_DOLFINX_USE_PARMETIS AND TARGET ParMETIS::ParMETIS)
  target_compile_definitions(dolfinx PUBLIC HAS_PARMETIS)
  target_link_libraries(dolfinx PRIVATE ParMETIS::ParMETIS)
endif()

# KaHIP
if(_DOLFINX_USE_KAHIP AND TARGET KaHIP::KaHIP)
  target_compile_definitions(dolfinx PUBLIC HAS_KAHIP)
  target_link_libraries(dolfinx PRIVATE KaHIP::KaHIP)
endif()

# SuperLU_DIST
if(_DOLFINX_USE_SUPERLU_DIST AND TARGET PkgConfig::SUPERLU_DIST)
  target_compile_definitions(dolfinx PUBLIC HAS_SUPERLU_DIST)
  target_link_libraries(dolfinx PRIVATE PkgConfig::SUPERLU_DIST)
endif()

# ------------------------------------------------------------------------------
# Install dolfinx library and header files

# Add all discovered linker paths to install rpath?
set_target_properties(
  dolfinx
  PROPERTIES INSTALL_RPATH_USE_LINK_PATH ${DOLFINX_INSTALL_RPATH_USE_LINK_PATH}
)

install(
  TARGETS dolfinx
  EXPORT DOLFINXTargets
  RUNTIME_DEPENDENCY_SET dependencies
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Development
  FILE_SET generated_headers
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT Development
  FILE_SET umbrella_header
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT Development
)

if(INSTALL_RUNTIME_DEPENDENCIES AND WIN32)
  install(
    RUNTIME_DEPENDENCY_SET dependencies DESTINATION ${CMAKE_INSTALL_BINDIR}
    PRE_EXCLUDE_REGEXES [[api-ms-win-.*]] [[ext-ms-.*]]
    POST_EXCLUDE_REGEXES [[.*(\\|/)system32(\\|/).*\.dll]]
    DIRECTORIES $<TARGET_FILE_DIR:Basix::basix> $<TARGET_FILE_DIR:hdf5::hdf5>
  )
endif()

# Generate DOLFINTargets.cmake
install(
  EXPORT DOLFINXTargets
  NAMESPACE dolfinx::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx
  COMPONENT Development
)

# ------------------------------------------------------------------------------
# Generate CMake config files (DOLFINXConfig{,Version}.cmake)
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/dolfinx/DOLFINXConfigVersion.cmake
  VERSION ${DOLFINX_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
  ${DOLFINX_SOURCE_DIR}/cmake/templates/DOLFINXConfig.cmake.in
  ${PROJECT_BINARY_DIR}/dolfinx/DOLFINXConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx
)

# Install CMake helper files
install(
  FILES
    ${PROJECT_BINARY_DIR}/dolfinx/DOLFINXConfig.cmake
    ${PROJECT_BINARY_DIR}/dolfinx/DOLFINXConfigVersion.cmake
    ${DOLFINX_SOURCE_DIR}/cmake/modules/FindUFCxPython.cmake
    ${DOLFINX_SOURCE_DIR}/cmake/modules/DolfinxDeveloperCompilerFlags.cmake
    ${DOLFINX_SOURCE_DIR}/cmake/modules/DolfinxDemo.cmake
    ${DOLFINX_SOURCE_DIR}/cmake/modules/DolfinxPkgConfigHelpers.cmake
    ${DOLFINX_SOURCE_DIR}/cmake/modules/FindParMETIS.cmake
    ${DOLFINX_SOURCE_DIR}/cmake/modules/FindKaHIP.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/dolfinx
  COMPONENT Development
)

# ------------------------------------------------------------------------------
# Generate pkg-config file and install it
include(DolfinxPkgConfigGenerate)
dolfinx_pkgconfig_generate(dolfinx)

# ------------------------------------------------------------------------------
