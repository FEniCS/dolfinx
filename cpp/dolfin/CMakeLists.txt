#------------------------------------------------------------------------------
include(GNUInstallDirs)

#------------------------------------------------------------------------------
# Configure the dolfin/common/version.h file

# NOTE: This modifies a file in the source directory, which probably isn't good
configure_file(${DOLFIN_SOURCE_DIR}/dolfin/common/version.h.in
  ${DOLFIN_SOURCE_DIR}/dolfin/common/version.h @ONLY)

#------------------------------------------------------------------------------
# Delcare the library (target)

add_library(dolfin "")  # The "" is needed for older CMake. Remove later.

#------------------------------------------------------------------------------
# Add source files to the target

set(DOLFIN_DIRS common fem function generation geometry graph io la mesh nls
  refinement)

install(FILES dolfin.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT Development)

# Add source to dolfin target, and get sets of header files
foreach(DIR ${DOLFIN_DIRS})
  add_subdirectory(${DIR})
endforeach()

# Set target include location (for build and installed)
target_include_directories(dolfin PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${DOLFIN_SOURCE_DIR};${DOLFIN_SOURCE_DIR}/dolfin>")

#------------------------------------------------------------------------------
# Set target properties

set_target_properties(dolfin PROPERTIES
  VERSION ${DOLFIN_VERSION}
  SOVERSION ${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR})

# Add git revision flag to the one affected file
set_source_files_properties(common/defines.cpp PROPERTIES
  COMPILE_DEFINITIONS "UFC_SIGNATURE=\"${UFC_SIGNATURE}\";DOLFIN_GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

#------------------------------------------------------------------------------
# Set compiler options and defintions

# Set 'Developer' build type flags
target_compile_options(dolfin PRIVATE $<$<CONFIG:Developer>:${DOLFIN_CXX_DEVELOPER_FLAGS}>)

# Set debug definitions (private)
target_compile_definitions(dolfin PRIVATE $<$<OR:$<CONFIG:Debug>,$<CONFIG:Developer>>:DEBUG>)

# Add version to definitions (public)
target_compile_definitions(dolfin PUBLIC DOLFIN_VERSION="${DOLFIN_VERSION}")

#------------------------------------------------------------------------------
# Add include directories and libraries of required packages

# UFC
target_include_directories(dolfin SYSTEM PUBLIC ${UFC_INCLUDE_DIRS})

# DOLFIN uses Eigen data structures for dense linear algebra operations. Eigen
# performs 'ideal' memory alignment based around the -march flag passed to the
# compiler.  However, because Python DOLFIN JIT compiles code at runtime, it is
# possible for the user to build shared objects with incompatible alignment
# (ABI) if they use a different -march flag than that used to originally build
# DOLFIN. DOLFIN_EIGEN_MAX_ALIGN_BYTES can be used to force alignment.
# See: https://eigen.tuxfamily.org/dox/TopicPreprocessorDirectives.html
# See: https://github.com/FEniCS/dolfinx/pull/143

# Note: The name EIGEN_MAX_ALIGN_BYTES is confusing. In practice, Eigen
# computes the ideal alignment based around -march.  If the ideal alignment is
# greater than EIGEN_MAX_ALIGN_BYTES, the ideal alignment is used. If the ideal
# alignment is less, then EIGEN_MAX_ALIGN_BYTES is used for alignment.
set(DOLFIN_EIGEN_MAX_ALIGN_BYTES "32" CACHE STRING "\
Minimum alignment in bytes used for Eigen data structures. Set to 32 for \
compatibility with AVX user-compiled code and 64 for AVX-512 user-compiled \
code. Set to 0 for ideal alignment according to -march. Note that if an architecture \
flag (e.g. \"-march=skylake-avx512\") is set for DOLFIN, Eigen will use the \
appropriate ideal alignment instead if it is stricter. Otherwise, the value \
of this variable will be used by Eigen for the alignment of all data structures.\\
")

# Eigen3
target_include_directories(dolfin SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_compile_definitions(dolfin PUBLIC "EIGEN_MAX_ALIGN_BYTES=${DOLFIN_EIGEN_MAX_ALIGN_BYTES}")

# Boost
target_link_libraries(dolfin PUBLIC Boost::boost)
foreach (BOOST_PACKAGE ${DOLFIN_BOOST_COMPONENTS_PUBLIC})
  target_link_libraries(dolfin PUBLIC "Boost::${BOOST_PACKAGE}")
endforeach()
foreach (BOOST_PACKAGE ${DOLFIN_BOOST_COMPONENTS_PRIVATE})
  target_link_libraries(dolfin PRIVATE "Boost::${BOOST_PACKAGE}")
endforeach()

# MPI
target_link_libraries(dolfin PUBLIC MPI::MPI_CXX)

# PETSc
target_link_libraries(dolfin PUBLIC PETSC::petsc)
target_link_libraries(dolfin PRIVATE PETSC::petsc_static)

# HDF5
target_compile_definitions(dolfin PUBLIC ${HDF5_DEFINITIONS})
target_link_libraries(dolfin PUBLIC ${HDF5_C_LIBRARIES})
target_include_directories(dolfin SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS})

# SCOTCH
target_link_libraries(dolfin PRIVATE ${SCOTCH_LIBRARIES})
target_include_directories(dolfin SYSTEM PRIVATE ${SCOTCH_INCLUDE_DIRS})

#------------------------------------------------------------------------------
# Optional packages

# SLEPC
if (DOLFIN_ENABLE_SLEPC AND SLEPC_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_SLEPC)
  target_link_libraries(dolfin PUBLIC SLEPC::slepc)
  target_link_libraries(dolfin PRIVATE SLEPC::slepc_static)
endif()

# ParMETIS
if (DOLFIN_ENABLE_PARMETIS AND PARMETIS_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_PARMETIS)
  target_link_libraries(dolfin PRIVATE ${PARMETIS_LIBRARIES})
  target_include_directories(dolfin SYSTEM PRIVATE ${PARMETIS_INCLUDE_DIRS})
endif()

# KaHIP
if (DOLFIN_ENABLE_KAHIP AND KAHIP_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_KAHIP)
  target_link_libraries(dolfin PRIVATE ${KAHIP_LIBRARIES})
  target_include_directories(dolfin SYSTEM PRIVATE ${KAHIP_INCLUDE_DIRS})
endif()

#------------------------------------------------------------------------------
# Install dolfin library and header files

install(TARGETS dolfin
  EXPORT DOLFINTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  )

# Generate DOLFINTargets.cmake
install(EXPORT DOLFINTargets DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfin/cmake)

# Install the header files
foreach(DIR ${DOLFIN_DIRS})
  install(FILES ${HEADERS_${DIR}} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfin/${DIR}
    COMPONENT Development)
endforeach()

#------------------------------------------------------------------------------
# Generate CMake config files (DOLFINConfig{,Version}.cmake)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_BINARY_DIR}/dolfin/DOLFINConfigVersion.cmake
  VERSION ${DOLFIN_VERSION}
  COMPATIBILITY ExactVersion)

configure_package_config_file(${DOLFIN_CMAKE_DIR}/templates/DOLFINConfig.cmake.in
${CMAKE_BINARY_DIR}/dolfin/DOLFINConfig.cmake
INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfin/cmake)

# Install CMake helper files
install(
  FILES
  ${CMAKE_SOURCE_DIR}/cmake/modules/FindPETSc.cmake
  ${CMAKE_SOURCE_DIR}/cmake/modules/FindSLEPc.cmake
  ${CMAKE_BINARY_DIR}/dolfin/DOLFINConfig.cmake
  ${CMAKE_BINARY_DIR}/dolfin/DOLFINConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfin/cmake
  COMPONENT Development)

#------------------------------------------------------------------------------
# Generate pkg-config file and install it

# Define packages that should be required by pkg-config file
set(PKG_REQUIRES "")

# Get link libraries and includes
get_target_property(PKGCONFIG_DOLFIN_TARGET_LINK_LIBRARIES dolfin INTERFACE_LINK_LIBRARIES)
get_target_property(PKGCONFIG_DOLFIN_INCLUDE_DIRECTORIES dolfin INTERFACE_INCLUDE_DIRECTORIES)

# Add imported targets to lists for creating pkg-config file
set(PKGCONFIG_DOLFIN_LIBS)
foreach(_target ${PKGCONFIG_DOLFIN_TARGET_LINK_LIBRARIES})

  if ("${_target}" MATCHES "^.*::.*$")
    # Get include paths
    get_target_property(_inc_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
    if (_inc_dirs)
      list(APPEND PKGCONFIG_DOLFIN_INCLUDE_DIRECTORIES ${_inc_dirs})
    endif()

    # Get libraries
    get_target_property(_libs ${_target} INTERFACE_LINK_LIBRARIES)
    if (_libs)
      list(APPEND PKGCONFIG_DOLFIN_LIBS ${_libs})
    endif()

  else()
    # 'regular' libs, i.e. not imported targets
    list(APPEND PKGCONFIG_DOLFIN_LIBS ${_target})
  endif()

  # Special handling for compiled Boost imported targets
  if (("${_target}" MATCHES "^.*Boost::.*$") AND NOT "${_target}" STREQUAL "Boost::boost")
    get_target_property(_libs ${_target} IMPORTED_LOCATION_RELEASE)
    if (_libs)
      list(APPEND PKGCONFIG_DOLFIN_LIBS ${_libs})
    endif()
  endif()

endforeach()

# Join include lists and remove duplicates
list(REMOVE_DUPLICATES PKGCONFIG_DOLFIN_INCLUDE_DIRECTORIES)
list(REMOVE_DUPLICATES PKGCONFIG_DOLFIN_LIBS)

# Convert include dirs to -I<incdir> form
foreach(_inc_dir ${PKGCONFIG_DOLFIN_INCLUDE_DIRECTORIES})
  set(PKG_INCLUDES "-I${_inc_dir} ${PKG_INCLUDES}")
endforeach()

# Get dolfin definitions
get_target_property(PKG_DOLFIN_DEFINITIONS dolfin INTERFACE_COMPILE_DEFINITIONS)
set(PKG_DEFINITIONS)
foreach(_def ${PKG_DOLFIN_DEFINITIONS})
    set(PKG_DEFINITIONS "${PKG_DEFINITIONS} -D${_def}")
endforeach()

# Convert compiler flags and definitions into space separated strings
string(REPLACE ";" " " PKG_CXXFLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE ";" " " PKG_LINKFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# Convert libraries to -L<libdir> -l<lib> form
foreach(_lib ${PKGCONFIG_DOLFIN_LIBS})
  # Add -Wl,option directives
  if ("${_lib}" MATCHES "-Wl,[^ ]*")
    set(PKG_LINKFLAGS "${_lib} ${PKG_LINKFLAGS}")
  else()
    string(REGEX REPLACE "(.?:?/[^ ]*)/lib([^ ]*)\\.(a|so|dylib|dll)" "-L\\1 -l\\2"
      _linkflags
      "${_lib}"
      )

    # Add libraries that matches the form -L<libdir> -l<lib>
    if ("${_linkflags}" MATCHES "-L.+ -l.+")
      set(PKG_LINKFLAGS "${_linkflags} ${PKG_LINKFLAGS}")
    endif()
  endif()
endforeach()

# Remove duplicated link flags
separate_arguments(PKG_LINKFLAGS)
list(REMOVE_DUPLICATES PKG_LINKFLAGS)
string(REPLACE ";" " " PKG_LINKFLAGS "${PKG_LINKFLAGS}")

# Add additional link flags
foreach(_linkflag ${DOLFIN_LINK_FLAGS})
  set(PKG_LINKFLAGS "${PKG_LINKFLAGS} ${_linkflag}")
endforeach()

# Boost include dir (used as pkg-config variable)
get_target_property(BOOST_INCLUDE_DIR Boost::boost INTERFACE_INCLUDE_DIRECTORIES)

# Configure and install pkg-config file
configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin.pc.in ${CMAKE_BINARY_DIR}/dolfin.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfin.pc
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
  COMPONENT Development
  )
#------------------------------------------------------------------------------
