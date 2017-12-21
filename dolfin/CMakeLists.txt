set(DOLFIN_H dolfin.h)
install(FILES ${DOLFIN_H} DESTINATION ${DOLFIN_INCLUDE_DIR}
  COMPONENT Development)

# Configure and install dolfin/common/version.h file
string(REPLACE "+" "" DOLFIN_VERSION_MICRO_STRIPPED ${DOLFIN_VERSION_MICRO})
configure_file(${DOLFIN_SOURCE_DIR}/dolfin/common/version.h.in
  ${CMAKE_BINARY_DIR}/dolfin/common/version.h @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfin/common/version.h
  DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/common/
  COMPONENT Development
  )
include_directories(${CMAKE_BINARY_DIR})

#------------------------------------------------------------------------------
# DOLFIN source directories

# All files and directories in this directory
file(GLOB DOLFIN_CONTENT RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

# Initialize list
set(DOLFIN_DIRS)

# List of excluded files and directories
set(EXCLUDES ${DOLFIN_H} CMakeLists.txt)

# Iterate over all directories and files and append to DOLFIN_DIRS
foreach(_DIR_NAME ${DOLFIN_CONTENT})
  list(FIND EXCLUDES ${_DIR_NAME} INDEX_EXCLUDES)
  if (${INDEX_EXCLUDES} LESS 0)
    list(APPEND DOLFIN_DIRS ${_DIR_NAME})
  endif()
endforeach()

#------------------------------------------------------------------------------
# Install header files

# Initialize lists
set(DOLFIN_HEADERS)
set(DOLFIN_SOURCES)

foreach(DIR ${DOLFIN_DIRS})
  # Each subdirectory defines HEADERS and SOURCES
  add_subdirectory(${DIR})
  set(HEADERS_FULL_PATH)
  foreach(HEADER_FILE ${HEADERS})
    list(APPEND HEADERS_FULL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/${HEADER_FILE})
  endforeach()
  install(FILES ${HEADERS_FULL_PATH} DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR}
    COMPONENT Development)
  list(APPEND DOLFIN_HEADERS ${HEADERS_FULL_PATH})
  foreach(SOURCE_FILE ${SOURCES})
    list(APPEND DOLFIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/${SOURCE_FILE})
  endforeach()
endforeach()

#------------------------------------------------------------------------------
# Define libraries

add_library(dolfin ${DOLFIN_H} ${DOLFIN_HEADERS} ${DOLFIN_SOURCES})
set_target_properties(dolfin PROPERTIES ${DOLFIN_LIBRARY_PROPERTIES})

#------------------------------------------------------------------------------
# General compiler defintions

if (DOLFIN_DEPRECATION_ERROR)
  target_compile_definitions(dolfin PUBLIC DOLFIN_DEPRECATION_ERROR)
endif()

# Set 'Developer' build type flags
set(CMAKE_CXX_FLAGS_DEVELOPER "${DOLFIN_CXX_DEVELOPER_FLAGS}" CACHE STRING
  "Flags used by the compiler during development." FORCE)

# FIXME: Do we want to add -DDEBUG to RelWithDebInfo?

# Set debug definitions
if (CMAKE_BUILD_TYPE STREQUAL "Developer" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(dolfin PUBLIC DEBUG)
else()
  target_compile_definitions(dolfin PUBLIC NDEBUG)
endif()

#------------------------------------------------------------------------------
# Add include directories and libs of required packages

# UFC
target_include_directories(dolfin SYSTEM PUBLIC ${UFC_INCLUDE_DIRS})

# Eigen3
target_include_directories(dolfin SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})

# Boost
target_link_libraries(dolfin PUBLIC Boost::boost)
foreach (BOOST_PACKAGE ${DOLFIN_BOOST_COMPONENTS_PUBLIC})
  target_link_libraries(dolfin PUBLIC "Boost::${BOOST_PACKAGE}")
endforeach()
foreach (BOOST_PACKAGE ${DOLFIN_BOOST_COMPONENTS_PRIVATE})
  target_link_libraries(dolfin PRIVATE "Boost::${BOOST_PACKAGE}")
endforeach()

#------------------------------------------------------------------------------
# Optional packages

# HDF5
if (DOLFIN_ENABLE_HDF5 AND HDF5_FOUND)
  #list(APPEND DOLFIN_LINK_FLAGS ${HDF5_LINK_FLAGS})
  target_compile_definitions(dolfin PUBLIC HAS_HDF5)
  target_compile_definitions(dolfin PUBLIC ${HDF5_DEFINITIONS})
  target_link_libraries(dolfin PUBLIC ${HDF5_C_LIBRARIES})
  target_include_directories(dolfin SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS})
endif()

# Trilinos
if (DOLFIN_ENABLE_TRILINOS AND DOLFIN_TRILINOS_PACKAGES_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_TRILINOS)
  target_compile_definitions(dolfin PUBLIC ${DOLFIN_TRILINOS_DEFINITIONS})
  target_link_libraries(dolfin PUBLIC ${DOLFIN_TRILINOS_LIBRARIES})
  target_include_directories(dolfin SYSTEM PUBLIC ${Trilinos_INCLUDE_DIRS})
endif()

# SLEPC
if (DOLFIN_ENABLE_PETSC AND DOLFIN_ENABLE_SLEPC AND SLEPC_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_SLEPC)
  target_link_libraries(dolfin PUBLIC SLEPC::slepc)
  target_link_libraries(dolfin PRIVATE SLEPC::slepc_static)
endif()

# PETSc
if (DOLFIN_ENABLE_PETSC AND PETSC_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_PETSC)
  target_link_libraries(dolfin PUBLIC PETSC::petsc)
  target_link_libraries(dolfin PRIVATE PETSC::petsc_static)
endif()

# UMFPACK
if (DOLFIN_ENABLE_UMFPACK AND UMFPACK_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_UMFPACK)
  target_link_libraries(dolfin PRIVATE ${UMFPACK_LIBRARIES})
  target_include_directories(dolfin SYSTEM PRIVATE ${UMFPACK_INCLUDE_DIRS})
endif()

# CHOLMOD
if (DOLFIN_ENABLE_CHOLMOD AND CHOLMOD_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_CHOLMOD)
  target_link_libraries(dolfin PRIVATE ${CHOLMOD_LIBRARIES})
  target_include_directories(dolfin SYSTEM PRIVATE ${CHOLMOD_INCLUDE_DIRS})
endif()

# SCOTCH
if (DOLFIN_ENABLE_SCOTCH AND SCOTCH_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_SCOTCH)
  target_link_libraries(dolfin PRIVATE ${SCOTCH_LIBRARIES})
  target_include_directories(dolfin SYSTEM PRIVATE ${SCOTCH_INCLUDE_DIRS})
endif()

# ParMETIS
if (DOLFIN_ENABLE_PARMETIS AND PARMETIS_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_PARMETIS)
  target_link_libraries(dolfin PRIVATE ${PARMETIS_LIBRARIES})
  target_include_directories(dolfin SYSTEM PRIVATE ${PARMETIS_INCLUDE_DIRS})
endif()

# ZLIB
if (DOLFIN_ENABLE_ZLIB AND ZLIB_FOUND)
  target_compile_definitions(dolfin PUBLIC HAS_ZLIB)
  target_link_libraries(dolfin PRIVATE ${ZLIB_LIBRARIES})
  target_include_directories(dolfin SYSTEM PRIVATE ${ZLIB_INCLUDE_DIRS})
endif()

# MPI
if (DOLFIN_ENABLE_MPI AND MPI_CXX_FOUND)
  set(DOLFIN_CXX_FLAGS "${DOLFIN_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
  target_compile_definitions(dolfin PUBLIC HAS_MPI)
  target_link_libraries(dolfin PUBLIC ${MPI_CXX_LIBRARIES})
  target_include_directories(dolfin SYSTEM PUBLIC ${MPI_CXX_INCLUDE_PATH})
endif()

if (DOLFIN_ENABLE_GEOMETRY_DEBUGGING AND GMP_FOUND AND MPFR_FOUND AND CGAL_FOUND)
  message(STATUS "Appending link flags for geometry debugging")
  target_compile_definitions(dolfin PUBLIC "-DDOLFIN_ENABLE_GEOMETRY_DEBUGGING")
  target_include_directories(dolfin PRIVATE ${CGAL_INCLUDE_DIRS})
  target_include_directories(dolfin PRIVATE ${GMP_INCLUDE_DIRS})
  target_include_directories(dolfin PRIVATE ${MPFR_INCLUDE_DIRS})
  target_link_libraries(dolfin PRIVATE ${GMP_LIBRARIES})
  target_link_libraries(dolfin PRIVATE ${MPFR_LIBRARIES})
endif()

#------------------------------------------------------------------------------
# Set compiler flags, include directories and library dependencies

# Add compiler include directories
include_directories(BEFORE ${DOLFIN_SOURCE_DIR})

# Add CXX definitions
#add_definitions(${DOLFIN_CXX_DEFINITIONS})
target_compile_definitions(dolfin PUBLIC DOLFIN_VERSION="${DOLFIN_VERSION}")

# Just add git revision flag to the one affected file
set_source_files_properties(common/defines.cpp PROPERTIES COMPILE_FLAGS
  "-DDOLFIN_GIT_COMMIT_HASH=\\\"${GIT_COMMIT_HASH}\\\" -DUFC_SIGNATURE=\\\"${UFC_SIGNATURE}\\\"")

# Add flags
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${DOLFIN_LINK_FLAGS}")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${DOLFIN_LINK_FLAGS}")

#------------------------------------------------------------------------------
# Install

install(TARGETS dolfin EXPORT DOLFINTargets
  RUNTIME DESTINATION ${DOLFIN_LIB_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${DOLFIN_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${DOLFIN_LIB_DIR} COMPONENT Development
  )

install(EXPORT DOLFINTargets DESTINATION ${DOLFIN_SHARE_DIR}/cmake)

#------------------------------------------------------------------------------
# Generate CMake config files (DOLFINConfig{,Version}.cmake)

configure_file(${DOLFIN_CMAKE_DIR}/templates/DOLFINConfig.cmake.in
  ${CMAKE_BINARY_DIR}/dolfin/DOLFINConfig.cmake @ONLY)
configure_file(${DOLFIN_CMAKE_DIR}/templates/DOLFINConfigVersion.cmake.in
  ${CMAKE_BINARY_DIR}/dolfin/DOLFINConfigVersion.cmake @ONLY)
configure_file(${DOLFIN_CMAKE_DIR}/templates/UseDOLFIN.cmake.in
  ${CMAKE_BINARY_DIR}/dolfin/UseDOLFIN.cmake @ONLY)

install(
  FILES
  ${CMAKE_SOURCE_DIR}/cmake/modules/FindPETSc.cmake
  ${CMAKE_SOURCE_DIR}/cmake/modules/FindSLEPc.cmake
  ${CMAKE_BINARY_DIR}/dolfin/DOLFINConfig.cmake
  ${CMAKE_BINARY_DIR}/dolfin/DOLFINConfigVersion.cmake
  ${CMAKE_BINARY_DIR}/dolfin/UseDOLFIN.cmake
  DESTINATION ${DOLFIN_SHARE_DIR}/cmake
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
  DESTINATION ${DOLFIN_PKGCONFIG_DIR}
  COMPONENT Development
  )
#------------------------------------------------------------------------------
