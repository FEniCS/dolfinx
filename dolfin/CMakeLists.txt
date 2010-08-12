set(DOLFIN_H dolfin.h)
install(FILES ${DOLFIN_H} DESTINATION ${DOLFIN_INCLUDE_DIR} COMPONENT Development)

set(DOLFIN_DIRS
  ale
  common
  elements
  fem
  function
  graph
  io
  la
  log
  main
  math
  mesh
  mf
  nls
  ode
  parameter
  adaptivity
  pde
  plot
  quadrature
  )

#set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

foreach(DIR ${DOLFIN_DIRS})
  file(GLOB _HEADERS ${DIR}/*.h)
  list(APPEND HEADERS ${_HEADERS})
  install(FILES ${_HEADERS} DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR} COMPONENT Development)

  file(GLOB _SOURCES ${DIR}/*.cpp)
  list(APPEND SOURCES ${_SOURCES})
endforeach()

#------------------------------------------------------------------------------

# Add include directories of required packages
list(APPEND DOLFIN_INCLUDE_DIRECTORIES
  ${UFC_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${ARMADILLO_INCLUDE_DIR}
  ${DOLFIN_LIBXML2_INCLUDE_DIR}
  )

set(DOLFIN_TARGET_LINK_LIBRARIES
  ${ARMADILLO_LIBRARY}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
  ${DOLFIN_LIBXML2_LIBRARIES}
  )

#------------------------------------------------------------------------------

if(DOLFIN_ENABLE_MPI AND MPI_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_MPI")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${MPI_INCLUDE_PATH})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${MPI_LIBRARIES})
  # FIXME: Need to consider these MPI variables as well:
  # MPI_COMPILE_FLAGS          Compilation flags for MPI programs
  # MPI_INCLUDE_PATH           Include path(s) for MPI header
  # MPI_LINK_FLAGS             Linking flags for MPI programs
endif(DOLFIN_ENABLE_MPI AND MPI_FOUND)

if(DOLFIN_ENABLE_CGAL AND DOLFIN_CGAL_FOUND)
  # FIXME: This shouldn't be here. Also need to test for GCC
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math")

  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_CGAL")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${DOLFIN_CGAL_INCLUDE_DIR})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES
    ${DOLFIN_CGAL_LIBRARY}
    ${DOLFIN_MPFR_LIBRARY}
    ${DOLFIN_GMP_LIBRARY}
    )
endif(DOLFIN_ENABLE_CGAL AND DOLFIN_CGAL_FOUND)

if(DOLFIN_ENABLE_ZLIB AND DOLFIN_ZLIB_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_ZLIB")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${DOLFIN_ZLIB_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${DOLFIN_ZLIB_LIBS})
endif(DOLFIN_ENABLE_ZLIB AND DOLFIN_ZLIB_FOUND)

if(DOLFIN_ENABLE_MTL4 AND MTL4_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_MTL4")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${MTL4_INCLUDE_DIR})
endif(DOLFIN_ENABLE_MTL4 AND MTL4_FOUND)

if(DOLFIN_ENABLE_TRILINOS AND Trilinos_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_TRILINOS")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${Trilinos_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES_DIRS ${Trilinos_LIBRARY_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${Trilinos_LIBRARIES})
endif(DOLFIN_ENABLE_TRILINOS AND Trilinos_FOUND)

if(DOLFIN_ENABLE_PETSC AND PETSC_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_PETSC")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${PETSC_INCLUDES})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${PETSC_LIBRARIES})
endif(DOLFIN_ENABLE_PETSC AND PETSC_FOUND)

if(DOLFIN_ENABLE_UMFPACK AND UMFPACK_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_UMFPACK")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${UMFPACK_INCLUDE_DIR})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${UMFPACK_LIBRARY})
endif(DOLFIN_ENABLE_UMFPACK AND UMFPACK_FOUND)

#------------------------------------------------------------------------------

# Add link directories
link_directories(${DOLFIN_TARGET_LINK_LIBRARIES_DIRS})

# Add compiler include directories
include_directories(${DOLFIN_SOURCE_DIR} ${DOLFIN_INCLUDE_DIRECTORIES})

# Add CXX defintions
add_definitions(${DOLFIN_CXX_DEFINITIONS})

# Define libraries
add_library(dolfin SHARED ${DOLFIN_H} ${HEADERS} ${SOURCES})
set_target_properties(dolfin PROPERTIES ${DOLFIN_LIBRARY_PROPERTIES})

# Add DOFLIN target libraries
target_link_libraries(dolfin ${DOLFIN_TARGET_LINK_LIBRARIES})

add_subdirectory(swig)

install(TARGETS dolfin
  RUNTIME DESTINATION ${DOLFIN_BIN_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${DOLFIN_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${DOLFIN_LIB_DIR} COMPONENT Development
  )

#------------------------------------------------------------------------------
# Write pkg-config file and install

# Define packages that should be required by pkg-config file
set(PKG_REQUIRES "ufc-1 libxml-2.0")

get_directory_property(cmake_inc INCLUDE_DIRECTORIES)
message(STATUS "Print inc dirs: ${cmake_inc}")
foreach(_inc_dir ${cmake_inc})
  set(PKG_CXXFLAGS "-I${_inc_dir} ${PKG_CXXFLAGS}")
endforeach()
string(REGEX REPLACE ";" " " PKG_DEFINITIONS "${PKG_CXXFLAGS}")

# Add CXX COMPILERflags
set(PKG_CXXFLAGS "${CMAKE_CXX_FLAGS} ${PKG_CXXFLAGS}")
string(REGEX REPLACE ";" " " PKG_CXXFLAGS "${PKG_CXXFLAGS}")

# Convert compiler defintions into a list for pkg-config
# Append definitions to PKG_CXXFLAGS
string(REGEX REPLACE ";" " " PKG_DEFINITIONS "${DOLFIN_CXX_DEFINITIONS}")
set(PKG_CXXFLAGS "${PKG_DEFINITIONS} ${PKG_CXXFLAGS}")

# Add definitions to PKG_CXXFLAGS
set(PKG_CXXFLAGS "${PKG_DEFINITIONS} ${PKG_CXXFLAGS}")

# Convert libraries to -L<libdir> -l<lib> form for use in pkg-config file
# FIXME: this is not cross-platform ready
#foreach(_lib ${DOLFIN_TARGET_LINK_LIBRARIES})
#  string(REGEX REPLACE "(/[^ ]*)/lib([^ ]*)\\.so" "-L\\1 -l\\2"
#    _linkflags
#    "${_lib}"
#    )
#  set(PKG_LINKFLAGS "${_linkflags} ${PKG_LINKFLAGS}")
#endforeach()

# Configure and install pkg-config file
configure_file(dolfin.pc.cmake.in dolfin.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dolfin.pc
  DESTINATION ${DOLFIN_PKGCONFIG_DIR}
  COMPONENT Development
  )
