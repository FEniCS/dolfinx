set(DOLFIN_H dolfin.h)
install(FILES ${DOLFIN_H} DESTINATION ${DOLFIN_INCLUDE_DIR} COMPONENT Development)

#set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#------------------------------------------------------------------------------
# DOLFIN sorce directories

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

#------------------------------------------------------------------------------
# Install header files

foreach(DIR ${DOLFIN_DIRS})
  file(GLOB _HEADERS ${DIR}/*.h)
  list(APPEND HEADERS ${_HEADERS})
  install(FILES ${_HEADERS} DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR} COMPONENT Development)

  file(GLOB _SOURCES ${DIR}/*.cpp)
  list(APPEND SOURCES ${_SOURCES})
endforeach()

#------------------------------------------------------------------------------
# Add include directories and libs of required packages

# libXml2
list(APPEND DOLFIN_INCLUDE_DIRECTORIES  ${LIBXML2_INCLUDE_DIR})
list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${LIBXML2_LIBRARIES})
list(APPEND DOLFIN_CXX_DEFINITIONS ${LIBXML2_DEFINITIONS})

# UFC
list(APPEND DOLFIN_INCLUDE_DIRECTORIES  ${UFC_INCLUDE_DIRS})

# Armadillo
list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${ARMADILLO_INCLUDE_DIR})
list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${ARMADILLO_LIBRARY})
list(APPEND DOLFIN_LINK_FLAGS ${ARMADILLO_LINK_FLAGS})

# Boost
list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${Boost_LIBRARIES})
list(APPEND APPEND DOLFIN_TARGET_LINK_LIBRARIES_DIRS ${Boost_LIBRARY_DIRS})

#------------------------------------------------------------------------------

if(DOLFIN_ENABLE_MPI AND MPI_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_MPI")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${MPI_INCLUDE_PATH})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${MPI_LIBRARIES})

  # FIXME: Check if we're using the MPI wrapper (e.g. mpicxx)
  # Note: It's awkward to pick up the MPI compiler wrappers, so just add
  #       all the flags
  # FIXME: The FindMPI.cmake module defines the variable MPI_COMPILER which
  # FIXME: should point to the MPI compiler wrappers. We could set
  # FIXME: CMAKE_CXX_COMPILER to be MPI_COMPILER but we should be careful
  # FIXME: to not owervriting any user-specified CMAKE_CXX_COMPILER.
  list(APPEND DOLFIN_CXX_FLAGS ${MPI_COMPILE_FLAGS})
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${MPI_INCLUDE_PATH})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${MPI_LINK_FLAGS})

  # Necessary flags for MPICH2
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DMPICH_IGNORE_CXX_SEEK")

  # FIXME: Need to consider these MPI variables as well:
  # FIXME: Don't the MPI wrapper mpicxx take care of this?
  # MPI_COMPILE_FLAGS          Compilation flags for MPI programs
  # MPI_INCLUDE_PATH           Include path(s) for MPI header
  # MPI_LINK_FLAGS             Linking flags for MPI programs
endif(DOLFIN_ENABLE_MPI AND MPI_FOUND)

# CGAL
if(DOLFIN_ENABLE_CGAL AND CGAL_FOUND)

  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_CGAL")

  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${CGAL_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES_DIRS ${CGAL_LIBRARIES_DIR})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${CGAL_LIBRARY})

  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${CGAL_3RD_PARTY_INCLUDE_DIRS})
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${CGAL_3RD_PARTY_LIBRARIES_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${CGAL_3RD_PARTY_LIBRARIES})

  # Add -frounding-math if we have a GNU compiler
  if(CGAL_DISABLE_ROUNDING_MATH_CHECK)
    list(APPEND DOLFIN_CXX_DEFINITIONS "-DCGAL_DISABLE_ROUNDING_MATH_CHECK")
  else ()
    if(CMAKE_COMPILER_IS_GNUCXX)
      set(DOLFIN_CXX_FLAGS "${DOLFIN_CXX_FLAGS} -frounding-math")
    endif(CMAKE_COMPILER_IS_GNUCXX)
  endif ()

endif(DOLFIN_ENABLE_CGAL AND CGAL_FOUND)

# ZLIB
if(DOLFIN_ENABLE_ZLIB AND DOLFIN_ZLIB_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_ZLIB")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${DOLFIN_ZLIB_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${DOLFIN_ZLIB_LIBS})
endif(DOLFIN_ENABLE_ZLIB AND DOLFIN_ZLIB_FOUND)

# MTL4
if(DOLFIN_ENABLE_MTL4 AND MTL4_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_MTL4")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${MTL4_INCLUDE_DIR})
endif(DOLFIN_ENABLE_MTL4 AND MTL4_FOUND)

# Trilinos
if(DOLFIN_ENABLE_TRILINOS AND Trilinos_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_TRILINOS")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${Trilinos_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES_DIRS ${Trilinos_LIBRARY_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${Trilinos_LIBRARIES})
endif(DOLFIN_ENABLE_TRILINOS AND Trilinos_FOUND)

# PETSc
if(DOLFIN_ENABLE_PETSC AND PETSC_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_PETSC")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${PETSC_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${PETSC_LIBRARIES})
endif(DOLFIN_ENABLE_PETSC AND PETSC_FOUND)

# SLEPC
if(DOLFIN_ENABLE_PETSC AND DOLFIN_ENABLE_SLEPC AND SLEPC_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_SLEPC")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${SLEPC_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${SLEPC_LIBRARIES})
endif(DOLFIN_ENABLE_PETSC AND DOLFIN_ENABLE_SLEPC AND SLEPC_FOUND)

# UMFPACK
if(DOLFIN_ENABLE_UMFPACK AND UMFPACK_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_UMFPACK")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${UMFPACK_INCLUDE_DIR})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${UMFPACK_LIBRARY})
endif(DOLFIN_ENABLE_UMFPACK AND UMFPACK_FOUND)

# CHOLMOD
if(DOLFIN_ENABLE_CHOLMOD AND CHOLMOD_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_CHOLMOD")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${CHOLMOD_INCLUDES})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${CHOLMOD_LIBRARIES})
endif(DOLFIN_ENABLE_CHOLMOD AND CHOLMOD_FOUND)

# ParMETIS
if(DOLFIN_ENABLE_PARMETIS AND PARMETIS_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_PARMETIS")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${PARMETIS_INCLUDE_DIR})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${PARMETIS_LIBRARIES})
endif(DOLFIN_ENABLE_PARMETIS AND PARMETIS_FOUND)

# SCOTCH
if(DOLFIN_ENABLE_SCOTCH AND SCOTCH_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_SCOTCH")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${SCOTCH_INCLUDE_DIR})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${SCOTCH_LIBRARIES})
endif(DOLFIN_ENABLE_SCOTCH AND SCOTCH_FOUND)

# GMP
if(DOLFIN_ENABLE_GMP AND GMP_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_GMP")
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${GMP_INCLUDE_DIR})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${GMP_LIBRARIES})
endif(DOLFIN_ENABLE_GMP AND GMP_FOUND)

# CppUnit
if(DOLFIN_ENABLE_UNIT_TESTS AND CPPUNIT_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS ${CPPUNIT_DEFINITIONS})
  list(APPEND DOLFIN_INCLUDE_DIRECTORIES ${CPPUNIT_INCLUDE_DIR})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${CPPUNIT_LIBRARIES})
endif(DOLFIN_ENABLE_UNIT_TESTS AND CPPUNIT_FOUND)

message(STATUS "DEDS  = ${DOLFIN_CXX_DEFINITIONS}")
message(STATUS "FLAGS = ${CMAKE_CXX_FLAGS}")

#------------------------------------------------------------------------------
# Set compiler flags, include directories and library dependencies

# Add link directories
link_directories(${DOLFIN_TARGET_LINK_LIBRARIES_DIRS})

# Add compiler include directories
include_directories(${DOLFIN_SOURCE_DIR} ${DOLFIN_INCLUDE_DIRECTORIES})

# Add CXX defintions
add_definitions(${DOLFIN_CXX_DEFINITIONS})

# Add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}")

# Define libraries
add_library(dolfin SHARED ${DOLFIN_H} ${HEADERS} ${SOURCES})
set_target_properties(dolfin PROPERTIES ${DOLFIN_LIBRARY_PROPERTIES})

# FIXME: GNW: What link flags are we adding? Is this platform-independent?
# FIXME: JR: Yes, it is. Currently it is only '-framework vecLib' defined in
#            ARMADILLO_LINK_FLAGS but this variable will be empty on non-apple.
# FIXME: JR: Maybe we should add DOLFIN_LINK_FLAGS to CMAKE_SHARED_LINKER_FLAGS
#            instead of setting it as a property on the dolfin library? We
#            should at least add either CMAKE_SHARED_LINKER_FLAGS or
#            DOLFIN_LINK_FLAGS to dolfin-config.cmake.
# Add link flags (if any)
if(DOLFIN_LINK_FLAGS)
  set_target_properties(dolfin PROPERTIES LINK_FLAGS ${DOLFIN_LINK_FLAGS})
endif(DOLFIN_LINK_FLAGS)

# Add DOFLIN target libraries
target_link_libraries(dolfin ${DOLFIN_TARGET_LINK_LIBRARIES})

#------------------------------------------------------------------------------
# SWIG

if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND SWIG_FOUND)

  # Make sure we can import the UFC module and that it is compiled with present
  # version of SWIG
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "from ufc import *; import sys; sys.stdout.write('.'.join([str(int(s)) for s in ufc.__swigversion__.split('.')]))"
    OUTPUT_VARIABLE UFC_SWIGVERSION
    RESULT_VARIABLE UFC_MODULE_NOT_FOUND
    )

  # Check that UFC can be imported
  if (UFC_MODULE_NOT_FOUND)
    message(WARNING "Unable to import UFC. Install latest UFC or check that PYTHONPATH is set appropriately. Python will be disabled.")
  endif ()

  # Check that UFC was built with the version of SWIG that is being used
  if (NOT UFC_MODULE_NOT_FOUND AND (NOT "${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}"))
    message(WARNING "UFC compiled with different version of SWIG. Please install SWIG version #{UFC_SWIGVERSION} or recompile UFC with present SWIG.")
  endif()

  # If suitble UFC SWIG extension is found, add SWIG subdirectory
  if (NOT UFC_MODULE_NOT_FOUND AND ("${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}"))
    add_subdirectory(swig)
  endif ()

endif (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND SWIG_FOUND)

#------------------------------------------------------------------------------
# Install

install(TARGETS dolfin
  RUNTIME DESTINATION ${DOLFIN_BIN_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${DOLFIN_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${DOLFIN_LIB_DIR} COMPONENT Development
  )

#------------------------------------------------------------------------------
# Write pkg-config file and install it

# Define packages that should be required by pkg-config file
set(PKG_REQUIRES "ufc-1 libxml-2.0")

get_directory_property(cmake_inc INCLUDE_DIRECTORIES)
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

# Configure and install pkg-config file
configure_file(dolfin.pc.cmake.in dolfin.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dolfin.pc
  DESTINATION ${DOLFIN_PKGCONFIG_DIR}
  COMPONENT Development
  )

#------------------------------------------------------------------------------
# Generate CMake config file (dolfin-config.cmake)

configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin-config.cmake.in dolfin-config.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dolfin-config.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share/dolfin/cmake
  COMPONENT Development
  )

#------------------------------------------------------------------------------
