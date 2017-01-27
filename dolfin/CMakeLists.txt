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
set(EXCLUDES ${DOLFIN_H} swig CMakeLists.txt)

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

# Add dependency on the compile_ale_forms target
add_dependencies(dolfin compile_ale_forms)

#------------------------------------------------------------------------------
# Add include directories and libs of required packages

# NOTE: The include path for dependecies should be appended to
#       DOLFIN_DEP_INCLUDE_DIRECTORIES, unless there are issues, such
#       as unused variable that can cause build errors when using strict
#       compiler flags, in which case the path should be appended to
#       DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES

# UFC
list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${UFC_INCLUDE_DIRS})

# Eigen3
list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})

# Boost
foreach (BOOST_PACKAGE ${DOLFIN_BOOST_COMPONENTS_PUBLIC})
  target_link_libraries(dolfin PUBLIC "Boost::boost Boost::${BOOST_PACKAGE}")
endforeach()
foreach (BOOST_PACKAGE ${DOLFIN_BOOST_COMPONENTS_PRIVATE})
  target_link_libraries(dolfin PRIVATE "Boost::${BOOST_PACKAGE}")
endforeach()

#------------------------------------------------------------------------------

# HDF5
if (DOLFIN_ENABLE_HDF5 AND HDF5_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_HDF5")
  list(APPEND DOLFIN_CXX_DEFINITIONS ${HDF5_DEFINITIONS})
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS})
  list(APPEND DOLFIN_LINK_FLAGS ${HDF5_LINK_FLAGS})
  target_link_libraries(dolfin PRIVATE ${HDF5_C_LIBRARIES})
endif()

# Trilinos
if (DOLFIN_ENABLE_TRILINOS AND DOLFIN_TRILINOS_PACKAGES_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_TRILINOS")
  list(APPEND DOLFIN_CXX_DEFINITIONS ${DOLFIN_TRILINOS_DEFINITIONS})
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${Trilinos_INCLUDE_DIRS})
  target_link_libraries(dolfin PUBLIC ${DOLFIN_TRILINOS_LIBRARIES})
endif()

# SLEPC
if (DOLFIN_ENABLE_PETSC AND DOLFIN_ENABLE_SLEPC AND SLEPC_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_SLEPC")
  target_link_libraries(dolfin PUBLIC SLEPC::slepc)
  target_link_libraries(dolfin PRIVATE SLEPC::slepc_static)
endif()

# PETSc
if (DOLFIN_ENABLE_PETSC AND PETSC_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_PETSC")
  target_link_libraries(dolfin PUBLIC PETSC::petsc)
  target_link_libraries(dolfin PRIVATE PETSC::petsc_static)
endif()

# petsc4py
if (DOLFIN_ENABLE_PETSC AND DOLFIN_ENABLE_PETSC4PY AND PETSC4PY_FOUND)
  list(APPEND DOLFIN_PYTHON_DEFINITIONS "-DHAS_PETSC4PY")
  list(APPEND PYTHON_INCLUDE_DIRS ${PETSC4PY_INCLUDE_DIRS})
endif()

# slepc4py
if (DOLFIN_ENABLE_SLEPC AND DOLFIN_ENABLE_SLEPC4PY AND SLEPC4PY_FOUND)
  list(APPEND DOLFIN_PYTHON_DEFINITIONS "-DHAS_SLEPC4PY")
  list(APPEND PYTHON_INCLUDE_DIRS ${SLEPC4PY_INCLUDE_DIRS})
endif()

# UMFPACK
if (DOLFIN_ENABLE_UMFPACK AND UMFPACK_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_UMFPACK")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${UMFPACK_INCLUDE_DIRS})
  target_link_libraries(dolfin PRIVATE ${UMFPACK_LIBRARIES})
endif()

# CHOLMOD
if (DOLFIN_ENABLE_CHOLMOD AND CHOLMOD_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_CHOLMOD")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${CHOLMOD_INCLUDE_DIRS})
  target_link_libraries(dolfin PRIVATE ${CHOLMOD_LIBRARIES})
endif()

# SCOTCH
if (DOLFIN_ENABLE_SCOTCH AND SCOTCH_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_SCOTCH")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${SCOTCH_INCLUDE_DIRS})
  target_link_libraries(dolfin PRIVATE ${SCOTCH_LIBRARIES})
endif()

# ParMETIS
if (DOLFIN_ENABLE_PARMETIS AND PARMETIS_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_PARMETIS")
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${PARMETIS_INCLUDE_DIRS})
  target_link_libraries(dolfin PRIVATE ${PARMETIS_LIBRARIES})
endif()

# ZLIB
if (DOLFIN_ENABLE_ZLIB AND ZLIB_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_ZLIB")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${ZLIB_INCLUDE_DIRS})
  target_link_libraries(dolfin PRIVATE ${ZLIB_LIBRARIES})
endif()

# MPI
if (DOLFIN_ENABLE_MPI AND MPI_CXX_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_MPI")
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_PATH})
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_PATH})
  set(DOLFIN_CXX_FLAGS "${DOLFIN_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
  target_link_libraries(dolfin PUBLIC ${MPI_CXX_LIBRARIES})
endif()

# OpenMP
if (DOLFIN_ENABLE_OPENMP AND OPENMP_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_OPENMP")
  set(DOLFIN_CXX_FLAGS "${DOLFIN_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  if (MINGW)
    list(APPEND DOLFIN_LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()
endif()


# VTK
if (DOLFIN_ENABLE_VTK AND VTK_FOUND)
  include(${VTK_USE_FILE})

  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_VTK")
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${VTK_INCLUDE_DIRS})
  target_link_libraries(dolfin PRIVATE ${VTK_LIBRARIES})
endif()

#------------------------------------------------------------------------------
# Set compiler flags, include directories and library dependencies

# Add compiler include directories
include_directories(BEFORE ${DOLFIN_SOURCE_DIR})
include_directories(${DOLFIN_DEP_INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES})

# Add CXX definitions
add_definitions(${DOLFIN_CXX_DEFINITIONS})
add_definitions(-DDOLFIN_VERSION="${DOLFIN_VERSION}")

# Just add git revision flag to the one affected file
set_source_files_properties(common/defines.cpp PROPERTIES COMPILE_FLAGS
  "-DDOLFIN_GIT_COMMIT_HASH=\\\"${GIT_COMMIT_HASH}\\\" -DUFC_SIGNATURE=\\\"${UFC_SIGNATURE}\\\"")

# Add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${DOLFIN_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${DOLFIN_LINK_FLAGS}")

#------------------------------------------------------------------------------
# SWIG

if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND AND SWIG_FOUND)

  # Add numpy defines to DOLFIN_PYTHON_DEFINITIONS
  if (NUMPY_FOUND)

    # Set numpy version define
    set(DOLFIN_PYTHON_DEFINITIONS ${DOLFIN_PYTHON_DEFINITIONS};-DNUMPY_VERSION_MAJOR=${NUMPY_VERSION_MAJOR};-DNUMPY_VERSION_MINOR=${NUMPY_VERSION_MINOR};-DNUMPY_VERSION_MICRO=${NUMPY_VERSION_MICRO})

    # Only set define for none deprecated API for NUMPY version 1.7
    # and larger
    if(NUMPY_VERSION VERSION_GREATER 1.6.2)
      set(DOLFIN_PYTHON_DEFINITIONS ${DOLFIN_PYTHON_DEFINITIONS} -DNPY_NO_DEPRECATED_API=NPY_${NUMPY_VERSION_MAJOR}_${NUMPY_VERSION_MINOR}_API_VERSION)
    endif()

  endif()

  add_subdirectory(swig)

endif()

#------------------------------------------------------------------------------
# Install

install(TARGETS dolfin EXPORT DOLFINTargets
  RUNTIME DESTINATION ${DOLFIN_LIB_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${DOLFIN_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${DOLFIN_LIB_DIR} COMPONENT Development
  )

install(EXPORT DOLFINTargets DESTINATION ${DOLFIN_SHARE_DIR}/cmake)

#------------------------------------------------------------------------------
# Generate pkg-config file and install it

# Define packages that should be required by pkg-config file
set(PKG_REQUIRES "")

# Convert include dirs to -I<incdir> form
foreach(_inc_dir ${DOLFIN_DEP_INCLUDE_DIRECTORIES})
  set(PKG_INCLUDES "-I${_inc_dir} ${PKG_INCLUDES}")
endforeach()

foreach(_inc_dir ${DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES})
  set(PKG_INCLUDES "-I${_inc_dir} ${PKG_INCLUDES}")
endforeach()

# Convert compiler flags and definitions into space separated strings
string(REPLACE ";" " " PKG_CXXFLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE ";" " " PKG_LINKFLAGS "${CMAKE_EXE_LINKER_FLAGS}")
string(REPLACE ";" " " PKG_DEFINITIONS "${DOLFIN_CXX_DEFINITIONS}")

# Get link libraries
get_target_property(DOLFIN_TARGET_LINK_LIBRARIES dolfin INTERFACE_LINK_LIBRARIES)

# Convert libraries to -L<libdir> -l<lib> form
foreach(_lib ${DOLFIN_TARGET_LINK_LIBRARIES})
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

# Configure and install pkg-config file
configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin.pc.in ${CMAKE_BINARY_DIR}/dolfin.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/dolfin.pc
  DESTINATION ${DOLFIN_PKGCONFIG_DIR}
  COMPONENT Development
  )

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
  COMPONENT Development
  )

#------------------------------------------------------------------------------
