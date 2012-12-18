set(DOLFIN_H dolfin.h)
install(FILES ${DOLFIN_H} DESTINATION ${DOLFIN_INCLUDE_DIR} COMPONENT Development)

#set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Macros

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

foreach(DIR ${DOLFIN_DIRS})
  file(GLOB _HEADERS ${DIR}/*.h)
  list(APPEND HEADERS ${_HEADERS})
  install(FILES ${_HEADERS} DESTINATION ${DOLFIN_INCLUDE_DIR}/dolfin/${DIR} COMPONENT Development)

  file(GLOB _SOURCES ${DIR}/*.cpp)
  list(APPEND SOURCES ${_SOURCES})
endforeach()

#------------------------------------------------------------------------------
# Add include directories and libs of required packages

# NOTE: The include path for dependecies should be appended to
#       DOLFIN_DEP_INCLUDE_DIRECTORIES, unless there are issues, such
#       as unused variable that can cause build errors when using strict
#       compiler flags, in which case the path should be appended to
#       DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES

# libXml2
list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${LIBXML2_INCLUDE_DIR})
list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${LIBXML2_LIBRARIES})
list(APPEND DOLFIN_CXX_DEFINITIONS ${LIBXML2_DEFINITIONS})

# UFC
list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${UFC_INCLUDE_DIRS})

# Armadillo
list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${ARMADILLO_INCLUDE_DIRS})
list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${ARMADILLO_LIBRARIES})
list(APPEND DOLFIN_LINK_FLAGS ${ARMADILLO_LINK_FLAGS})

# Boost
list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${Boost_LIBRARIES})
list(APPEND APPEND DOLFIN_TARGET_LINK_LIBRARIES_DIRS ${Boost_LIBRARY_DIRS})


#------------------------------------------------------------------------------

# HDF5
if (DOLFIN_ENABLE_HDF5 AND HDF5_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_HDF5")
  list(APPEND DOLFIN_CXX_DEFINITIONS ${HDF5_DEFINITIONS})
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${HDF5_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${HDF5_LIBRARIES})
  list(APPEND DOLFIN_LINK_FLAGS ${HDF5_LINK_FLAGS})
endif()

# Trilinos
if (DOLFIN_ENABLE_TRILINOS AND DOLFIN_TRILINOS_PACKAGES_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_TRILINOS")
  list(APPEND DOLFIN_CXX_DEFINITIONS ${DOLFIN_TRILINOS_DEFINITIONS})
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${Trilinos_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${DOLFIN_TRILINOS_LIBRARIES})
endif()

# SLEPC
if (DOLFIN_ENABLE_PETSC AND DOLFIN_ENABLE_SLEPC AND SLEPC_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_SLEPC")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${SLEPC_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${SLEPC_LIBRARIES})
endif()

# TAO
if (DOLFIN_ENABLE_PETSC AND DOLFIN_ENABLE_TAO AND TAO_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_TAO")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${TAO_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${TAO_LIBRARIES})
endif()

# PETSc
if (DOLFIN_ENABLE_PETSC AND PETSC_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_PETSC")
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${PETSC_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${PETSC_LIBRARIES})
  if (PETSC_CUSP_FOUND)
    list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_PETSC_CUSP")
  endif()
endif()

# UMFPACK
if (DOLFIN_ENABLE_UMFPACK AND UMFPACK_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_UMFPACK")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${UMFPACK_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${UMFPACK_LIBRARIES})
endif()

# CHOLMOD
if (DOLFIN_ENABLE_CHOLMOD AND CHOLMOD_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_CHOLMOD")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${CHOLMOD_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${CHOLMOD_LIBRARIES})
endif()

# PaStiX
if (DOLFIN_ENABLE_PASTIX AND PASTIX_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_PASTIX")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${PASTIX_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${PASTIX_LIBRARIES})
endif()

# SCOTCH
if (DOLFIN_ENABLE_SCOTCH AND SCOTCH_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_SCOTCH")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${SCOTCH_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${SCOTCH_LIBRARIES})
endif()

# ParMETIS
if (DOLFIN_ENABLE_PARMETIS AND PARMETIS_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_PARMETIS")
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${PARMETIS_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${PARMETIS_LIBRARIES})
endif()

# CGAL
if (DOLFIN_ENABLE_CGAL AND CGAL_FOUND)

  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_CGAL")
  # Add CGAL include path to DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES to
  # treat as system headers with g++ to not warn generate errors
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${CGAL_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${CGAL_LIBRARIES})

  # Add necessary CGAL flags (usually related to floating point)
  set(DOLFIN_CXX_FLAGS ${DOLFIN_CXX_FLAGS} ${CGAL_CXX_FLAGS_INIT})

  # Add -frounding-math if we have a GNU compiler
  if (CGAL_DISABLE_ROUNDING_MATH_CHECK)
    list(APPEND DOLFIN_CXX_DEFINITIONS "-DCGAL_DISABLE_ROUNDING_MATH_CHECK")
  else()
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
      set(DOLFIN_CXX_FLAGS "${DOLFIN_CXX_FLAGS} -frounding-math")
    endif()
  endif()

endif()

# ZLIB
if (DOLFIN_ENABLE_ZLIB AND ZLIB_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_ZLIB")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${ZLIB_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${ZLIB_LIBRARIES})
endif()

# CppUnit
if (DOLFIN_ENABLE_UNIT_TESTS AND CPPUNIT_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_CPPUNIT")
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${CPPUNIT_INCLUDE_DIRS})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${CPPUNIT_LIBRARIES})
endif()

# MPI
if (DOLFIN_ENABLE_MPI AND MPI_CXX_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_MPI")
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_PATH})
  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${MPI_CXX_LIBRARIES})
  list(APPEND DOLFIN_DEP_INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_PATH})
  set(DOLFIN_CXX_FLAGS "${DOLFIN_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")

  # FIXME: Is this linker flag required?
  #list(APPEND DOLFIN_LINK_FLAGS ${MPI_CXX_LINK_FLAGS})
endif()

# OpenMP
if (DOLFIN_ENABLE_OPENMP AND OPENMP_FOUND)
  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_OPENMP")
  set(DOLFIN_CXX_FLAGS "${DOLFIN_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  if (MINGW)
    list(APPEND DOLFIN_LINK_FLAGS "${OpenMP_CXX_FLAGS}")
  endif()

endif()

# Qt4
if (DOLFIN_ENABLE_QT AND QT_FOUND)
  include(${QT_USE_FILE})

  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_QT4")
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${QT_INCLUDE_DIR})

  # Loop over libs and get full path
  foreach (lib ${QT_LIBRARIES})
    find_library(QT_LIB_${lib} ${lib} HINTS ${QT_LIBRARY_DIRS})
    if (QT_LIB_${lib})
      list(APPEND DOLFIN_QT_LIBRARIES ${QT_LIB_${lib}})
    endif()
  endforeach()

  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${DOLFIN_QT_LIBRARIES})
endif()

# VTK
if (DOLFIN_ENABLE_VTK AND VTK_FOUND)
  include(${VTK_USE_FILE})

  if (APPLE)
    set(DOLFIN_CXX_FLAGS "${DOLFIN_CXX_FLAGS} -Wno-deprecated")
  endif()

  list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_VTK")
  list(APPEND DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES ${VTK_INCLUDE_DIRS})

  if (NOT VTK_LIBRARIES)
    # FIXME: Check which of these we actually need.
    set(VTK_LIBRARIES
      vtkCommon
      vtkFiltering
      vtkImaging
      vtkGraphics
      vtkGenericFiltering
      vtkIO
      vtkRendering
      vtkVolumeRendering
      vtkHybrid
      vtkWidgets
      vtkParallel
      vtkInfovis
      vtkGeovis
      vtkViews
      vtkCharts
      )
  endif()

  # Loop over libs and get full path
  foreach (lib ${VTK_LIBRARIES})
    find_library(VTK_LIB_${lib} ${lib} HINTS ${VTK_LIBRARY_DIRS})
    if (VTK_LIB_${lib})
      list(APPEND DOLFIN_VTK_LIBRARIES ${VTK_LIB_${lib}})
    endif()
  endforeach()

  # Check if the QVTK library is found
  if (DOLFIN_ENABLE_QT AND QT_FOUND)
    find_library(VTK_LIB_QVTK QVTK HINTS ${VTK_LIBRARY_DIRS})
    if (VTK_LIB_QVTK)
      list(APPEND DOLFIN_CXX_DEFINITIONS "-DHAS_QVTK")
      list(APPEND DOLFIN_VTK_LIBRARIES ${VTK_LIB_QVTK})
    else()
      message(WARNING "QVTK not found, disabling QT for plotting")
    endif()
  endif()

  list(APPEND DOLFIN_TARGET_LINK_LIBRARIES ${DOLFIN_VTK_LIBRARIES})
endif()

#------------------------------------------------------------------------------
# Set compiler flags, include directories and library dependencies

# Add compiler include directories
include_directories(${DOLFIN_SOURCE_DIR} ${DOLFIN_DEP_INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${DOLFIN_DEP_SYSTEM_INCLUDE_DIRECTORIES})

# Add CXX defintions
add_definitions(${DOLFIN_CXX_DEFINITIONS})
add_definitions(-DDOLFIN_VERSION="${DOLFIN_VERSION}")

# Add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DOLFIN_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${DOLFIN_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${DOLFIN_LINK_FLAGS}")

# Define libraries
add_library(dolfin ${DOLFIN_H} ${HEADERS} ${SOURCES})
set_target_properties(dolfin PROPERTIES ${DOLFIN_LIBRARY_PROPERTIES})

# Strip leading and trailing whitespaces
string(STRIP DOLFIN_TARGET_LINK_LIBRARIES "${DOLFIN_TARGET_LINK_LIBRARIES}")

# Add DOLFIN target libraries
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
  endif()

  # Check that UFC was built with the version of SWIG that is being used
  if (NOT UFC_MODULE_NOT_FOUND AND (NOT "${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}"))
    message(WARNING "UFC compiled with different version of SWIG. Please install SWIG version ${UFC_SWIGVERSION} or recompile UFC with present SWIG.")
  endif()

  # If suitble UFC SWIG extension is found, add SWIG subdirectory
  if (NOT UFC_MODULE_NOT_FOUND AND ("${SWIG_VERSION}" STREQUAL "${UFC_SWIGVERSION}"))
    add_subdirectory(swig)
  endif()

endif()

#------------------------------------------------------------------------------
# Install

install(TARGETS dolfin
  RUNTIME DESTINATION ${DOLFIN_LIB_DIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${DOLFIN_LIB_DIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${DOLFIN_LIB_DIR} COMPONENT Development
  )

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
# Generate CMake config files (dolfin-config{,-version}.cmake)

# Set library location
get_target_property(DOLFIN_LIBRARY_LOCATION dolfin LOCATION)
get_filename_component(DOLFIN_LIBRARY_FILENAME ${DOLFIN_LIBRARY_LOCATION} NAME)
set(DOLFIN_LIBRARY "${CMAKE_INSTALL_PREFIX}/${DOLFIN_LIB_DIR}/${DOLFIN_LIBRARY_FILENAME}")

configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin-config.cmake.in
  ${CMAKE_BINARY_DIR}/dolfin/dolfin-config.cmake @ONLY)
configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin-config-version.cmake.in
  ${CMAKE_BINARY_DIR}/dolfin/dolfin-config-version.cmake @ONLY)
install(
  FILES
    ${CMAKE_BINARY_DIR}/dolfin/dolfin-config.cmake
    ${CMAKE_BINARY_DIR}/dolfin/dolfin-config-version.cmake
  DESTINATION ${DOLFIN_SHARE_DIR}/cmake
  COMPONENT Development
  )

#------------------------------------------------------------------------------
