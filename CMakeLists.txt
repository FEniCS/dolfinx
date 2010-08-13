project(DOLFIN)

# Require CMake 2.8
cmake_minimum_required(VERSION 2.8)

# Set verbose output while testing CMake
set(CMAKE_VERBOSE_MAKEFILE 1)

# Testing compiler flags
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Devel)
set(CMAKE_CXX_FLAGS "-Wall ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "-ansi -pipe ${CMAKE_CXX_FLAGS}")

# Set DOLFIN version number
set(DOLFIN_VERSION_MAJOR "0")
set(DOLFIN_VERSION_MINOR "9")
set(DOLFIN_VERSION_MICRO "8")
set(DOLFIN_VERSION "${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR}.${DOLFIN_VERSION_MICRO}")
add_definitions(-DPACKAGE_VERSION="${DOLFIN_VERSION}")

# Set special link option, see `cmake --help-policy CMP0003`
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()

# FIXME: What do these lines do?
set(DOLFIN_CMAKE_DIR "${DOLFIN_SOURCE_DIR}/cmake" CACHE INTERNAL "")
set(CMAKE_MODULE_PATH "${DOLFIN_CMAKE_DIR}" ${CMAKE_MODULE_PATH})

#------------------------------------------------------------------------------
# Configurable options for how we want to build

option(BUILD_SHARED_LIBS "Build DOLFIN with shared libraries." ON)
#option(CMAKE_SKIP_RPATH "Do not add runtime paths when using shared libraries." ON)
option(CMAKE_USE_RELATIVE_PATHS "Use relative paths in makefiles and projects." OFF)
option(DOLFIN_ENABLE_DEBUG "Build with debug information." ON)
option(DOLFIN_ENABLE_EXTRADEBUG "Build with extra debug information (no -O2)." OFF)
option(DOLFIN_ENABLE_DEBUGUBLAS "Add some extra uBLAS debug information." OFF)
option(DOLFIN_ENABLE_OPTIMIZE "Compile with optimization." OFF)
option(DOLFIN_ENABLE_DOCS "Build documentation." OFF)
option(DOLFIN_ENABLE_DEMOS "Build demos." OFF)
option(DOLFIN_ENABLE_TESTS "Build tests." OFF)
option(DOLFIN_ENABLE_BENCHMARKS "Build benchmark programs." OFF)
option(DOLFIN_ENABLE_CODECOVERAGE "Enable code coverage.")

option(DOLFIN_WITH_LIBRARY_VERSION "Build with library version information." ON)

#------------------------------------------------------------------------------
# Enable or disable external packages

option(DOLFIN_ENABLE_MPI "Compile with support for MPI." ON)
option(DOLFIN_ENABLE_PETSC "Compile with support for PETSc linear algebra." ON)
option(DOLFIN_ENABLE_SLEPC "Compile with support for SLEPc." ON)
option(DOLFIN_ENABLE_SCOTCH "Compile with support for SCOTCH graph partitioning." ON)
option(DOLFIN_ENABLE_UMFPACK "Compile with support for UMFPACK." ON)
option(DOLFIN_ENABLE_TRILINOS "Compile with support for Trilinos." ON)
option(DOLFIN_ENABLE_CHOLMOD "Compile with support for CHOLMOD." ON)
option(DOLFIN_ENABLE_MTL4 "Compile with support for MTL4." ON)
option(DOLFIN_ENABLE_PARMETIS "Compile with support for ParMETIS." ON)
option(DOLFIN_ENABLE_GMP "Compile with support for GMP." ON)
option(DOLFIN_ENABLE_ZLIB "Compile with support for zlib." ON)
option(DOLFIN_ENABLE_CGAL "Compile with support for CGAL" ON)
option(DOLFIN_ENABLE_PYTHON "Compile the Python wrappers." ON)

#------------------------------------------------------------------------------
# Run tests to find required packages

find_package(UFC.dolfin)
find_package(Armadillo.dolfin)
find_package(LibXml2 REQUIRED)

# Find Boost
set(Boost_ADDITIONAL_VERSIONS 1.43 1.43.0)
set(Boost_USE_STATIC_LIBS OFF)
set(DOLFIN_BOOST_USE_MULTITHREADED OFF CACHE BOOL "Use multithreaded Boost libraries.")
set(BOOST_ROOT $ENV{BOOST_DIR})
find_package(Boost 1.36 COMPONENTS filesystem program_options system REQUIRED)

#------------------------------------------------------------------------------
# Run tests to find optional external packages

if (DOLFIN_ENABLE_PYTHON)
  find_package(PythonInterp)
  find_package(PythonLibs)

  # If Python is found, check for SWIG
  if (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)
    find_package(SWIG REQUIRED)
    include(UseSWIG)
  endif (PYTHONINTERP_FOUND AND PYTHONLIBS_FOUND)

endif(DOLFIN_ENABLE_PYTHON)

if (DOLFIN_ENABLE_MPI)
  find_package(MPI)
  if (DOLFIN_ENABLE_PARMETIS)
    find_package(ParMETIS.dolfin)
  endif (DOLFIN_ENABLE_PARMETIS)

endif (DOLFIN_ENABLE_MPI)

if (DOLFIN_ENABLE_SCOTCH)
  find_package(ScotchPT)
endif (DOLFIN_ENABLE_SCOTCH)

if (DOLFIN_ENABLE_UMFPACK)
  find_package(UMFPACK.dolfin)
endif (DOLFIN_ENABLE_UMFPACK)

if (DOLFIN_ENABLE_CHOLMOD)
  find_package(Cholmod)
endif (DOLFIN_ENABLE_CHOLMOD)

if (DOLFIN_ENABLE_TRILINOS)
  find_package(Trilinos PATHS $ENV{TRILINOS_DIR}/include $ENV{Trilinos_DIR}/include /usr/local/include /usr/include)
endif (DOLFIN_ENABLE_TRILINOS)

if (DOLFIN_ENABLE_MTL4)
  find_package(MTL4.dolfin)
endif (DOLFIN_ENABLE_MTL4)

if (DOLFIN_ENABLE_PETSC)
  find_package(PETSc.dolfin)

  if(DOLFIN_ENABLE_SLEPC)
    find_package(SLEPc)
  endif(DOLFIN_ENABLE_SLEPC)

endif (DOLFIN_ENABLE_PETSC)

if (DOLFIN_ENABLE_GMP)
  find_package(GMP)
endif (DOLFIN_ENABLE_GMP)

if (DOLFIN_ENABLE_ZLIB)
  find_package(ZLIB)
endif (DOLFIN_ENABLE_ZLIB)

if (DOLFIN_ENABLE_CGAL)
  find_package(CGAL PATHS
                $ENV{CGAL_DIR}/lib
                /usr/local/lib
                /usr/local/lib/CGAL
                /usr/lib
                /usr/lib/CGAL
                )
endif (DOLFIN_ENABLE_CGAL)

#------------------------------------------------------------------------------
# Get installation path for Python modules (if Python is enabled and installed)

# Get Python module path from distutils
if (PYTHONINTERP_FOUND)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import sys, distutils.sysconfig; sys.stdout.write(distutils.sysconfig.get_python_lib(plat_specific=True, prefix='${CMAKE_INSTALL_PREFIX}'))"
    OUTPUT_VARIABLE PYTHON_INSTALL_PATH
    )

  # Set Python installation path
  set(DOLFIN_INSTALL_PYTHON_MODULE_DIR ${PYTHON_INSTALL_PATH})
  set(DOLFIN_INSTALL_PYTHON_EXT_DIR ${PYTHON_INSTALL_PATH})
endif (PYTHONINTERP_FOUND)

#------------------------------------------------------------------------------

# Append the library version information to the library target properties
if(DOLFIN_WITH_LIBRARY_VERSION)
  set(DOLFIN_LIBRARY_PROPERTIES ${DOLFIN_LIBRARY_PROPERTIES}
    VERSION ${DOLFIN_VERSION}
    SOVERSION ${DOLFIN_VERSION_MAJOR}
  )
endif(DOLFIN_WITH_LIBRARY_VERSION)

# Set DOLFIN install sub-directories
set(DOLFIN_BIN_DIR "bin")
set(DOLFIN_LIB_DIR "lib")
set(DOLFIN_INCLUDE_DIR "include")
set(DOLFIN_PKGCONFIG_DIR "lib/pkgconfig")
set(DOLFIN_SHARE_DIR "share")
set(DOLFIN_MAN_DIR "share/man")

add_subdirectory(dolfin)

#------------------------------------------------------------------------------
# Generate helper file dolfin.conf

# FIXME: not cross-platform compatible
configure_file(${DOLFIN_CMAKE_DIR}/templates/dolfin.conf.in ${CMAKE_CURRENT_BINARY_DIR}/dolfin.conf @ONLY)

#------------------------------------------------------------------------------

