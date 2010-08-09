project(DOLFIN)

cmake_minimum_required(VERSION 2.8)

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()

set(DOLFIN_CMAKE_DIR "${DOLFIN_SOURCE_DIR}/cmake" CACHE INTERNAL "")
set(CMAKE_MODULE_PATH
  "${DOLFIN_CMAKE_DIR}"
  ${CMAKE_MODULE_PATH}
  )

# Set DOLFIN version number
set(DOLFIN_VERSION_MAJOR "0")
set(DOLFIN_VERSION_MINOR "9")
set(DOLFIN_VERSION_MICRO "8")
set(DOLFIN_VERSION "${DOLFIN_VERSION_MAJOR}.${DOLFIN_VERSION_MINOR}.${DOLFIN_VERSION_MICRO}")
add_definitions(-DPACKAGE_VERSION="${DOLFIN_VERSION}")

# Configurable options for how we want to build
option(BUILD_SHARED_LIBS "Build DOLFIN with shared libraries." ON)
option(CMAKE_SKIP_RPATH "Do not add runtime paths when using shared libraries." ON)
option(CMAKE_USE_RELATIVE_PATHS "use relative paths in makefiles and projects." OFF)
option(DOLFIN_ENABLE_DEBUG "Build with debug information." ON)
option(DOLFIN_ENABLE_EXTRADEBUG "Build with extra debug information (no -O2)." OFF)
option(DOLFIN_ENABLE_DEBUGUBLAS "Add some extra Ublas debug information." OFF)
option(DOLFIN_ENABLE_OPTIMIZE "Compile with optimization." OFF)
option(DOLFIN_ENABLE_DOCS "Build documentation." OFF)
option(DOLFIN_ENABLE_DEMOS "Build demos." OFF)
option(DOLFIN_ENABLE_TESTS "Build tests." OFF)
option(DOLFIN_ENABLE_BENCHMARKS "Build benchmark programs." OFF)
option(DOLFIN_ENABLE_CODECOVERAGE "Enable code coverage.")

# Enable or disable external packages
option(DOLFIN_ENABLE_MPI "Compile with support for MPI." ON)
option(DOLFIN_ENABLE_PETSC "Compile with support for PETSc linear algebra." ON)
option(DOLFIN_ENABLE_SLEPC "Compile with support for SLEPc." ON)
option(DOLFIN_ENABLE_SCOTCH "Compile with support for SCOTCH graph partitioning." ON)
option(DOLFIN_ENABLE_UMFPACK "Compile with support for UMFPACK." ON)
option(DOLFIN_ENABLE_TRILINOS "Compile with support for Trilinos." ON)
option(DOLFIN_ENABLE_CHOLMOD "Compile with support for CHOLMOD." ON)
option(MTL4_ENABLE_MTL4 "Compile with support for MTL4." ON)
option(DOLFIN_ENABLE_PARMETIS "Compile with support for ParMETIS." ON)
option(DOLFIN_ENABLE_GMP "Compile with support for GMP." OFF)
option(DOLFIN_ENABLE_ZLIB "Compile with support for zlib." ON)
option(DOLFIN_ENABLE_CGAL "Compile with support for CGAL" ON)
option(DOLFIN_ENABLE_LAPACK "Compile with support for LAPACK." ON)
option(DOLFIN_ENABLE_PYTHON "Compile the Python wrappers." ON)

# Run tests to find external packages ...
if(DOLFIN_ENABLE_PYTHON)
  include(FindPython.dolfin)
  find_package(SWIG REQUIRED)
  include(${SWIG_USE_FILE})
endif(DOLFIN_ENABLE_PYTHON)
include(FindUFC.dolfin)
include(FindArmadillo.dolfin)
include(FindLibXml2.dolfin)
if(DOLFIN_ENABLE_MPI)
  find_package(MPI)
endif(DOLFIN_ENABLE_MPI)
#if(DOLFIN_ENABLE_PETSC)
#  include(FindPETSc.dolfin)
#endif(DOLFIN_ENABLE_PETSC)
#if(DOLFIN_ENABLE_SLEPC)
#  include(FindSLEPc.dolfin)
#endif(DOLFIN_ENABLE_SLEPC)
#if(DOLFIN_ENABLE_SCOTCH)
#  include(FindSCOTCH.dolfin)
#endif(DOLFIN_ENABLE_SCOTCH)
#if(DOLFIN_ENABLE_UMFPACK)
#  include(FindUFMPACK.dolfin)
#endif(DOLFIN_ENABLE_UMFPACK)
#if(DOLFIN_ENABLE_TRILINOS)
#  include(FindTrilinos.dolfin)
#endif(DOLFIN_ENABLE_TRILINOS)
#if(DOLFIN_ENABLE_CHOLMOD)
#  include(FindCHOLMOD.dolfin)
#endif(DOLFIN_ENABLE_CHOLMOD)
if(MTL4_ENABLE_MTL4)
  find_package(MTL4.dolfin)
endif(MTL4_ENABLE_MTL4)
#if(DOLFIN_ENABLE_PARMETIS)
#  include(FindParMETIS.dolfin)
#endif(DOLFIN_ENABLE_PARMETIS)
#if(DOLFIN_ENABLE_GMP)
#  include(FindGMP.dolfin)
#endif(DOLFIN_ENABLE_GMP)
if(DOLFIN_ENABLE_ZLIB)
  include(FindZLIB.dolfin)
endif(DOLFIN_ENABLE_ZLIB)
if(DOLFIN_ENABLE_CGAL)
  include(FindCGAL.dolfin)
endif(DOLFIN_ENABLE_CGAL)
#if(DOLFIN_ENABLE_LAPACK)
#  include(FindLAPACK.dolfin)
#endif(DOLFIN_ENABLE_LAPACK)

# Find Boost
set(DOLFIN_BOOST_COMPONENTS filesystem program_options)
include(FindBoost.dolfin)

# Append the library version information to the library target properties.
option(DOLFIN_WITH_LIBRARY_VERSION "Build with library version information." ON)
if(DOLFIN_WITH_LIBRARY_VERSION)
  set(DOLFIN_LIBRARY_PROPERTIES ${DOLFIN_LIBRARY_PROPERTIES}
    VERSION ${DOLFIN_VERSION}
    SOVERSION ${DOLFIN_VERSION_MAJOR}
  )
endif(DOLFIN_WITH_LIBRARY_VERSION)

set(DOLFIN_BIN_DIR "bin")
set(DOLFIN_LIB_DIR "lib")
set(DOLFIN_INCLUDE_DIR "include")
set(DOLFIN_SHARE_DIR "share")
set(DOLFIN_MAN_DIR "share/man")
set(DOLFIN_INSTALL_PYTHON_MODULE_DIR "lib/python2.6/site-packages")
set(DOLFIN_INSTALL_PYTHON_EXT_DIR "")

add_subdirectory(dolfin)
