# - Try to find PETSc
# Once done this will define
#
#  PETSC_FOUND        - system has PETSc
#  PETSC_INCLUDE_DIRS - include directories for PETSc
#  PETSC_LIBRARIES    - libraries for PETSc
#  PETSC_DIR          - directory where PETSc is built
#  PETSC_ARCH         - architecture for which PETSc is built
#  PETSC_CUSP_FOUND   - PETSc has Cusp support 
#
# This config script is (very loosley) based on a PETSc CMake script by Jed Brown.

#=============================================================================
# Copyright (C) 2010-2012 Garth N. Wells, Anders Logg and Johannes Ring
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

# NOTE: The PETSc Makefile returns a bunch of libraries with '-L' and '-l',
# wheres we would prefer complete paths. For a discussion, see
# http://www.cmake.org/Wiki/CMake:Improving_Find*_Modules#Current_workarounds

message(STATUS "Checking for package 'PETSc'")

# Set debian_arches (PETSC_ARCH for Debian-style installations)
foreach (debian_arches linux kfreebsd)
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(DEBIAN_FLAVORS ${debian_arches}-gnu-c-debug ${debian_arches}-gnu-c-opt ${DEBIAN_FLAVORS})
  else()
    set(DEBIAN_FLAVORS ${debian_arches}-gnu-c-opt ${debian_arches}-gnu-c-debug ${DEBIAN_FLAVORS})
  endif()
endforeach()

# List of possible locations for PETSC_DIR
set(petsc_dir_locations "")
list(APPEND petsc_dir_locations "/usr/lib/petscdir/3.2")    # Debian location
list(APPEND petsc_dir_locations "/usr/lib/petscdir/3.1")    # Debian location
list(APPEND petsc_dir_locations "/usr/lib/petscdir/3.0.0")  # Debian location
list(APPEND petsc_dir_locations "/usr/local/lib/petsc")     # User location
list(APPEND petsc_dir_locations "$ENV{HOME}/petsc")         # User location

# Add other possible locations for PETSC_DIR
set(_SYSTEM_LIB_PATHS "${CMAKE_SYSTEM_LIBRARY_PATH};${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
string(REGEX REPLACE ":" ";" libdirs ${_SYSTEM_LIB_PATHS})
foreach (libdir ${libdirs})
  get_filename_component(petsc_dir_location "${libdir}/" PATH)
  list(APPEND petsc_dir_locations ${petsc_dir_location})
endforeach()

# Try to figure out PETSC_DIR by finding petsc.h
find_path(PETSC_DIR include/petsc.h
  HINTS ${PETSC_DIR} $ENV{PETSC_DIR}
  PATHS ${petsc_dir_locations}
  DOC "PETSc directory")

# Report result of search for PETSC_DIR
if (DEFINED PETSC_DIR)
  message(STATUS "PETSC_DIR is ${PETSC_DIR}")
else()
  message(STATUS "PETSC_DIR is empty")
endif()

# Try to figure out PETSC_ARCH if not set
if (PETSC_DIR AND NOT PETSC_ARCH)
  set(_petsc_arches
    $ENV{PETSC_ARCH}   # If set, use environment variable first
    ${DEBIAN_FLAVORS}  # Debian defaults
    x86_64-unknown-linux-gnu i386-unknown-linux-gnu)
  set(petscconf "NOTFOUND" CACHE FILEPATH "Cleared" FORCE)
  foreach (arch ${_petsc_arches})
    if (NOT PETSC_ARCH)
      find_path(petscconf petscconf.h
      HINTS ${PETSC_DIR}
      PATH_SUFFIXES ${arch}/include bmake/${arch}
      NO_DEFAULT_PATH)
      if (petscconf)
        set(PETSC_ARCH "${arch}" CACHE STRING "PETSc build architecture")
      endif()
    endif()
  endforeach()
  set(petscconf "NOTFOUND" CACHE INTERNAL "Scratch variable" FORCE)
endif()

# Report result of search for PETSC_ARCH
if (DEFINED PETSC_ARCH)
  message(STATUS "PETSC_ARCH is ${PETSC_ARCH}")
else()
  message(STATUS "PETSC_ARCH is empty")
endif()

# Look for petscconf.h
if (EXISTS ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h)
  message(STATUS "Found petscconf.h")
  set(FOUND_PETSC_CONF 1)
else()
  message(STATUS "Unable to find petscconf.h")
endif()

# Get variables from PETSc configuration
if (FOUND_PETSC_CONF)

  # Create a temporary Makefile to probe the PETSc configuration
  set(petsc_config_makefile ${PROJECT_BINARY_DIR}/Makefile.petsc)
  file(WRITE ${petsc_config_makefile}
"# This file was autogenerated by FindPETSc.cmake
PETSC_DIR  = ${PETSC_DIR}
PETSC_ARCH = ${PETSC_ARCH}
include ${PETSC_DIR}/conf/variables
show :
	-@echo -n \${\${VARIABLE}}
")

  # Define macro for getting PETSc variables from Makefile
  macro(PETSC_GET_VARIABLE var name)
    set(${var} "NOTFOUND" CACHE INTERNAL "Cleared" FORCE)
    execute_process(COMMAND ${CMAKE_MAKE_PROGRAM} --no-print-directory -f ${petsc_config_makefile} show VARIABLE=${name}
      OUTPUT_VARIABLE ${var}
      RESULT_VARIABLE petsc_return)
  endmacro()

  # Call macro to get the PETSc variables
  petsc_get_variable(PETSC_INCLUDE PETSC_INCLUDE)          # 3.1
  petsc_get_variable(PETSC_CC_INCLUDES PETSC_CC_INCLUDES)  # dev
  set(PETSC_INCLUDE ${PETSC_INCLUDE} ${PETSC_CC_INCLUDES})
  petsc_get_variable(PETSC_LIB_BASIC PETSC_LIB_BASIC)
  petsc_get_variable(PETSC_LIB_DIR PETSC_LIB_DIR)
  set(PETSC_LIB "-L${PETSC_LIB_DIR} ${PETSC_LIB_BASIC}")
 
  # Remove temporary Makefile
  file(REMOVE ${petsc_config_makefile})

  # Extract include paths and libraries from compile command line
  include(ResolveCompilerPaths)
  resolve_includes(PETSC_INCLUDE_DIRS "${PETSC_INCLUDE}")
  resolve_libraries(PETSC_LIBRARIES "${PETSC_LIB}")

  # Add X11 includes and libraries on Mac
  if (APPLE)
    find_package(X11)
    list(APPEND PETSC_INCLUDE_DIRS ${X11_X11_INCLUDE_PATH})
    list(APPEND PETSC_LIBRARIES ${X11_LIBRARIES})
  endif()

  # Add variables to CMake cache and mark as advanced
  set(PETSC_INCLUDE_DIRS ${PETSC_INCLUDE_DIRS} CACHE STRING "PETSc include paths." FORCE)
  set(PETSC_LIBRARIES ${PETSC_LIBRARIES} CACHE STRING "PETSc libraries." FORCE)
  mark_as_advanced(PETSC_INCLUDE_DIRS PETSC_LIBRARIES)

endif()

# Build PETSc test program
if (FOUND_PETSC_CONF)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${PETSC_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${PETSC_LIBRARIES})

  # Add MPI variables if MPI has been found
  if (MPI_FOUND)
    set(CMAKE_REQUIRED_INCLUDES  ${CMAKE_REQUIRED_INCLUDES} ${MPI_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${MPI_LIBRARIES})
    set(CMAKE_REQUIRED_FLAGS     "${CMAKE_REQUIRED_FLAGS} ${MPI_COMPILE_FLAGS}")
  endif()

  # Run PETSc test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
#include \"petscts.h\"
#include \"petsc.h\"
int main()
{
  PetscErrorCode ierr;
  TS ts;
  int argc = 0;
  char** argv = NULL;
  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
#if PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR <= 1
  ierr = TSDestroy(ts);CHKERRQ(ierr);
#else
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
#endif
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
" PETSC_TEST_RUNS)

  if (PETSC_TEST_RUNS)
    message(STATUS "PETSc test runs")
  else()
    message(STATUS "PETSc test failed")
  endif()

  # Run test program to check for PETSc Cusp
  check_cxx_source_runs("
#include \"petsc.h\"
int main()
{
#if PETSC_HAVE_CUSP
  return 0;
#else
  return 1;
#endif
}
" PETSC_CUSP_FOUND)

  if (PETSC_CUSP_FOUND)
    message(STATUS "PETSc configured with Cusp support")
  else()
      message(STATUS "PETSc configured without Cusp support")
  endif()

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PETSc
  "PETSc could not be found. Be sure to set PETSC_DIR and PETSC_ARCH."
  PETSC_LIBRARIES PETSC_DIR PETSC_INCLUDE_DIRS PETSC_TEST_RUNS)
