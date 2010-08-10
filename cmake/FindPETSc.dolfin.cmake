# - Try to find PETSc
# Once done this will define
#
#  PETSC_FOUND        - system has PETSc
#  PETSC_INCLUDES     - the PETSc include directories
#  PETSC_LIBRARIES    - Link these to use PETSc
#  PETSC_COMPILER     - Compiler used by PETSc, helpful to find a compatible MPI
#  PETSC_DEFINITIONS  - Compiler switches for using PETSc
#  PETSC_MPIEXEC      - Executable for running MPI programs
#  PETSC_VERSION      - Version string (MAJOR.MINOR.SUBMINOR)
#
#  Hack: PETSC_VERSION currently decides on the version based on the
#  layout.  Otherwise we need to run C code to determine the version.
#
# Setting these changes the behavior of the search
#  PETSC_DIR - directory in which PETSc resides
#  PETSC_ARCH - build architecture
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

set(PETSC_FOUND 0)

message(STATUS "checking for package 'PETSc'")

find_path (PETSC_DIR include/petsc.h
  HINTS ENV PETSC_DIR
  PATHS
  /usr/lib/petscdir/3.0.0 /usr/lib/petscdir/2.3.3 /usr/lib/petscdir/2.3.2 # Debian
  $ENV{HOME}/petsc
  DOC "PETSc Directory")

IF ( "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" )
  set( DEBIAN_FLAVORS "linux-gnu-c-debug linux-gnu-c-opt")
ELSE()
  set( DEBIAN_FLAVORS "linux-gnu-c-opt linux-gnu-c-debug")
ENDIF()

if (PETSC_DIR AND NOT PETSC_ARCH)
  set (_petsc_arches
    $ENV{PETSC_ARCH}                   # If set, use environment variable first
    ${DEBIAN_FLAVORS}                  # Debian defaults
    x86_64-unknown-linux-gnu i386-unknown-linux-gnu)
  set (petscconf "NOTFOUND" CACHE FILEPATH "Cleared" FORCE)
  foreach (arch ${_petsc_arches})
    if (NOT PETSC_ARCH)
      find_path (petscconf petscconf.h
      HINTS ${PETSC_DIR}
      PATH_SUFFIXES ${arch}/include bmake/${arch}
      NO_DEFAULT_PATH)
      if (petscconf)
        set (PETSC_ARCH "${arch}" CACHE STRING "PETSc build architecture")
      endif (petscconf)
    endif (NOT PETSC_ARCH)
  endforeach (arch)
  set (petscconf "NOTFOUND" CACHE INTERNAL "Scratch variable" FORCE)
endif (PETSC_DIR AND NOT PETSC_ARCH)

#set (petsc_slaves LIBRARIES_SYS LIBRARIES_VEC LIBRARIES_MAT LIBRARIES_DM LIBRARIES_KSP LIBRARIES_SNES LIBRARIES_TS
#  INCLUDE_DIR INCLUDE_CONF)
#include (FindPackageMultipass)
#find_package_multipass (PETSc petsc_config_current
#  STATES DIR ARCH
#  DEPENDENTS INCLUDES LIBRARIES COMPILER MPIEXEC ${petsc_slaves})

# Determine whether the PETSc layout is old-style (through 2.3.3) or
# new-style (3.0.0)
if (EXISTS ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h)   # > 2.3.3
  #set (petsc_conf_base ${PETSC_DIR}/conf/base)
  set (petsc_conf_base ${PETSC_DIR}/conf)
  set (PETSC_VERSION "3.0.0")
elseif (EXISTS ${PETSC_DIR}/bmake/${PETSC_ARCH}/petscconf.h) # <= 2.3.3
  set (petsc_conf_base ${PETSC_DIR}/bmake/common/base)
  set (PETSC_VERSION "2.3.3")
else (EXISTS ${PETSC_DIR}/bmake/${PETSC_ARCH}/petscconf.h)
  set (petsc_conf_base "NOTFOUND")
endif (EXISTS ${PETSC_DIR}/${PETSC_ARCH}/include/petscconf.h)

if (petsc_conf_base AND NOT petsc_config_current)
  # Put variables into environment since they are needed to get
  # configuration (petscvariables) in the PETSc makefile
  set (ENV{PETSC_DIR} ${PETSC_DIR})
  set (ENV{PETSC_ARCH} ${PETSC_ARCH})

  # A temporary makefile to probe the PETSc configuration
  set (petsc_config_makefile ${PROJECT_BINARY_DIR}/Makefile.petsc)
  file (WRITE ${petsc_config_makefile}
"# Retrive various flags from PETSc settings.

PETSC_DIR=${PETSC_DIR}
PETSC_ARCH=${PETSC_ARCH}

include ${petsc_conf_base}/variables

get_petsc_include:
	-@echo -I\${PETSC_DIR}/\${PETSC_ARCH}/include -I\${PETSC_DIR}/include \${MPI_INCLUDE} \${X11_INCLUDE}
get_petsc_libs:
	-@echo   \${C_SH_LIB_PATH} -L\${PETSC_LIB_DIR} \${PETSC_LIB_BASIC} \${X11_LIB}
get_petsc_cc:
	-@echo \${PCC}
get_petsc_ld:
	-@echo \${PCC_LINKER}
get_petsc_lib_dir:
	-@echo \${PETSC_LIB_DIR}
get_petsc_cpp_flags:
	-@echo \${PETSC_CCPPFLAGS}
")

  macro (PETSC_GET_VARIABLE name var)
    set (${var} "NOTFOUND" CACHE INTERNAL "Cleared" FORCE)
    execute_process (COMMAND ${CMAKE_BUILD_TOOL} -f ${petsc_config_makefile} ${name}
      OUTPUT_VARIABLE ${var} RESULT_VARIABLE petsc_return OUTPUT_STRIP_TRAILING_WHITESPACE)
  endmacro (PETSC_GET_VARIABLE)
  petsc_get_variable (get_petsc_lib_dir         petsc_lib_dir)
  petsc_get_variable (get_petsc_libs            petsc_libs)
  petsc_get_variable (get_petsc_include         petsc_include_dir)
  petsc_get_variable (get_petsc_cc              petsc_cc)
  petsc_get_variable (get_petsc_cpp_flags       petsc_cpp_flags)

  # We are done with the temporary Makefile, calling PETSC_GET_VARIABLE after this point is invalid
  file (REMOVE ${petsc_config_makefile})

  # Extract include paths and libraries from compile command line
  include (ResolveCompilerPaths)
  resolve_includes (petsc_includes_all "${petsc_cpp_flags}")

  macro (PETSC_FIND_LIBRARY suffix name)
    set (PETSC_LIBRARY_${suffix} "NOTFOUND" CACHE INTERNAL "Cleared" FORCE)
    find_library (PETSC_LIBRARY_${suffix} NAMES ${name} HINTS ${petsc_lib_dir})
    set (PETSC_LIBRARIES_${suffix} "${PETSC_LIBRARY_${suffix}}")
    mark_as_advanced (PETSC_LIBRARY_${suffix})
  endmacro (PETSC_FIND_LIBRARY suffix name)
  petsc_find_library (SYS  petsc)
  #petsc_find_library (VEC  petscvec)
  #petsc_find_library (MAT  petscmat)
  #petsc_find_library (DM   petscdm)
  #petsc_find_library (KSP  petscksp)
  #petsc_find_library (SNES petscsnes)
  #petsc_find_library (TS   petscts)

  # Try compiling and running test program
  include(CheckCXXSourceRuns)
  set(CMAKE_REQUIRED_INCLUDES  ${petsc_includes_all})
  set(CMAKE_REQUIRED_LIBRARIES ${petsc_libs})
  check_cxx_source_runs("
#include \"petscts.h\"
int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  TS ts;

  ierr = PetscInitializeNoArguments();CHKERRQ(ierr);
  //ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSDestroy(ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
" PETSC_TEST_RUNS)

if(NOT PETSC_TEST_RUNS)
  message("PETSc was found but a test program could not be run.")
else(NOT PETSC_TEST_RUNS)
  message("PETSc was found and test program succeeded.")
  set(PETSC_FOUND 1)
  include_directories(${petsc_includes_all})
  add_definitions(-DHAS_PETSC)
endif(NOT PETSC_TEST_RUNS)

  macro (PETSC_JOIN libs deps)
    list (APPEND PETSC_LIBRARIES_${libs} ${PETSC_LIBRARIES_${deps}})
  endmacro (PETSC_JOIN libs deps)
  petsc_join (VEC  SYS)
  petsc_join (MAT  VEC)
  petsc_join (DM   MAT)
  petsc_join (KSP  DM)
  petsc_join (SNES KSP)
  petsc_join (TS   SNES)
  petsc_join (ALL  TS)

  set (PETSC_DEFINITIONS "-D__SDIR__=\"\"" CACHE STRING "PETSc definitions" FORCE)
  set (PETSC_MPIEXEC ${petsc_mpiexec} CACHE FILEPATH "Executable for running PETSc MPI programs" FORCE)
  set (PETSC_INCLUDES ${petsc_includes_all} CACHE STRING "PETSc include path" FORCE)
  set (PETSC_LIBRARIES ${petsc_libs} CACHE STRING "PETSc libraries" FORCE)
  set (PETSC_COMPILER ${petsc_cc} CACHE FILEPATH "PETSc compiler" FORCE)
  # Note that we have forced values for all these choices.  If you
  # change these, you are telling the system to trust you that they
  # work.  It is likely that you will end up with a broken build.
  mark_as_advanced (PETSC_INCLUDES PETSC_LIBRARIES PETSC_COMPILER PETSC_DEFINITIONS PETSC_MPIEXEC PETSC_EXECUTABLE_RUNS)
endif (petsc_conf_base AND NOT petsc_config_current)

