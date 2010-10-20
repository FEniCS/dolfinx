# - Try to find ParMETIS
# Once done this will define
#
#  PARMETIS_FOUND        - system has ParMETIS
#  PARMETIS_INCLUDE_DIRS - include directories for ParMETIS
#  PARMETIS_LIBRARIES    - libraries for ParMETIS

if (MPI_FOUND)
  find_path(PARMETIS_INCLUDE_DIRS parmetis.h
    HINTS ${PARMETIS_DIR}/include $ENV{PARMETIS_DIR}/include
    DOC "Directory where the ParMETIS header files are located"
  )

  find_library(PARMETIS_LIBRARY parmetis
    HINTS ${PARMETIS_DIR}/lib $ENV{PARMETIS_DIR}/lib
    DOC "Directory where the ParMETIS library is located"
  )

  find_library(METIS_LIBRARY metis
    HINTS ${PARMETIS_DIR}/lib $ENV{PARMETIS_DIR}/lib
    DOC "Directory where the METIS library is located"
  )

  set(PARMETIS_LIBRARIES ${PARMETIS_LIBRARY} ${METIS_LIBRARY})

  # Try compiling and running test program
  if (PARMETIS_INCLUDE_DIRS AND PARMETIS_LIBRARY AND METIS_LIBRARY)

    # Set flags for building test program
    set(CMAKE_REQUIRED_INCLUDES ${PARMETIS_INCLUDE_DIRS} ${MPI_INCLUDE_PATH})
    set(CMAKE_REQUIRED_LIBRARIES ${PARMETIS_LIBRARIES}  ${MPI_LIBRARIES})

    # Build and run test program
    include(CheckCXXSourceRuns)
    check_cxx_source_runs("
#include <mpi.h>
#include <parmetis.h>

int main()
{
  // FIXME: Find a simple but sensible test for ParMETIS

  // Initialise MPI
  MPI::Init();

  // Finalize MPI
  MPI::Finalize();

  return 0;
}
" PARMETIS_TEST_RUNS)

  endif()
endif()

# Standard package handling
find_package_handle_standard_args(ParMETIS
                                  "ParMETIS could not be found/configured."
                                  PARMETIS_LIBRARIES
                                  PARMETIS_TEST_RUNS
                                  PARMETIS_INCLUDE_DIRS)
