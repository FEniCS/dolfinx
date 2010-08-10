set(UMFPACK_FOUND 0)

message(STATUS "checking for package 'UMFPACK'")

# Check for header file
find_path(UMFPACK_INCLUDE_DIR umfpack.h
  $ENV{UMFPACK_DIR}
  /usr/local/include
  /usr/include
  DOC "Directory where the UMFPACK header is located"
  )

# Try compiling and running test program
if(UMFPACK_INCLUDE_DIR)
  include(CheckCXXSourceRuns)
  set(CMAKE_REQUIRED_INCLUDES ${UMFPACK_INCLUDE_DIR})
  check_cxx_source_runs("
#include <stdio.h>
#include <umfpack.h>

int main() {
  #ifdef UMFPACK_MAIN_VERSION
    #ifdef UMFPACK_SUB_VERSION
      #ifdef UMFPACK_SUBSUB_VERSION
        printf("%d.%d.%d", UMFPACK_MAIN_VERSION,UMFPACK_SUB_VERSION,UMFPACK_SUBSUB_VERSION);
      #else
        printf("%d.%d", UMFPACK_MAIN_VERSION,UMFPACK_SUB_VERSION);
      #endif
    #else
      printf("%d", UMFPACK_MAIN_VERSION);
    #endif
  #endif
  return 0;
}
" UMFPACK_TEST_RUNS)

  if(NOT UMFPACK_TEST_RUNS)
    message("UMFPACK was found but a test program could not be run.")
  endif(NOT UMFPACK_TEST_RUNS)

endif(UMFPACK_INCLUDE_DIR)

# Report results of tests
if(UMFPACK_TEST_RUNS)
  message(STATUS "  found package 'UMFPACK'")
  set(UMFPACK_FOUND 1)
  include_directories(${UMFPACK_INCLUDE_DIR})
  add_definitions(-DHAS_UMFPACK)
else(UMFPACK_TEST_RUNS)
  message(STATUS "  package 'UMFPACK' could not be configured.")
endif(UMFPACK_TEST_RUNS)
