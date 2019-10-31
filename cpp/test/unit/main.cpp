#define CATCH_CONFIG_RUNNER
#include "catch/catch.hpp"
#include <mpi.h>

int main(int argc, char* argv[])
{
  // Parallel tests require MPI initialization before any tests run and
  // termination only after all tests complete.
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();

  return result;
}
