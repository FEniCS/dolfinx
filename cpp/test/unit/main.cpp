#define CATCH_CONFIG_RUNNER
#include "catch/catch.hpp"
#include <mpi.h>

int main(int argc, char* argv[])
{

  // Parallel tests require MPI_Init and MPI_Finalize before and after the test
  // framework
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();

  return result;
}
