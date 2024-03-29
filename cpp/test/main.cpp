#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <dolfinx/common/log.h>
#include <mpi.h>

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);

  // Parallel tests require MPI initialization before any tests run and
  // termination only after all tests complete.
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}
