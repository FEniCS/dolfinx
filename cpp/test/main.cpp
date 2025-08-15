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

  // Configure test session
  auto session = Catch::Session();

  // Default order is for Catch version >= 3.9.0 changed to random.
  // MPI testing requires same order of execution across processes.
  session.configData().runOrder = Catch::TestRunOrder::Declared;

  // Run test session.
  int result = session.run(argc, argv);

  MPI_Finalize();
  return result;
}
