#include "real.h"
#include <cmath>
#include <dolfinx.h>

using namespace dolfinx;
using T = PetscScalar;

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  {
    // Create mesh and function space
    auto mesh = std::make_shared<mesh::Mesh>(mesh::create_rectangle(
        MPI_COMM_WORLD, {{{0.0, 0.0}, {2.0, 1.0}}}, {16, 16},
        mesh::CellType::triangle, mesh::GhostMode::shared_facet));

    auto V = std::make_shared<fem::FunctionSpace>(
        fem::create_functionspace(functionspace_form_real_a, "u", mesh));
    std::cout << std::endl << V->dofmap()->list().num_nodes() << std::endl;
  }

  PetscFinalize();

  return 0;
}
