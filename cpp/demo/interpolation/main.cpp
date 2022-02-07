#include <algorithm>
#include <cmath>

#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>

#include "interpolation.h"

using namespace dolfinx;

int main(int argc, char *argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  std::string thread_name = "RANK " + std::to_string(mpi_rank);
  loguru::set_thread_name(thread_name.c_str());
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0) {
      loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
  }

  {
    auto meshL = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {16, 16, 16},
        mesh::CellType::tetrahedron, mesh::GhostMode::shared_facet));

    auto meshR = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
        MPI_COMM_WORLD, {{{0.25, 0.25, 0.25}, {0.75, 0.75, 0.75}}},
        {16, 16, 16}, mesh::CellType::tetrahedron,
        mesh::GhostMode::shared_facet));

    auto VL = fem::create_functionspace(functionspace_form_interpolation_a, "u",
                                        meshL);
    auto VR = fem::create_functionspace(functionspace_form_interpolation_a, "u",
                                        meshR);

    auto uL = std::make_shared<fem::Function<PetscScalar>>(VL);
    auto uR = std::make_shared<fem::Function<PetscScalar>>(VR);
    auto f = std::make_shared<fem::Constant<PetscScalar>>(1);

    uL->interpolate([](auto& x) {
      xt::xtensor<PetscScalar, 1> r(xt::shape({x.shape()[1]}), 0);
      for (std::size_t i = 0; i < x.shape(1); ++i)
      {
        r(0, i) = std::cos(10 * x(0, i)) * std::sin(10 * x(2, i));
        //              r(0, i) = x(1, i);
      }
      return r;
    });

    fem::interpolate(*uR, *uL);

    io::XDMFFile fileL(meshL->mpi_comm(), "uL.xdmf", "w");
    fileL.write_mesh(*meshL);
    fileL.write_function(*uL, 0);
    fileL.close();

    io::XDMFFile fileR(meshR->mpi_comm(), "uR.xdmf", "w");
    fileR.write_mesh(*meshR);
    fileR.write_function(*uR, 0);
    fileR.close();
  }

  common::subsystem::finalize_petsc();

  return 0;
}
