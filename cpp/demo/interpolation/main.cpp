#include <algorithm>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/generation.h>

#include "interpolation.h"

using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_petsc(argc, argv);

  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  std::string thread_name = "RANK " + std::to_string(mpi_rank);
  loguru::set_thread_name(thread_name.c_str());
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
  {
    loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
  }

  {
    const std::array<std::size_t, 3> subdivisions = {5, 5, 5};

    auto meshL = std::make_shared<mesh::Mesh>(mesh::create_box(
        MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}}, subdivisions,
        mesh::CellType::tetrahedron, mesh::GhostMode::shared_facet));

    auto meshR = std::make_shared<mesh::Mesh>(mesh::create_box(
        MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}}, subdivisions,
        mesh::CellType::tetrahedron, mesh::GhostMode::shared_facet));

    auto VL = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(
        functionspace_form_interpolation_a, "u", meshL));
    auto VR = std::make_shared<fem::FunctionSpace>(fem::create_functionspace(
        functionspace_form_interpolation_a, "u", meshR));

    auto uL = std::make_shared<fem::Function<PetscScalar>>(VL);
    auto uR = std::make_shared<fem::Function<PetscScalar>>(VR);

    uL->interpolate(
        [](auto& x)
        {
          auto r = xt::zeros_like(x);
          for (std::size_t i = 0; i < x.shape(1); ++i)
          {
            r(0, i) = std::cos(10 * x(0, i)) * std::sin(10 * x(2, i));
            r(1, i) = std::sin(10 * x(0, i)) * std::sin(10 * x(2, i));
            r(2, i) = std::cos(10 * x(0, i)) * std::cos(10 * x(2, i));
          }
          return r;
        });

    common::Timer timer("Interpolating");
    uR->interpolate(*uL);
    timer.stop();
    list_timings(MPI_COMM_WORLD, {TimingType::wall});

    auto uR_ex = std::make_shared<fem::Function<PetscScalar>>(VR);
    uR_ex->interpolate(
        [](auto& x)
        {
          auto r = xt::zeros_like(x);
          for (std::size_t i = 0; i < x.shape(1); ++i)
          {
            r(0, i) = std::cos(10 * x(0, i)) * std::sin(10 * x(2, i));
            r(1, i) = std::sin(10 * x(0, i)) * std::sin(10 * x(2, i));
            r(2, i) = std::cos(10 * x(0, i)) * std::cos(10 * x(2, i));
          }
          return r;
        });

    la::petsc::Vector _uR(la::petsc::create_vector_wrap(*uR->x()), false);
    la::petsc::Vector _uR_ex(la::petsc::create_vector_wrap(*uR_ex->x()), false);

    VecAXPY(_uR.vec(), -1, _uR_ex.vec());
    PetscReal diffNorm;
    VecNorm(_uR.vec(), NORM_2, &diffNorm);

    LOG(ERROR) << "diffNorm = " << diffNorm;

    LOG(ERROR) << "diffNorm2 = "
               << std::sqrt(std::transform_reduce(
                      uR->x()->array().cbegin(), uR->x()->array().cend(),
                      uR_ex->x()->array().cbegin(), 0, std::plus<>(),
                      [](const auto& a, const auto& b)
                      { return (a - b) * (a - b); }));
  }

  common::subsystem::finalize_petsc();

  return 0;
}
