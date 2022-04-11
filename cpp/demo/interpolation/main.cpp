#include <algorithm>
#include <cmath>

#include <basix/e-lagrange.h>

#include <dolfinx.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/mesh/generation.h>

using namespace dolfinx;

void interpolation_different_meshes()
{
  const std::array<std::size_t, 3> subdivisions = {5, 5, 5};

  auto meshL = std::make_shared<mesh::Mesh>(mesh::create_box(
      MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}}, subdivisions,
      mesh::CellType::tetrahedron, mesh::GhostMode::shared_facet));

  auto meshR = std::make_shared<mesh::Mesh>(mesh::create_box(
      MPI_COMM_WORLD, {{{0.1, 0.1, 0.1}, {1.1, 1.1, 1.1}}}, subdivisions,
      mesh::CellType::tetrahedron, mesh::GhostMode::shared_facet));

  basix::FiniteElement eL = basix::element::create_lagrange(
      mesh::cell_type_to_basix_type(meshL->topology().cell_type()), 1,
      basix::element::lagrange_variant::equispaced, false);
  auto VL = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(meshL, eL, 3));

  basix::FiniteElement eR = basix::element::create_lagrange(
      mesh::cell_type_to_basix_type(meshR->topology().cell_type()), 1,
      basix::element::lagrange_variant::equispaced, false);
  auto VR = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(meshR, eR, 3));

  auto uL = std::make_shared<fem::Function<PetscScalar>>(VL);
  auto uR = std::make_shared<fem::Function<PetscScalar>>(VR);

  auto fun = [](auto& x)
  {
    auto r = xt::zeros_like(x);
    for (std::size_t i = 0; i < x.shape(1); ++i)
    {
      r(0, i) = std::cos(10 * x(0, i)) * std::sin(10 * x(2, i));
      r(1, i) = std::sin(10 * x(0, i)) * std::sin(10 * x(2, i));
      r(2, i) = std::cos(10 * x(0, i)) * std::cos(10 * x(2, i));
    }
    return r;
  };

  uL->interpolate(fun);

  uR->interpolate(*uL);

  auto uR_ex = std::make_shared<fem::Function<PetscScalar>>(VR);
  uR_ex->interpolate(fun);

  const PetscReal diffNorm = std::sqrt(std::transform_reduce(
      uR->x()->array().cbegin(), uR->x()->array().cend(),
      uR_ex->x()->array().cbegin(), static_cast<PetscReal>(0), std::plus<>(),
      [](const auto& a, const auto& b)
      { return std::real((a - b) * std::conj(a - b)); }));

  if (diffNorm > 1e-13)
  {
    throw std::runtime_error("Interpolation on different meshes failed.");
  }
}

int main(int argc, char* argv[])
{
  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  std::string thread_name = "RANK " + std::to_string(mpi_rank);
  loguru::set_thread_name(thread_name.c_str());
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
  {
    loguru::g_stderr_verbosity = loguru::Verbosity_INFO;
  }

  interpolation_different_meshes();

  PetscFinalize();

  return 0;
}
