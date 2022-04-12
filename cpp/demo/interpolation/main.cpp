#include "interpolation.h"
#include <algorithm>
#include <basix/e-lagrange.h>
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/ADIOS2Writers.h>
#include <dolfinx/mesh/generation.h>

using namespace dolfinx;

void interpolation_different_meshes()
{
  const std::array<std::size_t, 3> subdivisions = {20, 20, 20};

  auto meshL = std::make_shared<mesh::Mesh>(mesh::create_box(
      MPI_COMM_WORLD, {{{0, 0, 0}, {1, 1, 1}}}, subdivisions,
      mesh::CellType::tetrahedron, mesh::GhostMode::shared_facet));

  auto meshR = std::make_shared<mesh::Mesh>(mesh::create_box(
      MPI_COMM_WORLD, {{{0.5, 0.5, 0.5}, {1.5, 1.5, 1.5}}}, {11, 12, 15},
      mesh::CellType::hexahedron, mesh::GhostMode::shared_facet));

  basix::FiniteElement eL = basix::element::create_lagrange(
      mesh::cell_type_to_basix_type(meshL->topology().cell_type()), 1,
      basix::element::lagrange_variant::equispaced, false);
  auto VL = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(meshL, eL, 3));

  basix::FiniteElement eR = basix::element::create_lagrange(
      mesh::cell_type_to_basix_type(meshR->topology().cell_type()), 2,
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

  io::VTXWriter writeruL(meshL->comm(), "uL.bp", {uL});
  writeruL.write(0.0);

  io::VTXWriter writeruR(meshR->comm(), "uR.bp", {uR});
  writeruR.write(0.0);
}

int main(int argc, char* argv[])
{
  dolfinx::init_logging(argc, argv);
  MPI_Init(&argc, &argv);

  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  interpolation_different_meshes();

  MPI_Finalize();

  return 0;
}