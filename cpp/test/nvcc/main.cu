// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

// Checks that DOLFINx headers compile with nvcc, for CUDA code built
// against a host-only DOLFINx. Device code is compiled but not run, so
// no GPU is needed.

#include <basix/finite-element.h>
#include <cstddef>
#include <cstdint>
#include <dolfinx.h>
#include <memory>
#include <mpi.h>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace dolfinx;

// Explicit instantiation compiles every member through nvcc's front
// end, not only those used below.
template class mesh::Mesh<double>;
template class mesh::Geometry<double>;
template class mesh::MeshTags<std::int32_t>;
template class fem::CoordinateElement<double>;
template class fem::FiniteElement<double>;
template class fem::FunctionSpace<double>;
template class fem::Function<double>;
template class fem::Constant<double>;
template class fem::Form<double>;
template class fem::DirichletBC<double>;
template class fem::Expression<double>;
template class la::Vector<double>;
template class la::MatrixCSR<double>;
template class geometry::BoundingBoxTree<double>;

/// Device kernel, so headers also pass through nvcc's device front end.
__global__ void axpy(int n, double a, const double* x, double* y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] += a * x[i];
}

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  {
    auto mesh
        = std::make_shared<mesh::Mesh<double>>(mesh::create_rectangle<double>(
            MPI_COMM_WORLD, {{{0.0, 0.0}, {1.0, 1.0}}}, {8, 8},
            mesh::CellType::triangle));

    basix::FiniteElement element = basix::create_element<double>(
        basix::element::family::P, basix::cell::type::triangle, 1,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);
    auto V = std::make_shared<fem::FunctionSpace<double>>(
        fem::create_functionspace<double>(
            mesh, std::make_shared<fem::FiniteElement<double>>(
                      element, mesh->geometry().dim())));

    fem::Function<double> u(V);
    u.interpolate(
        [](auto x) -> std::pair<std::vector<double>, std::vector<std::size_t>>
        {
          std::vector<double> f(x.extent(1));
          for (std::size_t p = 0; p < x.extent(1); ++p)
            f[p] = x(0, p) + x(1, p);
          return {std::move(f), {f.size()}};
        });

    // Map cells of a sub-mesh to the parent mesh, and back
    const int tdim = mesh->topology()->dim();
    std::vector<std::int32_t> cells
        = mesh::locate_entities(*mesh, tdim,
                                [](auto x)
                                {
                                  std::vector<std::int8_t> marker(x.extent(1));
                                  for (std::size_t p = 0; p < x.extent(1); ++p)
                                    marker[p] = x(0, p) <= 0.5;
                                  return marker;
                                });
    auto [submesh, cell_map, vertex_map, x_map]
        = mesh::create_submesh(*mesh, tdim, cells);
    std::int32_t num_subcells
        = submesh.topology()->index_map(tdim)->size_local();
    std::vector<std::int32_t> subcells(num_subcells);
    std::iota(subcells.begin(), subcells.end(), 0);
    std::vector<std::int32_t> parent_cells = cell_map.sub_topology_to_topology(
        std::span<const std::int32_t>(subcells), false);
    std::vector<std::int32_t> subcells_r = cell_map.sub_topology_to_topology(
        std::span<const std::int32_t>(parent_cells), true);
    if (subcells_r != subcells)
      throw std::runtime_error("EntityMap round trip failed.");
  }
  MPI_Finalize();

  return 0;
}
