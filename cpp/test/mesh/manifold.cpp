// Copyright (C) 2024 Paul T. Kühner
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "dolfinx/mesh/generation.h"
#include <algorithm>
#include <basix/finite-element.h>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <iterator>
#include <mpi.h>
#include <numeric>
#include <ostream>
#include <vector>

using namespace dolfinx;

//          (3)
//           |
//           |[2]
//           |
//  (1)-----(0)-----(2) ----- (4)
//      [0]     [1]      [3]
TEST_CASE("debug")
{
  // spdlog::set_level(spdlog::level::debug);
  using T = double;
  MPI_Comm comm = MPI_COMM_WORLD;

  std::vector<std::int64_t> cells;
  std::vector<double> x;
  if (dolfinx::MPI::rank(comm) == 0)
  {
    cells = {0, 1, 0, 2, 0, 3, 2, 4};
    x = {0, 0, -1, 0, 1, 0, 0, 1, 2, 0};
  }
  auto element
      = std::make_shared<basix::FiniteElement<double>>(basix::create_element<T>(
          basix::element::family::P, basix::cell::type::interval, 1,
          basix::element::lagrange_variant::unset,
          basix::element::dpc_variant::unset, false));
  fem::CoordinateElement<T> cmap(element);

  auto branching_manifold_boundary_v_fn
      = [](const std::vector<mesh::CellType>& celltypes,
           const std::vector<fem::ElementDofLayout>& doflayouts,
           const std::vector<std::vector<int>>& ghost_owners,
           std::vector<std::vector<std::int64_t>>& cells,
           std::vector<std::vector<std::int64_t>>& cells_v,
           std::vector<std::vector<std::int64_t>>& original_idx)
      -> std::vector<std::int64_t>
  {
    std::vector<std::int64_t> all_vertices;
    // TODO: reserve!
    for (std::size_t i = 0; i < celltypes.size(); i++)
    {
      int num_cell_vertices = mesh::num_cell_vertices(celltypes[i]);
      // std::cout << num_cell_vertices << std::endl;
      // std::cout << "num ghosts: " << ghost_owners[i].size() << std::endl;
      std::size_t num_owned_cells
          = cells_v[i].size() / num_cell_vertices - ghost_owners[i].size();
      // std::cout << "owned:" << num_owned_cells << std::endl;
      all_vertices.insert(
          all_vertices.end(), cells_v[i].begin(),
          std::next(cells_v[i].begin(), num_owned_cells * num_cell_vertices));
    }

    std::ranges::sort(all_vertices);
    auto [unique_end, range_end] = std::ranges::unique(all_vertices);
    all_vertices.erase(unique_end, range_end);
    return all_vertices;
  };

  auto rank = dolfinx::MPI::rank(comm);
  // std::cout << "num vertices: " << x.size() / 2 << std::endl;
  auto my_partitioner
      = [&](MPI_Comm comm, int nparts,
            const std::vector<dolfinx::mesh::CellType>& cell_types,
            const std::vector<std::span<const std::int64_t>>& cells)
  {
    if (rank == 0)
    {
      std::vector<std::vector<std::int32_t>> list{{{{0}}, {{1}}, {{0}}, {{1}}}};
      return graph::AdjacencyList<std::int32_t>(list);
    }
    return graph::AdjacencyList<std::int32_t>(0);
  };
  auto mesh = dolfinx::mesh::create_mesh(
      comm, rank == 0 ? MPI_COMM_SELF : MPI_COMM_NULL, cells, {cmap},
      rank == 0 ? MPI_COMM_SELF : MPI_COMM_NULL, x, {x.size() / 2, 2},
      // mesh::create_cell_partitioner(mesh::GhostMode::shared_facet),
      my_partitioner, branching_manifold_boundary_v_fn);

  mesh.topology()->create_connectivity(0, 1);
  auto v_to_c = mesh.topology()->connectivity(0, 1);
  // std::cout << dolfinx::MPI::rank(comm) << " has " << v_to_c->num_nodes()
  //           << " vertices" << std::endl;
  for (int i = 0; i < v_to_c->num_nodes(); i++)
  {
    std::cout << "[" << i << "] ";
    for (auto link : v_to_c->links(i))
      std::cout << link << ", ";
    std::cout << "\n";
  }
  std::cout << std::endl;
  // CHECK(false);
}