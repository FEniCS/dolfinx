// Copyright (C) 2025 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#ifdef HAS_ADIOS2

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/utils.h>
#include <vector>

using namespace dolfinx;

namespace
{
common::IndexMap create_index_map(MPI_Comm comm, int size_local, int num_ghosts)
{
  const int mpi_size = dolfinx::MPI::size(comm);
  const int mpi_rank = dolfinx::MPI::rank(comm);

  // Create some ghost entries on next process
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  return common::IndexMap(MPI_COMM_WORLD, size_local, ghosts,
                          global_ghost_owner);
}

void test_adjacency_list_create()
{
  std::vector<std::int32_t> edges{1, 2, 0, 0, 1};
  std::vector<std::int32_t> offsets{0, 2, 3, 5};
  graph::AdjacencyList g0(edges, offsets);

  CHECK(std::ranges::equal(g0.links(0), std::vector<std::int32_t>{1, 2}));
  CHECK(std::ranges::equal(g0.links(1), std::vector<std::int32_t>{0}));
  CHECK(std::ranges::equal(g0.links(2), std::vector<std::int32_t>{0, 1}));

  std::vector<std::int64_t> node_data{-1, 5, -20};
  graph::AdjacencyList g1(edges, offsets, node_data);
  CHECK(std::ranges::equal(g1.node_data().value(), node_data));
}

void test_comm_graphs()
{
  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  constexpr int size_local = 100;
  const common::IndexMap idx_map
      = create_index_map(MPI_COMM_WORLD, size_local, (mpi_size - 1) * 3);

  auto g = graph::comm_graph(idx_map);
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
    graph::comm_to_json(g);
}
} // namespace

TEST_CASE("AdjacencyList create")
{
  CHECK_NOTHROW(test_adjacency_list_create());
}

TEST_CASE("IndexMap communication graph", "[index_map_graph]")
{
  CHECK_NOTHROW(test_comm_graphs());
}

#endif
