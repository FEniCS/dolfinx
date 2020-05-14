// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch.hpp>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <numeric>
#include <set>
#include <vector>

using namespace dolfinx;

namespace
{
void test_scatter_fwd()
{
  // Block size
  auto n = GENERATE(1, 5, 10);

  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  const int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  common::IndexMap idx_map(MPI_COMM_WORLD, size_local, ghosts,
                           global_ghost_owner, 1);

  // Create some data to scatter
  const std::int64_t val = 11;
  std::vector<std::int64_t> data_local(n * size_local, val * mpi_rank);
  std::vector<std::int64_t> data_ghost(n * num_ghosts, -1);

  // Scatter values to ghost and check value is correctly received
  idx_map.scatter_fwd(data_local, data_ghost, n);
  CHECK(data_ghost.size() == n * num_ghosts);
  CHECK(std::all_of(data_ghost.begin(), data_ghost.end(), [=](auto i) {
    return i == val * ((mpi_rank + 1) % mpi_size);
  }));
}

void test_scatter_rev()
{
  // Block size
  auto n = GENERATE(1, 5, 10);

  const int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  const int size_local = 100;

  // Create some ghost entries on next process
  const int num_ghosts = (mpi_size - 1) * 3;
  Eigen::Array<std::int64_t, Eigen::Dynamic, 1> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  std::vector<int> global_ghost_owner(ghosts.size(), (mpi_rank + 1) % mpi_size);

  // Create an IndexMap
  common::IndexMap idx_map(MPI_COMM_WORLD, size_local, ghosts,
                           global_ghost_owner, 1);

  // Create some data, setting ghost values
  std::int64_t value = 15;
  std::vector<std::int64_t> data_local(n * size_local, 0);
  std::vector<std::int64_t> data_ghost(n * num_ghosts, value);
  idx_map.scatter_rev(data_local, data_ghost, n, common::IndexMap::Mode::add);

  std::int64_t sum;
  CHECK(data_local.size() == n * size_local);
  sum = std::accumulate(data_local.begin(), data_local.end(), 0);
  CHECK(sum == n * value * num_ghosts);

  idx_map.scatter_rev(data_local, data_ghost, n,
                      common::IndexMap::Mode::insert);
  sum = std::accumulate(data_local.begin(), data_local.end(), 0);
  CHECK(sum == n * value * num_ghosts);

  idx_map.scatter_rev(data_local, data_ghost, n, common::IndexMap::Mode::add);
  sum = std::accumulate(data_local.begin(), data_local.end(), 0);
  CHECK(sum == 2 * n * value * num_ghosts);
}
} // namespace

TEST_CASE("Scatter forward using IndexMap", "[index_map_scatter_fwd]")
{
  CHECK_NOTHROW(test_scatter_fwd());
}

TEST_CASE("Scatter reverse using IndexMap", "[index_map_scatter_rev]")
{
  CHECK_NOTHROW(test_scatter_rev());
}
