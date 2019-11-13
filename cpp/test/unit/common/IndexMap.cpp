// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch.hpp>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <numeric>
#include <set>
#include <vector>

using namespace dolfin;

namespace
{
void test_scatter_fwd()
{
  const int mpi_size = dolfin::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfin::MPI::rank(MPI_COMM_WORLD);
  const int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  // Create an IndexMap
  common::IndexMap idx_map(MPI_COMM_WORLD, size_local, ghosts, 1);

  // Create some data to scatter
  const std::int64_t val = 11;
  std::vector<std::int64_t> data_local(size_local, val * mpi_rank);
  std::vector<std::int64_t> data_ghost(num_ghosts, -1);

  // Scatter values to ghost and check value is correctly received
  idx_map.scatter_fwd(data_local, data_ghost, 1);
  CHECK(std::all_of(data_ghost.begin(), data_ghost.end(), [=](auto i) {
    return i == val * ((mpi_rank + 1) % mpi_size);
  }));

  // Test block of values
  const int n = 7;
  std::vector<std::int64_t> data_local_n(n * size_local, val * mpi_rank);
  std::vector<std::int64_t> data_ghost_n(n * num_ghosts, -1);

  // Scatter values to ghost and check value is correctly received
  idx_map.scatter_fwd(data_local_n, data_ghost_n, n);
  CHECK(data_ghost_n.size() == n * num_ghosts);
  CHECK(std::all_of(data_ghost_n.begin(), data_ghost_n.end(), [=](auto i) {
    return i == val * ((mpi_rank + 1) % mpi_size);
  }));
}

void test_scatter_rev()
{
  const int mpi_size = dolfin::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfin::MPI::rank(MPI_COMM_WORLD);
  const int size_local = 100;

  // Create some ghost entries on next process
  const int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::int64_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  // Create an IndexMap
  common::IndexMap idx_map(MPI_COMM_WORLD, size_local, ghosts, 1);

  // Create some data, setting ghost values
  std::int64_t value = 15;
  std::vector<std::int64_t> data_local(size_local, 0);
  std::vector<std::int64_t> data_ghost(num_ghosts, value);

  // Scatter ghost values back to owner (sum)
  idx_map.scatter_rev(data_local, data_ghost, 1, common::IndexMap::Mode::add);
  std::int64_t sum = std::accumulate(data_local.begin(), data_local.end(), 0);
  CHECK(sum == value * num_ghosts);

  // Test block of values
  const int n = 5;
  std::vector<std::int64_t> data_local_n(n * size_local, 0);
  std::vector<std::int64_t> data_ghost_n(n * num_ghosts, value);
  idx_map.scatter_rev(data_local_n, data_ghost_n, n,
                      common::IndexMap::Mode::add);

  CHECK(data_local_n.size() == n * size_local);
  std::int64_t sum_n
      = std::accumulate(data_local_n.begin(), data_local_n.end(), 0);
  CHECK(sum_n == n * value * num_ghosts);
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
