// Copyright (C) 2018 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch.hpp>
#include <dolfin/common/IndexMap.h>
#include <dolfin/common/MPI.h>
#include <vector>

using namespace dolfin;

namespace
{
void test_scatter()
{
  const int mpi_size = dolfin::MPI::size(MPI_COMM_WORLD);
  const int mpi_rank = dolfin::MPI::rank(MPI_COMM_WORLD);

  const int size_local = 100;

  // Create some ghost entries on next process
  int num_ghosts = (mpi_size - 1) * 3;
  std::vector<std::size_t> ghosts(num_ghosts);
  for (int i = 0; i < num_ghosts; ++i)
    ghosts[i] = (mpi_rank + 1) % mpi_size * size_local + i;

  // Create an IndexMap with some local data
  const std::int64_t val = 11;
  std::vector<std::int64_t> data_local(size_local, val * mpi_rank);
  std::vector<std::int64_t> data_ghost(num_ghosts, -1);
  common::IndexMap idx_map(MPI_COMM_WORLD, size_local, ghosts, 1);

  // Scatter values to ghost and check value is correctly received
  idx_map.scatter_fwd(data_local, data_ghost);
  for (std::int64_t data : data_ghost)
  {
    if (data == val * ((mpi_rank + 1) % mpi_size))
      continue;
    else
      throw std::runtime_error("Received data incorrect.");
  }

} // namespace
} // namespace

TEST_CASE("Scatter using IndexMap", "[index_map_scatter]")
{
  CHECK_NOTHROW(test_scatter());
}
